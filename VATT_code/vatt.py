import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Literal
import numpy as np
import os
from processing import process_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# input 차원 바꾸는 역할
class ModalityProjection(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.projection(x)

# multi modal 벡터가 결합된 벡터를 차원확장 -> 비선형 -> 차원축소를 거쳐 분류 층에 넘길 최종벡터 생성
# 최종벡터 : VATTClassifier 에 전달될 벡터
class MultimodalProjectionHead(nn.Module):
    def __init__(self, dim: int):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(dim, dim * 2), # 차원 확장
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim), # 차원 축소
                nn.LayerNorm(dim)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.projection(x)

class VATTFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 4):
            super().__init__()
            self.modality_embeddings = nn.Parameter(torch.randn(2, dim))
            
            # Transformer Encoder 정의
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim, # 각 토큰의 차원
                nhead=num_heads, 
                dim_feedforward=dim * 4,
                batch_first=True # 입력 텐서의 첫 번째 차원이 배치 차원임을 지정
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
            # 입력 벡터를 시퀀스 길이 1로 변환
            if audio_emb.dim() == 2:  # (batch_size, dim)
                audio_emb = audio_emb.unsqueeze(1)  # (batch_size, 1, dim)
            if text_emb.dim() == 2:  # (batch_size, dim)
                text_emb = text_emb.unsqueeze(1)  # (batch_size, 1, dim)

            # 모달리티 임베딩 추가
            audio_emb = audio_emb + self.modality_embeddings[0]
            text_emb = text_emb + self.modality_embeddings[1]

            # 두 시퀀스 결합
            combined = torch.cat([audio_emb, text_emb], dim=1)  # 결합 후 크기: (batch_size, 2, dim)
            seq_len = combined.size(1)

            # 포지셔널 인코딩 생성
            pos_encoding = self.get_positional_encoding(seq_len, audio_emb.size(-1)).to(combined.device)
            combined = combined + pos_encoding

            # Transformer 적용
            transformed = self.transformer(combined)  # 출력 크기: (batch_size, seq_len, dim)

            # 단일 벡터 반환 (평균 연산 또는 첫 번째 토큰 사용)
            output = transformed.mean(dim=1)  # (batch_size, dim)
            return output

    def get_positional_encoding(self, seq_len: int, dim: int) -> torch.Tensor:
            """
            Sine-Cosine 포지셔널 인코딩을 동적으로 생성
            seq_len: 시퀀스 길이
            dim: 차원 크기
            """
            position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))  # (dim//2,)

            pos_encoding = torch.zeros(seq_len, dim)
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
            
            return pos_encoding.unsqueeze(0)  # (1, seq_len, dim)

class ConcatFusion(nn.Module):
    def __init__(self, dim: int):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(dim * 2, dim), # concat으로 단순 결합된 input(=combined)의 차원(=dim*2)을 dim으로 변환
                nn.LayerNorm(dim),
                nn.GELU()
            )
            
    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
            # 이제 이미 벡터이므로 mean 연산 필요 없음
            # 두 벡터를 마지막 차원 기준으로 결합
            # 결합된 벡터의 크기는 (batch_size, dim * 2)
            combined = torch.cat([audio_emb, text_emb], dim=-1)
            return self.projection(combined)

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int):
            super().__init__()
            self.audio_attn = nn.Linear(dim, dim)
            self.text_attn = nn.Linear(dim, dim)
            self.fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
            
    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
            audio_attended = self.audio_attn(audio_emb)
            text_attended = self.text_attn(text_emb)
            combined = torch.cat([audio_attended, text_attended], dim=-1)
            return self.fusion(combined)

# Late Fusion : 서로 다른 모달리티의 예측 결과를 결합하는 방식
# 오디오와 텍스트 임베딩을 독립적으로 처리하고, 각 모달의 예측을 결합해서 최종예측을 만듦
class LateFusion(nn.Module):
    def __init__(self, dim: int, num_classes: int):
            super().__init__()
            self.audio_classifier = nn.Linear(dim, num_classes)
            self.text_classifier = nn.Linear(dim, num_classes)
            # 두 모달리티에 대해 동일한 가중치(0.5)가 부여
            # self.weights 초기값 : [0.5 0.5]
            # nn.Parameter로 지정했으므로 learnable parameter
            self.weights = nn.Parameter(torch.ones(2) / 2) 
            #self.weights = nn.Parameter([0.3, 0.7]) 
            
    def forward(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
            # 이제 이미 벡터이므로 mean 연산 필요 없음
            audio_pred = self.audio_classifier(audio_emb) # 오디오 embedding을 활용한 예측
            text_pred = self.text_classifier(text_emb) # text embedding을 활용한 예측
            weights = F.softmax(self.weights, dim=0) # self.weights : 오디오 기반 예측 & 텍스트 기반 예측 간의 가중치 조정 파라미터
            return weights[0] * audio_pred + weights[1] * text_pred

class VATTClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int = 5):
            super().__init__()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, num_classes)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp_head(x)

class VATT(nn.Module):
    def __init__(
            self,
            audio_input_dim: Optional[int],
            text_input_dim: Optional[int],
            feature_dim: int = 256,
            fusion_type: Literal['transformer', 'concat', 'cross_attention', 'late'] = 'transformer',
            modality: Literal['both', 'audio_only', 'text_only'] = 'both',
            num_classes: int = 5
        ):
            super().__init__()
            self.modality = modality
            self.fusion_type = fusion_type
            
            if modality in ['both', 'audio_only']:
                self.audio_input_proj = ModalityProjection(audio_input_dim, feature_dim)
            if modality in ['both', 'text_only']:
                self.text_input_proj = ModalityProjection(text_input_dim, feature_dim)
            
            if modality == 'both':
                if fusion_type == 'transformer':
                    self.fusion = VATTFusion(feature_dim)
                elif fusion_type == 'concat':
                    self.fusion = ConcatFusion(feature_dim)
                elif fusion_type == 'cross_attention':
                    self.fusion = CrossAttentionFusion(feature_dim)
                elif fusion_type == 'late':
                    self.fusion = LateFusion(feature_dim, num_classes)
            
            # late_fusion : late_fusion을 통해 num_classifier 차원으로 변환 일어남
            # 따라서 late_fusion인 경우 MultimodalProjection 작업 필요 없음
            if fusion_type != 'late':
                self.projection_head = MultimodalProjectionHead(feature_dim)
                self.classifier = VATTClassifier(feature_dim, num_classes)
        
    def forward(
            self,
            audio_input: Optional[torch.Tensor] = None,
            text_input: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
            audio_emb, text_emb = None, None
            
            # if self.modality in ['both', 'audio_only'] and audio_input is not None:
            #     # ModalityProjection으로 audio_input_dim -> feature_dim
            #     audio_emb = self.audio_input_proj(audio_input)

            if self.modality in ['both', 'audio_only'] and audio_input is not None:
                # 여기에서 audio_input_dim -> feature_dim 변환
                audio_emb = self.audio_input_proj(audio_input)
                audio_emb = audio_emb * 0.5 # 오디오 임베딩 기여도 축소

            if self.modality in ['both', 'text_only'] and text_input is not None:
                # ModalityProjection으로 text_input_dim -> feature_dim
                text_emb = self.text_input_proj(text_input)
            
            # modal이 하나인 경우 feature는 단일 모달의 embedding만 사용
            if self.modality == 'audio_only':
                features = audio_emb
            elif self.modality == 'text_only':
                features = text_emb

            # modal이 두 개인 경우
            else:
                # fusion_type이 late인 경우
                # 즉 두 모달을 모두 사용하지만 예측에 대한 임베딩 벡터는 각 모달 별로 추출하고 이를 마지막에 결합하는 경우
                if self.fusion_type == 'late':
                    logits = self.fusion(audio_emb, text_emb)
                    return {'logits': logits}
                # fusion_type == 'VATTFusion' ,'ConcatFusion', 'CrossAttentionFusion'
                else:
                    features = self.fusion(audio_emb, text_emb)
            
            # self.projection_head = MultimodalProjectionHead(feature_dim)
            # projected : 두 모달의 벡터가 결합된 특징 벡터
            projected = self.projection_head(features)
            logits = self.classifier(projected)         
            outputs = {'logits': logits, 'fused_features': projected}
            
            if self.modality == 'both' and self.fusion_type != 'late':
                outputs.update({
                    # late가 아닐 때, 두 모달의 차원을 미리 맞춰줄 필요가 있기 때문에 projection_head 필요
                    # late는 마지막에 num_class로 차원을 동일하게 만들기 때문에 두 모달의 벡터를 미리 맞춰줄 필요 없음
                    'audio_proj': self.projection_head(audio_emb),
                    'text_proj': self.projection_head(text_emb)
                })
            
            return outputs

# 오디오와 텍스트의 유사한 특성을 학습하는 데 사용됨
def contrastive_loss(audio_proj: torch.Tensor, text_proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        audio_proj = F.normalize(audio_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        # 오디오와 텍스트 간의 유사성 계산
        # text_proj.T : (dim, batch_size)
        # similarity : (batch_size, batch_size) = logit
        similarity = torch.matmul(audio_proj, text_proj.T) / temperature   
        
        # 대조 손실을 계산하는 데 사용되는 라벨을 생성하는 코드
        # 배치 크기에 따른 label이 부여됨       
        labels = torch.arange(similarity.size(0), device=similarity.device)
        
        # similarity 행렬에서 각 행의 값들에 대해 labels에 정의된 텍스트 샘플 인덱스의 값을 예측하도록 만듦
        return F.cross_entropy(similarity, labels)

def create_dataloaders(
        train_audio_embeddings: Optional[torch.Tensor] = None,
        train_text_embeddings: Optional[torch.Tensor] = None,
        train_labels: torch.Tensor = None,
        val_audio_embeddings: Optional[torch.Tensor] = None,
        val_text_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        modality: str = 'both',
        batch_size: int = 32
    ):
        # Create appropriate datasets based on modality
        if modality == 'audio_only':
            train_dataset = torch.utils.data.TensorDataset(train_audio_embeddings, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_audio_embeddings, val_labels) if val_audio_embeddings is not None else None
        elif modality == 'text_only':
            train_dataset = torch.utils.data.TensorDataset(train_text_embeddings, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_text_embeddings, val_labels) if val_text_embeddings is not None else None
        else:
            train_dataset = torch.utils.data.TensorDataset(train_audio_embeddings, train_text_embeddings, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_audio_embeddings, val_text_embeddings, val_labels) if all([val_audio_embeddings is not None, val_text_embeddings is not None]) else None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
        
        return train_loader, val_loader

def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        modality: str,
        fusion_type: str,
        contrastive_weight: float = 0.5
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0
        total_acc = 0
        
        for batch in dataloader:
            if modality == 'audio_only':
                audio, labels = [x.to(device) for x in batch]
                text = None
            elif modality == 'text_only':
                text, labels = [x.to(device) for x in batch]
                audio = None
            else:
                audio, text, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(audio, text)
            
            cls_loss = F.cross_entropy(outputs['logits'], labels)
            
            if modality == 'both' and fusion_type != 'late':
                cont_loss = contrastive_loss(outputs['audio_proj'], outputs['text_proj'])
                loss = cls_loss + contrastive_weight * cont_loss
                #loss = cls_loss 
            else:
                loss = cls_loss
            
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            acc = (preds == labels).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        return total_loss / len(dataloader), total_acc / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, device: str, modality: str) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0
    total_acc = 0
    all_preds = []  # Initialize empty list for predictions
    all_labels = []  # Initialize empty list for ground-truth labels

    with torch.no_grad():
        for batch in dataloader:
            if modality == 'audio_only':
                audio, labels = [x.to(device) for x in batch]
                text = None
            elif modality == 'text_only':
                text, labels = [x.to(device) for x in batch]
                audio = None
            else:
                audio, text, labels = [x.to(device) for x in batch]

            # Forward pass
            outputs = model(audio_input=audio, text_input=text)
            loss = F.cross_entropy(outputs['logits'], labels)
            preds = torch.argmax(outputs['logits'], dim=1)

            # Append predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            acc = (preds == labels).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()

    # Check if `all_preds` and `all_labels` are not empty
    if len(all_preds) == 0 or len(all_labels) == 0:
        raise ValueError("No data processed in the evaluation loop. Check your dataloader or dataset.")

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return total_loss / len(dataloader), total_acc / len(dataloader), f1


def train_vatt(
        train_audio_embeddings: Optional[torch.Tensor] = None,
        train_text_embeddings: Optional[torch.Tensor] = None,
        train_labels: torch.Tensor = None,
        val_audio_embeddings: Optional[torch.Tensor] = None,
        val_text_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        feature_dim: int = 256,
        fusion_type: Literal['transformer', 'concat', 'cross_attention', 'late'] = 'transformer',
        modality: Literal['both', 'audio_only', 'text_only'] = 'both',
        num_classes: int = 5,
        batch_size: int = 32,
        num_epochs: int = 15,
        learning_rate: float = 1e-5,
        #contrastive_weight: float = 0.5,
        contrastive_weight: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> nn.Module:
        
        # 입력 데이터의 차원 자동 계산
        if train_audio_embeddings is not None:
            audio_input_dim = train_audio_embeddings.shape[-1]
        if train_text_embeddings is not None:
            text_input_dim = train_text_embeddings.shape[-1]
            
        # 모델 초기화
        model = VATT(
            audio_input_dim=audio_input_dim if train_audio_embeddings is not None else None,
            text_input_dim=text_input_dim if train_text_embeddings is not None else None,
            feature_dim=feature_dim,
            fusion_type=fusion_type,
            modality=modality,
            num_classes=num_classes
        ).to(device)
        
        # 옵티마이저 초기화
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 데이터로더 생성
        train_loader, val_loader = create_dataloaders(
            train_audio_embeddings=train_audio_embeddings,
            train_text_embeddings=train_text_embeddings,
            train_labels=train_labels,
            val_audio_embeddings=val_audio_embeddings,
            val_text_embeddings=val_text_embeddings,
            val_labels=val_labels,
            modality=modality,
            batch_size=batch_size
        )
        
        # 학습 루프
        for epoch in range(num_epochs):
            # 학습
            train_loss, train_acc = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                modality=modality,
                fusion_type=fusion_type,
                contrastive_weight=contrastive_weight
            )
            
            # 검증
            if val_loader is not None:
                val_loss, val_acc, val_f1 = evaluate(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    modality=modality
                )
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}\n')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n')
        
        return model

if __name__ == "__main__":
    # 임베딩 로드
    train_text_path = "/home/wangan/LLM_text_embedding/embeddings/text_embeddings_Llama3openko8b_train.pt"
    test_text_path = "/home/wangan/LLM_text_embedding/embeddings/text_embeddings_Llama3openko8b_test.pt"
    train_audio_path = "/home/wangan/Embeddings_train/audio/wav2vec2/audio_wav2vec_train_embeddings.pt"
    test_audio_path = "/home/wangan/Embeddings_test/audio/wav2vec2/audio_wav2vec_test_embeddings.pt" 
    train_label_path = "final_data/label_traning.csv"
    test_label_path = "final_data/label_test.csv"

    config = {
    "train_text_path": train_text_path,
    "test_text_path": test_text_path,
    "train_audio_path": train_audio_path,
    "test_audio_path": test_audio_path,
    "train_label_path": "final_data/label_traning.csv",
    "test_label_path": "final_data/label_test.csv",
    "batch_size": 64,
    "feature_dim": 512,
    "num_epochs": 20,
    "learning_rate": 1e-5,
    "contrastive_weight": 0.3,
    "base_model_path": "./models",
    "random_state": 42,
    "test_size": 0.1,
    "modality": "both",  # 'text_only', 'audio_only', 또는 'both'
    "fusion_types": ['transformer', 'concat', 'cross_attention', 'late'],
    }

    # Best Model 저장 경로 생성
    os.makedirs(config["base_model_path"], exist_ok=True)

    # 데이터 로드
    train_text = torch.load(config["train_text_path"]).float()
    test_text = torch.load(config["test_text_path"]).float()
    train_audio = torch.load(config["train_audio_path"]).float()
    test_audio = torch.load(config["test_audio_path"]).float()
    train_labels = torch.tensor(process_labels(config["train_label_path"]).values)
    test_labels = torch.tensor(process_labels(config["test_label_path"]).values)

    # 랜덤 Validation Split
    train_audio_final, val_audio, train_text_final, val_text, train_labels_final, val_labels = train_test_split(
        train_audio, train_text, train_labels, test_size=config["test_size"], random_state=config["random_state"]
    )

    # 데이터 정보 출력
    print(f"Dataset sizes:")
    print(f"Train samples: {train_audio_final.shape[0]}")
    print(f"Validation samples: {val_audio.shape[0]}")
    print(f"Test samples: {test_audio.shape[0]}")
    print(f"Number of classes: {len(torch.unique(train_labels_final))}")
    print(f"Audio Input Dimension: {train_audio_final.shape[-1]}")
    print(f"Text Input Dimension: {train_text_final.shape[-1]}")

    # 결과 저장용 딕셔너리
    results = {}

    # 싱글 모달리티 학습
    if config["modality"] in ['text_only', 'audio_only']:
        print(f"\n=== Training with {config['modality']} modality ===")
        model = train_vatt(
            train_audio_embeddings=train_audio_final if config["modality"] == 'audio_only' else None,
            train_text_embeddings=train_text_final if config["modality"] == 'text_only' else None,
            train_labels=train_labels_final,
            val_audio_embeddings=val_audio if config["modality"] == 'audio_only' else None,
            val_text_embeddings=val_text if config["modality"] == 'text_only' else None,
            val_labels=val_labels,
            feature_dim=config["feature_dim"],
            modality=config["modality"],
            num_classes=len(torch.unique(train_labels_final)),
            num_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            contrastive_weight=config["contrastive_weight"]
        )

        # Best Model 저장 및 로드
        best_model_path = os.path.join(config["base_model_path"], f"best_model_{config['modality']}.pth")
        torch.save(model.state_dict(), best_model_path)
        model.load_state_dict(torch.load(best_model_path))

        # 테스트 평가
        test_loader = create_dataloaders(
            train_audio_embeddings=test_audio if config["modality"] == 'audio_only' else None,
            train_text_embeddings=test_text if config["modality"] == 'text_only' else None,
            train_labels=test_labels,
            modality=config["modality"],
            batch_size=config["batch_size"]
        )[0]
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device='cuda', modality=config["modality"])
        print(f"\nTest Results for {config['modality']} modality:")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        results[config["modality"]] = test_acc

    else:
        # Multimodal 학습
        for fusion_type in config["fusion_types"]:
            print(f"\n=== Training with {fusion_type} fusion ===")
            model = train_vatt(
                train_audio_embeddings=train_audio_final,
                train_text_embeddings=train_text_final,
                train_labels=train_labels_final,
                val_audio_embeddings=val_audio,
                val_text_embeddings=val_text,
                val_labels=val_labels,
                feature_dim=config["feature_dim"],
                fusion_type=fusion_type,
                modality=config["modality"],
                num_classes=len(torch.unique(train_labels_final)),
                num_epochs=config["num_epochs"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                contrastive_weight=config["contrastive_weight"]
            )

            # Best Model 저장 및 로드
            best_model_path = os.path.join(config["base_model_path"], f"best_model_{fusion_type}.pth")
            best_val_loss = float('inf')

            for epoch in range(config["num_epochs"]):  
                train_loss, train_acc = train_epoch(
                    model=model,
                    dataloader=create_dataloaders(
                        train_audio_embeddings=train_audio_final,
                        train_text_embeddings=train_text_final,
                        train_labels=train_labels_final,
                        modality=config["modality"],
                        batch_size=config["batch_size"]
                    )[0],
                    optimizer=torch.optim.AdamW(model.parameters(), lr=config["learning_rate"]),
                    device='cuda',
                    modality=config["modality"],
                    fusion_type=fusion_type,
                    contrastive_weight=config["contrastive_weight"]
                )
                val_loss, val_acc, val_f1 = evaluate(
                    model=model,
                    dataloader=create_dataloaders(
                        train_audio_embeddings=val_audio,
                        train_text_embeddings=val_text,
                        train_labels=val_labels,
                        modality=config["modality"],
                        batch_size=config["batch_size"]
                    )[0],
                    device='cuda',
                    modality=config["modality"]
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"New Best Model Saved for {fusion_type} at Epoch {epoch + 1}")

            # Best Model 로드 후 테스트
            model.load_state_dict(torch.load(best_model_path))
            test_loader = create_dataloaders(
                train_audio_embeddings=test_audio,
                train_text_embeddings=test_text,
                train_labels=test_labels,
                modality=config["modality"],
                batch_size=config["batch_size"]
            )[0]
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, device='cuda', modality=config["modality"])
            results[fusion_type] = test_acc
            print(f"\nTest Results for {fusion_type} fusion:")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Tet F1: {test_f1:.4f}")

        # 최종 결과 출력
        print("\n=== Final Test Accuracies by Fusion Type ===")
        for fusion_type, acc in results.items():
            print(f"{fusion_type.capitalize()} Fusion: Test Acc: {acc:.4f}")
        