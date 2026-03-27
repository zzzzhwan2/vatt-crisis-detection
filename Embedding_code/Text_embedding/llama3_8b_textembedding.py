import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_model_and_tokenizer(model_name):
    if model_name == "beomi/Llama-3-Open-Ko-8B":
        MODEL_NAME = "beomi/Llama-3-Open-Ko-8B"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

        # 패딩 토큰 설정
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        
    else:
        raise ValueError("모델이 존재하지 않습니다. 올바른 모델 이름을 입력하세요.")

    return tokenizer, model

def get_files(path):
    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def make_dataframe(data):
    info = data['info']
    list_data = data['list']
    rows = []
    for i, item in enumerate(list_data, start=1):
        text = []
        for sub_item in item['list']:
            if 'audio' in sub_item:
                for audio in sub_item['audio']:
                    speaker = "상담자" if audio['type'] == "Q" else "내담자"
                    text.append(f"{speaker}: {audio['text']}")
        문항대화 = ' '.join(text)
        rows.append({
            "id": f"{info['ID']}-{i}",
            "대화": 문항대화
        })
    return pd.DataFrame(rows)

def process_data(path):
    json_files = get_files(path)
    df_list = []
    for item in json_files:
        with open(item, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = make_dataframe(data)
        df_list.append(df)
    dfs = pd.concat(df_list, ignore_index=True)
    dfs = dfs.sort_values(by="id")
    return dfs['대화'].tolist()

def embed_texts(texts, model, tokenizer):
    embeddings = []
    batch_size = 8  # Llama 모델은 큰 모델이므로 배치 크기를 줄입니다
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_batches = (len(texts) + batch_size - 1) // batch_size  # 총 배치 수 계산

    with tqdm(total=total_batches, desc="텍스트 임베딩", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu())

            pbar.update(1)

    embedded = torch.cat(embeddings, dim=0)
    print(f"\n임베딩 완료. Embeddings shape: {embedded.shape}")
    return embedded

def get_text_emb(model_name):
    try:
        tokenizer, model = load_model_and_tokenizer(model_name)
    except ValueError as e:
        return str(e)

    text_train = process_data('Data/Training/02/')
    text_valid = process_data('Data/Validation/02/')

    text_embed_train = embed_texts(text_train, model, tokenizer)
    text_embed_valid = embed_texts(text_valid, model, tokenizer)

    return text_embed_train, text_embed_valid

text_embed_train, text_embed_valid = get_text_emb("beomi/Llama-3-Open-Ko-8B")

# 저장할 디렉토리 경로
save_dir = '/home/wangan/LLM_text_embedding/embeddings/'
filename = 'text_embeddings_Llama3openko8b_train.pt'
# 전체 경로 생성
full_path = os.path.join(save_dir, filename)
# 임베딩 저장
torch.save(text_embed_train, full_path)

# 테스트셋
save_dir = '/home/wangan/LLM_text_embedding/embeddings/'
filename = 'text_embeddings_Llama3openko8b_test.pt'
full_path = os.path.join(save_dir, filename)
torch.save(text_embed_valid, full_path)