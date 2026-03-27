import os
import pandas as pd
import torch
import torchaudio
import re
import gc
from tqdm.auto import tqdm
from transformers import WavLMModel, WavLMConfig

# 환경 변수 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 경로 설정
preprocessed_train_audio_path = "/home/wangan/Data/Training/preprocessed_audio_train/"
preprocessed_test_audio_path = "/home/wangan/Data/Validation/preprocessed_audio_test/"

def load_and_prepare_data(audio_path):
    audio_files = os.listdir(audio_path)
    file_paths = [os.path.join(audio_path, p) for p in audio_files]
    
    ids = []
    valid_paths = []
    for path in file_paths:
        match = re.search(r'/([\d]{4})-', path)
        if match:
            ids.append(match.group(1))
            valid_paths.append(path)
    
    df = pd.DataFrame({'file_path': valid_paths, 'id': ids})
    df['sort_keys'] = df['file_path'].apply(lambda x: (
        re.search(r'/([\d]{4})-', x).group(1),
        int(re.search(r'-(\d+)\.mp3', x).group(1)) if re.search(r'-(\d+)\.mp3', x) else 0
    ))
    return df.sort_values(by='sort_keys').drop('sort_keys', axis=1).reset_index(drop=True)

def process_audio(audio_path, target_sr=16000):
    waveform, orig_sr = torchaudio.load(audio_path)
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform

def segment_audio(audio, segment_length=16000*10, overlap=16000, min_segment_length=400):
    segments = []
    for start in range(0, audio.size(1), segment_length - overlap):
        end = min(start + segment_length, audio.size(1))
        segment = audio[:, start:end]
        if segment.size(1) >= min_segment_length:
            segments.append(segment)
    return segments

@torch.no_grad()
def generate_audio_embeddings(audio_paths, model_name="microsoft/wavlm-large"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = WavLMConfig.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name, config=config).to(device)
    model.eval()

    embeddings = []
    
    for path in tqdm(audio_paths, desc="Generating embeddings"):
        try:
            audio = process_audio(path)
            segments = segment_audio(audio)
            
            if not segments:
                print(f"Warning: No valid segments for file {path}")
                continue
            
            segment_embeddings = []
            for segment in segments:
                segment = segment.to(device)
                attention_mask = torch.ones(segment.shape, dtype=torch.bool, device=device)
                
                outputs = model(segment, attention_mask=attention_mask)
                embedding = torch.mean(outputs.last_hidden_state, dim=1)
                segment_embeddings.append(embedding.cpu())
            
            # Average embeddings of all segments
            file_embedding = torch.mean(torch.cat(segment_embeddings, dim=0), dim=0)
            embeddings.append(file_embedding)
            
            del outputs, segment_embeddings
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")
            continue
    
    return torch.stack(embeddings)

def main():
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing training data...")
    df_train = load_and_prepare_data(preprocessed_train_audio_path)
    embeddings_train = generate_audio_embeddings(df_train['file_path'].tolist())
    print("Training embeddings shape:", embeddings_train.shape)

    save_dir_train = '/home/wangan/Embeddings_train/audio/wavLM/'
    os.makedirs(save_dir_train, exist_ok=True)
    torch.save(embeddings_train, os.path.join(save_dir_train, 'audio_wavlm_train_nohalf_embeddings.pt'))
    
    del embeddings_train
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing test data...")
    df_test = load_and_prepare_data(preprocessed_test_audio_path)
    embeddings_test = generate_audio_embeddings(df_test['file_path'].tolist())
    print("Test embeddings shape:", embeddings_test.shape)

    save_dir_test = '/home/wangan/Embeddings_test/audio/wavLM/'
    os.makedirs(save_dir_test, exist_ok=True)
    torch.save(embeddings_test, os.path.join(save_dir_test, 'audio_wavlm_test_nohalf_embeddings.pt'))

if __name__ == "__main__":
    main()
