import os
import json
import pickle
import pandas as pd
import torch
import torchaudio
import re
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
from datetime import datetime
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from concurrent.futures import ProcessPoolExecutor

preprocessed_train_audio_path = "/home/wangan/Data/Training/preprocessed_audio_train/"
preprocessed_test_audio_path = "/home/wangan/Data/Validation/preprocessed_audio_test/"
pr_train = os.listdir(preprocessed_train_audio_path)
pr_test = os.listdir(preprocessed_test_audio_path)

pr_train_path = [os.path.join(preprocessed_train_audio_path, p) for p in pr_train]
pr_test_path = [os.path.join(preprocessed_test_audio_path, p) for p in pr_test]

# 각 파일 경로에서 ID 추출
ids = [re.search(r'/([\d]{4})-', path).group(1) for path in pr_train_path]

# DataFrame 생성
df = pd.DataFrame({'file_path': pr_train_path, 'id': ids})

# ID와 파일명의 숫자 부분을 추출하는 함수
def extract_sort_keys(file_path):
    id_match = re.search(r'/([\d]{4})-', file_path)
    file_num_match = re.search(r'-(\d+)\.mp3', file_path)
    id_key = id_match.group(1) if id_match else '0000'
    file_num_key = int(file_num_match.group(1)) if file_num_match else 0
    return id_key, file_num_key

# 정렬 키 추출 및 정렬
df['sort_keys'] = df['file_path'].apply(extract_sort_keys)
df_sorted = df.sort_values(by='sort_keys')

# 정렬 키 열 제거 및 인덱스 재설정
df_sorted = df_sorted.drop('sort_keys', axis=1).reset_index(drop=True)

# 테스트 데이터에 대해서도 동일한 처리
ids = [re.search(r'/([\d]{4})-', path).group(1) for path in pr_test_path]
df = pd.DataFrame({'file_path': pr_test_path, 'id': ids})
df['sort_keys'] = df['file_path'].apply(extract_sort_keys)
df_sorted_test = df.sort_values(by='sort_keys')
df_sorted_test = df_sorted_test.drop('sort_keys', axis=1).reset_index(drop=True)

def generate_audio_embeddings(audio_paths, model_path, processor_path):
    # 파인튜닝된 Wav2Vec2 모델과 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(processor_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    embeddings = []
    
    for audio_path in tqdm(audio_paths, desc="Generating embeddings"):
        waveform, orig_sr = torchaudio.load(audio_path)
        
        sample_rate = 16000
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze().tolist()
        
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embedding = torch.mean(outputs.last_hidden_state, dim=1)
        embeddings.append(embedding.cpu())
    
    all_embeddings = torch.cat(embeddings, dim=0)
    
    return all_embeddings

# 파인튜닝된 모델과 프로세서 경로
finetuned_model_path = "/home/wangan/wav2vec_finetune/parameter/SFT_wav2vec2_model_best"
finetuned_processor_path = "/home/wangan/wav2vec_finetune/parameter/SFT_wav2vec2_model_processor_best"

# 훈련 데이터 임베딩 생성
audio_paths = df_sorted['file_path'].to_list()
embeddings_train = generate_audio_embeddings(audio_paths, finetuned_model_path, finetuned_processor_path)

print("Training embeddings shape:", embeddings_train.shape)

# 훈련 데이터 임베딩 저장
save_dir_train = '/home/wangan/Embeddings_train/audio/SFT_wav2vec2/'
filename_train = 'audio_SFTwav2vec_train_embeddings_finetuned22.pt'
full_path_train = os.path.join(save_dir_train, filename_train)
torch.save(embeddings_train, full_path_train)

# 테스트 데이터 임베딩 생성
audio_paths_test = df_sorted_test['file_path'].to_list()
embeddings_test = generate_audio_embeddings(audio_paths_test, finetuned_model_path, finetuned_processor_path)

print("Test embeddings shape:", embeddings_test.shape)

# 테스트 데이터 임베딩 저장
save_dir_test = '/home/wangan/Embeddings_test/audio/SFT_wav2vec2/'
filename_test = 'audio_SFTwav2vec_test_embeddings_finetuned22.pt'
full_path_test = os.path.join(save_dir_test, filename_test)
torch.save(embeddings_test, full_path_test)
