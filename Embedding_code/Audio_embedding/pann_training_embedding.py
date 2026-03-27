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
import librosa
import panns_inference
from panns_inference import AudioTagging
import torch
from panns_inference import AudioTagging
from tqdm import tqdm
import librosa
import os

preprocessed_train_audio_path = "/home/wangan/Data/Training/preprocessed_audio_train/"
preprocessed_test_audio_path = "/home/wangan/Data/Validation/preprocessed_audio_test/"
pr_train=os.listdir(preprocessed_train_audio_path)
pr_test=os.listdir(preprocessed_test_audio_path)

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

# 각 파일 경로에서 ID 추출
ids = [re.search(r'/([\d]{4})-', path).group(1) for path in pr_test_path]

# DataFrame 생성
df = pd.DataFrame({'file_path': pr_test_path, 'id': ids})

# ID와 파일명의 숫자 부분을 추출하는 함수
def extract_sort_keys(file_path):
    id_match = re.search(r'/([\d]{4})-', file_path)
    file_num_match = re.search(r'-(\d+)\.mp3', file_path)
    id_key = id_match.group(1) if id_match else '0000'
    file_num_key = int(file_num_match.group(1)) if file_num_match else 0
    return id_key, file_num_key

# 정렬 키 추출 및 정렬
df['sort_keys'] = df['file_path'].apply(extract_sort_keys)
df_sorted_test = df.sort_values(by='sort_keys')

# 정렬 키 열 제거 및 인덱스 재설정
df_sorted_test = df_sorted_test.drop('sort_keys', axis=1).reset_index(drop=True)

def generate_audio_embeddings(audio_paths, device='cuda:0'):
    # PANNs 체크포인트 경로 하드코딩
    checkpoint_path = "/home/wangan/panns_data/Cnn14_mAP=0.431.pth"

    # PANNs AudioTagging 모델 초기화
    at = AudioTagging(checkpoint_path=checkpoint_path, device=device)
    
    embeddings = []
    
    for audio_path in tqdm(audio_paths, desc="Generating embeddings"):
        # 오디오 파일 로드
        audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)

        # 샘플링 레이트를 32kHz로 리샘플링
        target_sample_rate = 32000
        if orig_sr != target_sample_rate:
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sample_rate)

        # 배치 차원을 추가
        audio = torch.tensor(audio[None, :], dtype=torch.float32)
        
        # GPU로 이동 (가능한 경우)
        audio = audio.to(device)

        # 모델 추론
        with torch.no_grad():
            _, embedding = at.inference(audio)
        
        # 임베딩을 numpy -> tensor로 변환 후 저장
        embeddings.append(torch.tensor(embedding, device=device))

    # 모든 임베딩을 하나의 텐서로 결합
    all_embeddings = torch.cat(embeddings, dim=0)
    
    return all_embeddings

# 사용 예시
audio_paths = df_sorted_test['file_path'].to_list()  # 오디오 파일 경로 리스트
embeddings = generate_audio_embeddings(audio_paths)

print(embeddings.shape)  # 출력: torch.Size([파일 수, 임베딩 차원])

# 저장할 디렉토리 경로
save_dir = '/home/wangan/Embeddings_test/audio/pann/'

# 파일명
filename = 'audio_pann_test_embeddings.pt'


# 전체 경로 생성
full_path = os.path.join(save_dir, filename)

# 임베딩 저장
torch.save(embeddings, full_path)

# # 나중에 임베딩 불러오기
# loaded_embeddings = torch.load('/home/wangan/Embeddings/audio/audio_train_embeddings.pt')

# print(loaded_embeddings.shape)  # 원본 embeddings와 동일한 shape를 가져야 합니다