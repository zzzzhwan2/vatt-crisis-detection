from transformers import AutoTokenizer, RobertaModel
def get_text_emb(model):
    import os 
    import json
    import pandas as pd
    import torch


    #json 데이터 파일 경로 반환 함수: get_files(경로명)
    def get_files(path):
        json_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files
    
    #입력된 json 데이터를 데이터프레임으로 만드는 함수: make_dataframe(파일명)
    def make_dataframe(data):
        info = data['info']
        list_data = data['list']
        rows = []
        for i, item in enumerate(list_data, start=1):
            text = [] 
            for sub_item in item['list']:
                if 'audio' in sub_item:
                    for audio in sub_item['audio']:
                        if audio['type'] == "Q":
                            speaker = "상담자"
                        else: 
                            speaker = "내담자"
                        text.append(f"{speaker}: {audio['text']}")
            문항대화 =' '.join(text)
            rows.append({
                "id": f"{info['ID']}-{i}",
                "대화": 문항대화
            })
        return pd.DataFrame(rows)
    
    #train 데이터 데이터프레임화하기 
    json_files = get_files('Data/Training/02/')
    df_list = [] 
    for item in json_files: 
        with open(item, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = make_dataframe(data)
        df_list.append(df)
    dfs = pd.concat(df_list, ignore_index = True)
    dfs.sort_values(by="id") # 마지막에 합치기(id 오름차순)
    text_train = dfs['대화'].tolist() #임베딩할 부분만 리스트형으로 추출(리스트로 받음)
    
    #validation 데이터 데이터프레임화하기
    json_files = get_files('Data/Validation/02/')
    df_list = [] 
    for item in json_files: 
        with open(item, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = make_dataframe(data)
        df_list.append(df)
    dfs = pd.concat(df_list, ignore_index = True)
    dfs.sort_values(by="id") 
    text_valid = dfs['대화'].tolist() 

    # 배치별 임베딩하는 함수: embed_texts(임베딩할 텍스트)
    def embed_texts(texts):
        embeddings = []
        batch_size=64
        
        device = "cuda:0"
        model.to(device)
        model.eval()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 입력테스트 토큰화 및 텐서변환
            inputs = tokenizer(batch_texts, padding="longest", truncation=True, return_tensors="pt", max_length=512)
            # GPU로 텐서이동
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device) #패딩된 부분 무시하도록 함
            # 임베딩
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]  # 마지막 출력값의 임베딩
                embeddings.append(batch_embeddings.cpu())  # GPU -> CPU로 이동
            
            print(f"텍스트 임베딩 중: batch {i},{i/len(texts)*100:.0f}%")

        embedded = torch.cat(embeddings, dim=0) # 배치별 임베딩을 하나의 텐서로 합침
        print("임베딩 완료. Embeddings shape:", embedded.shape)
        return embedded
    
    if model == "KoRoBERTa":
        TOKENIZER_NAME = 'klue/roberta-base'
        MODEL_NAME = "/home/wangan/pre_training/roberta-mlm-pretrain"
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = RobertaModel.from_pretrained(MODEL_NAME)

     

    else:
        return "모델이 없습니다."

    text_embed_train = embed_texts(text_train)
    text_embed_valid = embed_texts(text_valid)

    return text_embed_train, text_embed_valid

text_embed_train, text_embed_valid = get_text_emb("KoRoBERTa")

import os
import torch

# 저장할 디렉토리 경로
save_dir = '/home/wangan/fine_tuning/Embedding_train/'
filename = 'fine_tuned_train_text_embeddings_KoRoBERTa.pt'
# 전체 경로 생성
full_path = os.path.join(save_dir, filename)
# 임베딩 저장
torch.save(text_embed_train, full_path)

# 테스트셋
save_dir = '/home/wangan/fine_tuning/Embedding_test/'
filename = 'fine_tuned_test_text_embeddings_KoRoBERTa.pt'
full_path = os.path.join(save_dir, filename)
torch.save(text_embed_valid, full_path)