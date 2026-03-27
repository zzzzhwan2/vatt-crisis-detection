# 🎙️ VATT-based Multimodal Crisis Detection (아동·청소년 정신건강 상태 모니터링)

**[딥러닝이론과실습]** 아동·청소년의 심리 상담 대화(텍스트)와 음성 데이터를 동시에 활용하여, 내담자의 정신건강 위기 단계를 5가지로 자동 분류하는 멀티모달(Multimodal) 딥러닝 프로젝트입니다. 

## 📝 프로젝트 배경 및 목적
* 아동·청소년기의 정신적 어려움은 성인기까지 악영향을 미치며, 전국 청소년 중 약 16%가 정신장애를 경험하고 7.1%는 즉각적인 전문가 개입이 필요한 상태입니다.
* 그러나 전문 상담사의 부족, 비용 부담, 정기적인 모니터링의 한계 등 현실적인 제약이 존재합니다.
* 이를 해결하기 위해 물리적, 시간적 제약 없이 내담자의 초기 상태를 평가할 수 있는 **정신건강 상태 자동 분류 시스템**을 제안합니다.

## 📊 데이터셋 (Dataset)
* **출처**: AI HUB 아동·청소년 상담 데이터 (7~12세 대상)
* **규모**: 원본 상담 데이터 3,236건을 문항 기준으로 분리하여 **총 22,652개**의 데이터 포인트로 구축
* **분류 라벨 (5 Classes)**: 정상군, 관찰 필요, 상담 필요, 학대 의심, 응급(위기 아동)

### 데이터 전처리 (Preprocessing)
* **Audio**: 내담자의 음성만 추출하여 병합한 뒤, 고정된 Sampling Rate(16KHz/32KHz) 및 [-1, 1] 스케일링을 거쳐 512차원 임베딩으로 변환합니다.
* **Text**: 형태소 분석 및 불용어 제거 등의 전처리보다 문장 전체를 입력했을 때 성능이 더 우수하여, 상담 전체 맥락을 유지한 원본 대화를 512차원 임베딩으로 변환합니다.

## 🏗️ 시스템 아키텍처 (Architecture)
텍스트와 오디오 인코더를 통과한 벡터를 다양한 방식으로 융합하고, 대조 학습을 적용해 멀티모달 시너지를 극대화하는 VATT(Video Audio Text Transformer) 기반 구조입니다.

1. **Speech & Text Encoders**: 각각 독립된 모델을 통해 512차원의 임베딩 생성
2. **Fusion Module**: 두 모달리티 결합
   * `Transformer Fusion`: Self-attention을 활용하여 순서와 모달리티 정보 보존
   * `Concat Fusion`: 단순 결합 후 다층 퍼셉트론(MLP) 통과
   * `Cross-Attention Fusion`: 모달리티 간 상호 참조를 통한 학습
   * `Late Fusion`: 독립 예측 후 최종 단계에서 가중 결합
3. **Multimodal Projection Head**: **Contrastive Learning (대조 학습)**을 통해 동일한 감정을/위기 단계를 가진 벡터 간의 거리를 좁혀 융합된 특성 표현 최적화
4. **Classification Head**: 최종 5개 클래스로 위기 단계 예측

## 🏆 주요 실험 결과 (Key Results)
다양한 텍스트(Llama-3, KoRoBERTa 등) 및 오디오(Wav2Vec2, HuBERT 등) 모델 조합 실험 결과:
* **최고 성능 (Accuracy 0.8388)**: 한국어 텍스트 이해 능력이 뛰어난 `Llama-3-Open-Ko-8B`와 `Wav2Vec2 (Base)` 오디오 조합에 **Transformer Fusion**을 적용했을 때 가장 높은 예측 정확도를 달성했습니다.
* **인사이트**: 텍스트(LLM) 모델의 성능이 압도적일 경우, 파인튜닝된 오디오 모델을 결합하면 오히려 과적합 혹은 성능 저하가 발생할 수 있음을 확인했습니다.

---

## 💻 실행 방법 (How to Run)

본 프로젝트는 모듈화되어 있으며 아래 순서대로 실행하여 파이프라인을 재현할 수 있습니다.

```bash
# 1. 데이터 전처리 (텍스트 및 오디오 파싱)
python Preprocessing_code/Text_processing.py
# (오디오는 Preprocessing_code/audio_processing.ipynb 참고)

# 2. 사전학습 모델 파인튜닝 (선택 사항)
python Fine-tuning_code/fine_tuning_KoRoberTa.py
# (오디오 파인튜닝은 Fine-tuning_code/fine_tuning_wav2vec2.ipynb 참고)

# 3. 임베딩 추출 (각 모달리티별 임베딩 .pt 파일 생성)
python Embedding_code/Text_embedding/llama3_8b_textembedding.py
python Embedding_code/Audio_embedding/wavLM_embedding_audio.py

# 4. VATT 모델 학습 및 평가 (Fusion 및 Classification)
python VATT_code/vatt.py
```

## 📂 폴더 구조 (Directory Structure)
```Plaintext
.
├── Preprocessing_code/          # 오디오 분할/라벨링 및 텍스트 정제 스크립트
├── Fine-tuning_code/            # KoRoBERTa, Wav2Vec2 등 개별 모델 도메인 적응 파인튜닝
├── Embedding_code/              # 사전학습 모델을 활용한 Feature 임베딩 추출
│   ├── Audio_embedding/         # HuBERT, PANNs, Wav2Vec2, WavLM 등
│   └── Text_embedding/          # Llama-3, KoRoBERTa, BERT 모델 등
├── VATT_code/                   # VATT 멀티모달 모델 정의 및 통합 학습 (Main)
└── Presentation_material/       # 프로젝트 최종 발표 자료 및 PDF
```
