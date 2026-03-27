from transformers import RobertaForMaskedLM  # 수정된 부분
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling
import pandas as pd
from torch.optim import AdamW
from torch.nn import functional as F
import torch

# Load data
df = pd.read_csv('/home/wangan/pre_training/processed_pre_training_raw_data.csv', index_col=0)
text_column = 'text'
df[text_column].tolist()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
model = RobertaForMaskedLM.from_pretrained('klue/roberta-base')  # 수정된 부분

# Custom Dataset class for processing the text data
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            encoding = tokenizer.encode_plus(
                text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt'
            )
            self.examples.append(encoding['input_ids'].squeeze(0))  # Remove batch dimension

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Prepare dataset and dataloader
dataset = CustomDataset(df[text_column].tolist(), tokenizer, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)  # MLM 활성화
dataloader = DataLoader(dataset, batch_size=32, collate_fn=data_collator)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Set up training parameters
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the model
model.save_pretrained("/home/wangan/pre_training/roberta-mlm-pretrain")
