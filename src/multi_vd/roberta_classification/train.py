import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import classification_report

class VulDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['code'])  # Assuming 'code' column
        label = int(self.data.iloc[idx]['vuln'])  # Assuming 'label' column
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    
    # Get basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    
    # Print detailed classification report
    print(classification_report(labels, predictions, target_names=['Non-vulnerable', 'Vulnerable']))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(labels, predictions, target_names=['Non-vulnerable', 'Vulnerable'])
    }

# Initialize tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)

# Load datasets
train_dataset = VulDataset('/drive1/cuongtm/hienlt/roberta/dataset/primevul_train_full_nounder.csv', tokenizer)
val_dataset = VulDataset('/drive1/cuongtm/hienlt/roberta/dataset/primevul_val_full_nounder.csv', tokenizer)
test_dataset = VulDataset('/drive1/cuongtm/hienlt/roberta/dataset/primevul_test_full_nounder.csv', tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")
os.makedirs('./logs', exist_ok=True)

with open('./logs/test_primevul_full_nounder_moe.txt', 'w', encoding='utf-8') as f:
    f.write(str(test_results))