from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

def read_file(file_path):
    return pd.read_pickle(file_path)

# トークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v2")

def tokenizer_function(examples):
    encodings = tokenizer(
        examples['text'],
        add_special_tokens=True,
        max_length=600,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist()
    }

# データの読み込みと前処理
print("データの読み込み")
df = read_file("autoencoder/processed_data/auto_risks.pkl")

print(df.columns)
df = df[['file_suffix', 'text', 'year', 'sec_code']]
df["text"] = df["file_suffix"] + " " + df["text"]
df.drop(columns=["file_suffix"], inplace=True)
# df = df[:300]

print("データの分割")
train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.9, random_state=42)

print("データセットの作成")
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# データセットの前処理
print("データセットのトークン化")
train_dataset = train_dataset.map(
    tokenizer_function,
    batched=True,
    batch_size=32,
    num_proc=8  # CPUコア数に合わせて調整
)
val_dataset = val_dataset.map(
    tokenizer_function,
    batched=True,
    batch_size=32,
    num_proc=8
)
test_dataset = test_dataset.map(
    tokenizer_function,
    batched=True,
    batch_size=32,
    num_proc=8
)

# トークン化されたデータを確認
print(val_dataset['input_ids'][0])

# PyTorch 形式に設定
print("PyTorch 形式への変換")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'year', 'sec_code', 'text'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'year', 'sec_code', 'text'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'year', 'sec_code', 'text'])

# データセットの保存
print("データセットの保存")
torch.save(train_dataset, "autoencoder/processed_data/train_pytorch_dataset.pt")
torch.save(val_dataset, "autoencoder/processed_data/val_pytorch_dataset.pt")
torch.save(test_dataset, "autoencoder/processed_data/test_pytorch_dataset.pt")

# トークナイザーの保存
tokenizer.save_pretrained("autoencoder/processed_data/tokenizer/")