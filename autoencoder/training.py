# training.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoTokenizer

# model_architectures
from model_architectures.model_arc_standard_emb import MlpAutoencoder
# setup_device
from modules.setup_device import setup_device_optimizer_model_for_distributed, setup, cleanup

from torch.distributed import get_rank, get_world_size
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist
import random
import os


# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained("autoencoder/processed_data/tokenizer/")
vocab_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id  

# カスタムデータセットクラス
class JapaneseDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=600):
        self.data = torch.load(data_path) 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def sequence_length(self):
        return self.max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),           # 整数のまま保持
            'attention_mask': encoded['attention_mask'].squeeze(0), # 整数のまま保持
            'text': item['text'],
            'year': item['year'],      
            'sec_code': item['sec_code'] 
        }

# テスト関数
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    embeddings = []
    texts = []
    years = []
    sec_codes = []
    total_samples = torch.tensor(0).to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs, embs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            # 埋め込み、テキスト、年、証券コードを収集
            embeddings.extend(embs.cpu().tolist())
            texts.extend(batch['text'])
            years.extend(batch['year'].cpu().tolist())          # 年情報を追加
            sec_codes.extend(batch['sec_code'].cpu().tolist())  # 証券コードを追加

    # 全プロセスで損失とサンプル数を集計
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / total_samples.item()

    rank = get_rank()
    if rank == 0:
        print(f'Test Loss: {avg_loss:.4f}')

    # 辞書形式で保存するためにデータを整形
    results = [
        {
            'test_embeddings': embedding, 
            'test_year': year, 
            'test_sec_code': sec_code
        }
        for embedding, year, sec_code in zip(embeddings, years, sec_codes)
    ]

    return results, avg_loss, texts

# 評価関数（Validation 用）
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0).to(device)
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs, _ = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    # 全プロセスで損失とサンプル数を集計
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / total_samples.item()

    rank = get_rank()
    if rank == 0:
        print(f'Validation Loss: {avg_loss:.4f}')

    return avg_loss

# 学習関数
def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_loss_list = []
    val_loss_list = []
    rank = get_rank()

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)  # 追加
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)  # バッチ損失をサンプル数で乗算
            total_samples += input_ids.size(0)

        # 全プロセスで損失とサンプル数を集計
        total_loss = torch.tensor(running_loss).to(device)
        total_samples_tensor = torch.tensor(total_samples).to(device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        epoch_train_loss = total_loss.item() / total_samples_tensor.item()
        train_loss_list.append(epoch_train_loss)

        # 検証損失の計算
        epoch_val_loss = evaluate(model, val_loader, criterion, device)
        val_loss_list.append(epoch_val_loss)

        if rank == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    return train_loss_list, val_loss_list

def plot_training_curve(train_loss_list, val_loss_list, save_path="autoencoder/results/training_curve.png"):
    """
    学習曲線を描画する関数。

    Args:
        train_loss_list (list of float): 各エポックの訓練損失のリスト。
        val_loss_list (list of float): 各エポックの検証損失のリスト。
        save_path (str): グラフを保存するパス。
    """
    epochs = range(1, len(train_loss_list) + 1)
    
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss_list, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss_list, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # グラフをファイルに保存
    plt.savefig(save_path)
    print(f"Training curve has been saved to '{save_path}'.")
    
    # グラフを表示（オプション）
    plt.show()

def main():

    # モデルの初期化（プロセスグループが初期化される前）
    model = MlpAutoencoder(
        vocab_size=vocab_size,
        embedding_dim=128,  # 必要に応じて設定
        sequence_length=600,  # 最大シーケンス長
        latent_size=128
    )

    # デバイスの設定とモデルの分散ラップ
    device, optimizer, ddp_model = setup_device_optimizer_model_for_distributed(
        model, learning_rate=0.001, optimizer_class=torch.optim.Adam
    )
    model = ddp_model  # DDP モデルを使用

    # 分散プロセスのランクを取得
    rank = get_rank()
    world_size = get_world_size()

    # 既存の tokenizer を使用
    train_dataset = JapaneseDataset("autoencoder/processed_data/train_pytorch_dataset.pt", tokenizer)
    val_dataset = JapaneseDataset("autoencoder/processed_data/val_pytorch_dataset.pt", tokenizer)
    test_dataset = JapaneseDataset("autoencoder/processed_data/test_pytorch_dataset.pt", tokenizer)
    
    # DistributedSampler の設定
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # データローダーの設定
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    if rank == 0:
        # debug
        item = train_dataset[0]  # 最初のアイテムを取得
        print(f"Input IDs shape: {item['input_ids'].shape}")
        print(f"vocab_size: {vocab_size}")

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # 学習の実行
    num_epochs = 15
    train_loss_list, val_loss_list = train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
    if rank == 0:
        # 学習曲線の描画
        plot_training_curve(train_loss_list, val_loss_list)
    
    # テストデータでの評価
    test_results, test_loss, test_texts = test(model, test_loader, criterion, device)

    if get_rank() == 0:
        # 結果を保存するためのディレクトリを作成
        os.makedirs("autoencoder/results", exist_ok=True)

        # test_results を辞書形式で保存
        with open("autoencoder/results/test_results.json", "w", encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=4)

        print(f"Final Test Loss: {test_loss:.4f}")
        print("Test results (embeddings, year, sec_code) have been saved to 'autoencoder/results/test_results.json'.")
    
    # プロセスグループのクリーンアップ
    cleanup()

if __name__ == "__main__":
    main()