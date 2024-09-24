from transformers import AutoTokenizer
import torch

# トークナイザーをグローバルに初期化
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v2")

def tokenizer_function(examples):
    # バッチ処理でトークン化
    encodings = tokenizer(
        examples['text'],
        add_special_tokens=True,
        max_length=600,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # トークン化されたテキストを取得
    tokenized_texts = [tokenizer.convert_ids_to_tokens(ids) for ids in encodings['input_ids']]
    
    return {
        'tokenized_text': tokenized_texts,
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist()
    }