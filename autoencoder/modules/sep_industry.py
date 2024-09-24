import pandas as pd

def get_industry_category(sec_code):
    sec_code = int(sec_code) // 10
    if 1300 <= sec_code < 1400:
        return "水産・農業"
    elif 1500 <= sec_code < 1600:
        return "鉱業"
    elif 1600 <= sec_code < 1700:
        return "鉱業（石油／ガス開発）"
    elif 1700 <= sec_code < 2000:
        return "建設"
    elif 2000 <= sec_code < 3000:
        return "食品"
    elif 3000 <= sec_code < 4000:
        return "繊維・紙"
    elif 4000 <= sec_code < 5000:
        return "化学・薬品"
    elif 5000 <= sec_code < 6000:
        return "資源・素材"
    elif 6000 <= sec_code < 7000:
        return "機械・電機"
    elif 7000 <= sec_code < 8000:
        return "自動車・輸送機"
    elif 8000 <= sec_code < 9000:
        return "金融・商業"
    elif 9000 <= sec_code < 10000:
        return "運輸・通信・放送・ソフトウェア"
    else:
        return "その他"

print("データの読み込み")
df = pd.read_csv("all_combined_docs.tsv", sep="\t") 

print(df.columns)
df = df[['doc_id', 'file_suffix', 'text', 'year', 'sec_code']]

df['industry'] = df['sec_code'].apply(get_industry_category)

#"自動車・輸送機"のfile_suffixが”business_risks”のものを抽出
df = df[(df['industry'] == "自動車・輸送機") & (df['file_suffix'] == "business_risks")]

print(df.head())
print(df.shape)
print(df.info())

df.to_pickle("autoencoder/processed_data/auto_risks.pkl")
df.to_csv("autoencoder/processed_data/auto_risks.csv", index=False)
