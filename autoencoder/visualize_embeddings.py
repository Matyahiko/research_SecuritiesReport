import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# JSONファイルから埋め込みと属性を読み込む関数
def load_test_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        test_results = json.load(f)
    
    embeddings = np.array([item['test_embeddings'] for item in test_results])
    years = np.array([item['test_year'] for item in test_results])
    sec_codes = np.array([item['test_sec_code'] for item in test_results])
    
    return embeddings, years, sec_codes

# t-SNEで埋め込みベクトルを可視化する関数
def visualize_tsne(embeddings, years, sec_codes, output_html="tsne_visualization.html"):
    # t-SNEで2次元に圧縮
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne_model.fit_transform(embeddings)

    # DataFrame形式に変換
    data = {
        'TSNE-1': tsne_embeddings[:, 0],
        'TSNE-2': tsne_embeddings[:, 1],
        'Year': years,
        'Sec Code': sec_codes
    }

    # Plotly Expressで散布図を作成
    fig = px.scatter(
        data, x='TSNE-1', y='TSNE-2',
        color='Year',  # Yearで色分け
        hover_data=['Sec Code'],  # マウスホバー時にSec Codeを表示
        title="t-SNE Visualization of Embeddings"
    )

    # HTMLファイルに保存
    fig.write_html(output_html)
    print(f"t-SNE visualization saved as {output_html}")

# メイン処理
def main():
    # JSONファイルのパス
    file_path = "autoencoder/results/test_results.json"
    
    # JSONファイルから埋め込みと年、証券コードを読み込み
    embeddings, years, sec_codes = load_test_results(file_path)
    
    # t-SNEを使って埋め込みを可視化し、HTMLファイルに保存
    visualize_tsne(embeddings, years, sec_codes, output_html="autoencoder/results/tsne_visualization.html")

if __name__ == "__main__":
    main()
