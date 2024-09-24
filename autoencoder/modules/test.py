import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from torchsummary import summary
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pyarrow as pa

# データの読み込みと前処理
df = pd.read_csv("all_combined_docs.tsv", sep="\t") 
print(df.columns)   
df.to_pickle("all_combined_docs.pkl")
