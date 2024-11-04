import os
import signal
import time
import argparse
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from data_utils import LLaMAJsonDataset

dataset = "commoncrawl"
dataset = "github"
dataset = "code"
counter_file_path = f"./dataset_analysis/{dataset}_counter.pkl"
cdf_file_path = f"./dataset_analysis/{dataset}_cdf.png"
max_counter_file_path = f"./dataset_analysis/{dataset}_max_counter.pkl"
max_cdf_file_path = f"./dataset_analysis/{dataset}_max_cdf.png"
simulation_file_path = f"./dataset_analysis/{dataset}_simulation.png"

def read_counter():
    with open(counter_file_path, 'rb') as file:
        counter = pickle.load(file)
    print(f"Max seq len is {max(counter.keys())}")

def scan_and_dump(dataset, batch_size=1000):
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # Initialize an empty list to store sequence lengths
    seqlen_list = []
    # Process data in batches
    for i in tqdm(range(0, total_seqs, batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        seqlen_list.extend(batch_seqlen)
    # Convert the list to a numpy array for further processing
    # Count the occurrences of each sequence length
    counter = Counter(seqlen_list)
    with open(counter_file_path, 'wb') as file:
        pickle.dump(counter, file)
    x_vals, counts = zip(*sorted(counter.items()))  
    # Calculate the cumulative distribution function (CDF)
    y_vals = np.cumsum(counts) / total_seqs
    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='CDF', color='blue', lw=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.3)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(cdf_file_path)
    plt.show()
    
def scan_and_dump_max_seqlen(dataset, batch_size=64):
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # Initialize an empty list to store sequence lengths
    max_seqlen_list = []
    # Process data in batches
    for i in tqdm(range(0, total_seqs, batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        max_seqlen_list.append(max(batch_seqlen))
    # Convert the list to a numpy array for further processing
    # Count the occurrences of each sequence length
    counter = Counter(max_seqlen_list)
    with open(max_counter_file_path, 'wb') as file:
        pickle.dump(counter, file)
    x_vals, counts = zip(*sorted(counter.items()))  
    # Calculate the cumulative distribution function (CDF)
    y_vals = np.cumsum(counts) / len(max_seqlen_list)
    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='CDF', color='blue', lw=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.3)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(max_cdf_file_path)
    plt.show()

def draw_sample_simulation(dataset, batch_size=64):
    # 假设 dataset 和 batch_size 已经定义
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # 初始化列表以存储每个batch的最大序列长度和每个batch中所有序列的长度
    max_seqlen_list = []
    batch_indices = []
    all_seqlen_list = []
    # 处理数据
    for i in tqdm(range(0, min(total_seqs, 100 * batch_size), batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        # 计算每个序列的长度，假设pad_id是用来填充的标记
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        # 记录当前batch的最大序列长度
        max_seqlen_list.append(max(batch_seqlen))
        # 记录当前batch的所有序列长度
        all_seqlen_list.extend(batch_seqlen)
        # 为每个序列分配相同的批次索引
        batch_indices.extend([i // batch_size] * len(batch_seqlen))
    # 绘制图像
    plt.figure(figsize=(12, 6))
    # 绘制最大序列长度的折线图
    plt.plot(range(len(max_seqlen_list)), max_seqlen_list, label='Max Sequence Length', color='blue', marker='o', markersize=4)
    # 绘制所有序列长度的散点图，使用相同的 batch 索引
    plt.scatter(batch_indices, all_seqlen_list, color='r', label='Sequence Length', alpha=0.5, s=10)
    # 添加标题和标签
    plt.title('Sequence Length Distribution of Different Batches')
    plt.xlabel('Batch Index')
    # 添加图例
    plt.legend()
    # 美化图形
    plt.grid(True, linestyle='--', alpha=0.6)
    # 展示图像
    plt.tight_layout()
    plt.savefig(simulation_file_path)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_seq_len", type=int, default=32768, help="maximum sequence len"
    )
    parser.add_argument(
        "--json_file", type=str, default=f"data/{dataset}/code.json", help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, default="content", help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, default="data/vocab.json", help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, default="data/merges.txt", help='gpt merge file path'
    )
    args = parser.parse_args()
    dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file
    )
    scan_and_dump(dataset)
    scan_and_dump_max_seqlen(dataset)
    # read_counter()
    # draw_sample_simulation(dataset)