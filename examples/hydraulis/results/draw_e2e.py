import re
import matplotlib.pyplot as plt

def extract_total_run_times(file_path):
    total_run_times = []

    # 使用正则表达式匹配 "total run time: <number> ms"
    pattern = re.compile(r'total run time:\s*(\d+)\s*ms')

    with open(file_path, 'r') as file:
        try:
            for line in file:
                match = pattern.search(line)
                if match:
                    total_run_times.append(int(match.group(1)))
        except Exception:
            pass 

    return total_run_times[1:50]

def plot_total_run_times(file_paths, labels):
    plt.figure(figsize=(12, 8))

    for file_path, label in zip(file_paths, labels):
        total_run_times = extract_total_run_times(file_path)
        plt.plot(total_run_times, marker='o', linestyle='-', label=label)

    plt.title('Total Run Time Comparison (gbs=64)')
    plt.xlabel('Index')
    plt.ylabel('Total Run Time (ms)')
    plt.ylim(0, 3000)
    plt.legend()
    plt.grid(True)
    plt.savefig('./optimizer_strategy.png')

if __name__ == "__main__":
    file_paths = [
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case1/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case2/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case3/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        # '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case2/llama7b_gpus16_gbs128_msl8192/log_0.txt',
        # '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case3/llama7b_gpus16_gbs128_msl8192/log_0.txt'
    ]
    labels = [
        "Greedy Packing (Static Shape)",
        "Greedy Packing (Dynamic Shape)",
        "Our Packing",
        # "Balanced Packing",
        # "Ours Estimated Packing"
        # "Hetero Packing"
    ]
    file_paths = [
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/8181_single_comm/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/4242/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/2424/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/1818/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/8181_single_comm/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/4218_split_all_gather/llama7b_gpus16_gbs64_msl8192/log_0.txt'
    ]
    labels = [
        "optimizer strategy dp2tp8pp1",
        "optimizer strategy dp2tp4pp2",
        "optimizer strategy dp2tp2pp4",
        "optimizer strategy dp2tp1pp8",
        "optimizer-compute strategy aligned",
    ]
    plot_total_run_times(file_paths, labels)