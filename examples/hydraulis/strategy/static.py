import json
import pickle
import re
from tqdm import tqdm
from cost_model import get_strategy_max_seqlen, static_strategy_time_cost

def dynamic_programming(data, counter, N, S, S_STEP, K, max_seqlen_list):
    # 初始化表t和trace
    # t[N][S]表示使用不超过N个GPU去处理数据集中所有不超过S的seqs
    t = [[float('inf')] * (S + 1) for _ in range(N + 1)]
    trace = [[None] * (S + 1) for _ in range(N + 1)]
    
    # 初始化边界条件
    for n in range(N + 1):
        t[n][0] = 0
        trace[n][0] = []
    
    # 动态规划求解
    for n in tqdm(range(1, N + 1), desc=f"Enumerating on GPUs number"):
        for s in range(S_STEP, S + 1, S_STEP):
            t[n][s] = t[n - 1][s]  # 默认选择不使用额外的GPU
            trace[n][s] = trace[n - 1][s]
            
            # 枚举策略k
            for k in range(K):
                tp = data['strategies'][k]['tp']
                pp = data['strategies'][k]['pp']
                strategy_num_gpus = tp * pp  # 策略k所需的GPU数量
                
                # 如果策略k不能支持当前序列长度，或者所需GPU超过当前GPU数量，跳过
                if max_seqlen_list[k] < s or strategy_num_gpus > n:
                    continue
                
                # 枚举区间长度l
                for l in range(S_STEP, s + 1, S_STEP):
                    # 枚举并行策略的数量d
                    for d in range(1, n // strategy_num_gpus + 1):  # d个策略并行处理
                        cost = static_strategy_time_cost(data, counter, k, s - l, s, S_STEP)
                        new_value = max(t[n - d * strategy_num_gpus][s - l], cost / d)
                        # 更新最优解
                        if new_value < t[n][s]:
                            t[n][s] = new_value
                            trace[n][s] = trace[n - d * strategy_num_gpus][s - l] + [(s - l, s, f"dp{d}tp{tp}pp{pp}")]  # 记录选择的策略和区间

    return t, trace

if __name__ == '__main__':
    # 读取并打印strategy数据
    file_path = 'strategy_pool.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Read strategy data:")
    print(json.dumps(data, indent=4))
    
    # 读取数据集的counter
    file_path = '../_counter.pkl'
    with open(file_path, 'rb') as f:
        counter = pickle.load(f)
    print("Read dataset counter")

    # 示例调用
    N = 16  # 总的GPU数量
    S = 8192  # 数据集中最长的序列长度
    S_STEP = 128  # 序列长度的步长
    K = len(data['strategies'])  # 策略数量
    os_dp_tp_pp = (2, 4, 2)
    
    # 提前获取每个策略支持的最长seqlen
    max_seqlen_list = []
    for k in range(K):
        tp = data['strategies'][k]['tp']
        pp = data['strategies'][k]['pp']
        max_seqlen = get_strategy_max_seqlen(data, k, os_dp_tp_pp=os_dp_tp_pp)
        max_seqlen_list.append(max_seqlen)  # 策略k支持的最大序列长度
        print(f"tp{tp}pp{pp} max_seqlen is {max_seqlen}")
    
    # 调用动态规划函数，找到最优解
    t, trace = dynamic_programming(data, counter, N, S, S_STEP, K, max_seqlen_list)
    
    # 打印最优解和最优方案
    multi_tp_pp_list = [[(os_dp_tp_pp[1], os_dp_tp_pp[2]) for _ in range(os_dp_tp_pp[0])]]
    for s in range(S_STEP, S + 1, S_STEP):
        # print(f"Optimal time cost: {t[N][s]}")
        print(f"max_seqlen: {s}, optimal trace: {trace[N][s]}")
        tp_pp_list = []
        for x in trace[N][s]:
            pattern = r'(dp|tp|pp)(\d+)'
            matches = re.findall(pattern, x[2])
            result = {key: int(value) for key, value in matches}
            for _ in range(result['dp']):
                tp_pp_list.append((result['tp'], result['pp']))
        if tp_pp_list not in multi_tp_pp_list:
            multi_tp_pp_list.append(tp_pp_list)
    print(f"multi_tp_pp_list: {multi_tp_pp_list}")