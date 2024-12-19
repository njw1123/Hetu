import os
import argparse
import yaml
from pssh.clients import ParallelSSHClient
import multiprocessing.spawn
import signal
import time
import argparse
import ast
import fcntl
import json
import numpy as np
from mpi4py import MPI
from hetu_llama import LLamaLMHeadModel
from llama_config import LLaMAConfig
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_ds_parallel_config
from strategy import get_strategy_max_seqlen, find_optimal_strategy

# enable_host_logger()

ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def pssh(args, multi_dp_representive_gpu):
    hostnames = []
    train_command = args.command
    cwd = os.getcwd()
    cmd = "cd " + cwd 
    cmd += f" && source {args.envs} && " 
    cmd_list = []
    for i in range(args.ngpus):
        unused = True
        for dp_representive_gpu in multi_dp_representive_gpu:
            # print("values", dp_representive_gpu.values())
            if i in dp_representive_gpu.values():
                unused = False
                break
        if unused:
            continue
        hostnames.append('localhost')
        cmd_list.append(cmd + train_command + f" --rank {i} 2>&1 | tee {args.log_path}" + "/log_" + f"{i}" + ".txt")
    clients = []
    outputs = []
    for hostname, cmd in zip(hostnames, cmd_list):
        client = ParallelSSHClient([hostname])
        output = client.run_command(cmd)
        clients.append(client)
        outputs.append(output)
    for client in clients:
        client.join() 
    for output in outputs:
        for host_out in output:
            for line in host_out.stderr:
                print("[stderr]:", line)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command", type=str, default='uname', help="command for pssh"
    )
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num gpus"
    )
    parser.add_argument(
        "--envs", type=str, help="multi-node shared envs"
    )
    parser.add_argument(
        "--log_path", type=str, help="log folder path"
    )
    parser.add_argument(
        "--strategy_pool", type=str, default="./strategy/strategy_pool.json", help="json path to the strategy pool"
    )
    parser.add_argument(
        "--multi_tp_pp_list", type=str, default="[]", help="multi hetero dp strategy list"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="number of layers"
    )
    args = parser.parse_args()
    args.multi_tp_pp_list = ast.literal_eval(args.multi_tp_pp_list)
    assert len(args.multi_tp_pp_list) >= 1, "there should be at least one strategy"            
    # Generate & read configs
    with open(args.strategy_pool, 'r') as f:
        strategy_pool = json.load(f)
    multi_tp_pp_list = args.multi_tp_pp_list
    num_strategy = len(multi_tp_pp_list)
    multi_dp_size = [len(tp_pp_list) for tp_pp_list in multi_tp_pp_list]
    multi_gpu_pos = []
    multi_config_file_path = []
    multi_match_id_list = []
    multi_max_seqlen_list = []
    multi_dp_representive_gpu = []
    
    # 默认策略list中第一个放optimizer的同构的strategy
    os_tp, os_pp = multi_tp_pp_list[0][0]
    os_dp = args.ngpus // os_tp // os_pp
    for tp_pp in multi_tp_pp_list[0]:
        assert tp_pp[0] == os_tp and tp_pp[1] == os_pp, "must ensure the first strategy is a homo optimizer strategy"
    
    for strategy_id in range(num_strategy):
        # 获取当前异构dp策略下每个tp+pp子策略在pool中的id以及其支持的最大seq长度
        match_id_list = []
        max_seqlen_list = []
        dp_representive_gpu = {}
        for tp_pp in multi_tp_pp_list[strategy_id]:
            tp = tp_pp[0]
            pp = tp_pp[1]
            match_id = None
            for i, data in enumerate(strategy_pool['strategies']):
                if data['tp'] == tp and data['pp'] == pp:
                    match_id = i
                    break
            assert match_id != None, f"can't find tp{tp}pp{pp} in the strategy pool, please use the strategy within the pool"
            match_id_list.append(match_id)
            max_seqlen = get_strategy_max_seqlen(strategy_pool, match_id, os_dp_tp_pp=(os_dp, os_tp, os_pp))
            aligned_max_seqlen = max_seqlen // alignment * alignment
            max_seqlen_list.append(aligned_max_seqlen)
        multi_match_id_list.append(match_id_list)
        multi_max_seqlen_list.append(max_seqlen_list)
        print(f"Strategy {strategy_id}, match strategy id list: {match_id_list} and max seqlen list: {max_seqlen_list}")
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        layers_tp_groups, gpu_pos = convert_strategy(multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        generate_ds_parallel_config(args.ngpus, layers_tp_groups, config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
        # 找到每个dp中编号最小的gpu_id
        # 后面需要用这些gpu代表去跑决策算法
        for cur_gpu_id, cur_pos in gpu_pos.items():
            if cur_pos.dp_id not in dp_representive_gpu:
                dp_representive_gpu[cur_pos.dp_id] = cur_gpu_id
            else:
                dp_representive_gpu[cur_pos.dp_id] = min(dp_representive_gpu[cur_pos.dp_id], cur_gpu_id)
        print(f"Strategy {strategy_id}, DP representive gpu:", dp_representive_gpu)
        multi_dp_representive_gpu.append(dp_representive_gpu)
        
    pssh(args, multi_dp_representive_gpu)
