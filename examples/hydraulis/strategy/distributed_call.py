import os
import pickle
import time
import fcntl
import concurrent.futures
import numpy as np
import hetu as ht
from typing import Callable, Any, Tuple, Dict
from .dynamic_pulp import dynamic_strategy, batching_strategy

func_call_folder = "./func_call"
write_tag = {}
read_tag = {}

# workaround
# hope to leverage grpc in the future
def distributed_call(need_max_cost: bool, distributed_status: Tuple[int, int, int, int, Dict[int, int]], func: Callable, *args: Any, **kwargs: Any):
    # synchronize all processes
    # but seems doesn't work
    # ht.global_comm_barrier() 
    global write_tag, read_tag
    start_time = time.time()
    strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu = distributed_status
    if strategy_id not in write_tag:
        write_tag[strategy_id] = 0
    if strategy_id not in read_tag:
        read_tag[strategy_id] = 0
    path = func_call_folder + f"/{func.__name__}_strategy{strategy_id}_dp{dp_id}.pkl"
    assert os.path.exists(func_call_folder), f"please ensure {func_call_folder} is created"
        # os.makedirs(func_call_folder)        
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, call func {func.__name__} begin...")
    if gpu_id == dp_representive_gpu[dp_id]:
        # representive rank process call the function and write the result to the file
        # print(f"call func {func.__name__} begin...")
        result = func(*args, **kwargs)
        with open(path, 'wb') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                pickle.dump((result, write_tag[strategy_id]), file)
                file.flush()
                os.fsync(file.fileno())
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
        # print(f"call func {func.__name__} end...")
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, call func {func.__name__} end...")
    write_tag[strategy_id] += 1
    # ht.global_comm_barrier() 
    all_dp_results = []
    dp_range = list(range(dp_size)) if need_max_cost else [dp_id,]
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, get dp range {dp_range} results of {func.__name__} begin...")
    for cur_id in dp_range:
        path = func_call_folder + f"/{func.__name__}_strategy{strategy_id}_dp{cur_id}.pkl"
        while True:
            try:
                with open(path, 'rb') as file:
                    try:
                        fcntl.flock(file, fcntl.LOCK_SH)
                        result, tag = pickle.load(file)
                    finally:
                        fcntl.flock(file, fcntl.LOCK_UN)
            except Exception as e:
                # print("Exception raise")
                time.sleep(1)  # 等待文件写入完成
                continue
            if tag == read_tag[strategy_id]:
                break
            else:
                # print(f"read tag = {read_tag[strategy_id]} but file tag = {tag}")
                time.sleep(1)  # 等待文件写入完成
        all_dp_results.append(result)
    read_tag[strategy_id] += 1
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, get dp range {dp_range} results of {func.__name__} end...")
    # ht.global_comm_barrier() 
    end_time = time.time()
    if need_max_cost:
        max_cost_result = max(all_dp_results, key=lambda x: x[0])
        final_result = (max_cost_result[0], all_dp_results[dp_id][1])
    else:
        final_result = all_dp_results[0]
    # print(f"Distributed func call {func.__name__} time cost: {end_time - start_time}s")
    return final_result

def process_strategy(args):
    estimated_cost_1 = None
    batch_indices = None
    estimated_cost_2 = None
    batching_option_matrix = None
    
    (
        compute_strategy_id, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
        multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
        all_devices, local_device, batching_method, 
        strategy_pool, sorted_len
    ) = args
    
    dp_size = multi_dp_size[compute_strategy_id]
    tp_pp_list = multi_tp_pp_list[compute_strategy_id]
    max_seqlen_list = multi_max_seqlen_list[compute_strategy_id]
    match_id_list = multi_match_id_list[compute_strategy_id]
    gpu_pos = multi_gpu_pos[compute_strategy_id]
    dp_representive_gpu = multi_dp_representive_gpu[compute_strategy_id]
    gpu_id = all_devices.get_index(local_device)
    
    assert gpu_id in gpu_pos, f"gpu {gpu_id} is not included in this training"
    dp_id, stage_id = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id
    assert dp_id < dp_size, "dp size mismatches"
    
    # Call dynamic strategy (distributed)
    estimated_cost_1, batch_indices = distributed_call(
        True, (compute_strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu), 
        dynamic_strategy, strategy_pool, match_id_list, max_seqlen_list, dp_id, sorted_len
    )
    
    # hydraulis packing: balanced packing with utilization guaranteed
    if batching_method == 4:
        # Call batching strategy (distributed)
        estimated_cost_2, batching_option_matrix = distributed_call(
            True, (compute_strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu), 
            batching_strategy, strategy_pool, match_id_list[dp_id], 
            sorted_len[batch_indices], max_seqlen_list[dp_id]
        )
        if not isinstance(batching_option_matrix, np.ndarray):
            print(f"{local_device}: {compute_strategy_id}-th strategy {dp_id}-th dp cannot guarantee the sequence utilization, the seqs that need to pack is {sorted_len[batch_indices]}")
    # greedy packing
    else:
        estimated_cost_2, batching_option_matrix = None, None
    
    return estimated_cost_1, batch_indices, estimated_cost_2, batching_option_matrix

def find_optimal_strategy(
    compute_strategy_id_list, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
    multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
    all_devices, local_device, batching_method, 
    strategy_pool, sorted_len
):
    # 先筛选出可以跑当前max_seqlen的strategy
    compute_strategy_id_list = [id for id in compute_strategy_id_list if max(multi_max_seqlen_list[id]) >= sorted_len[-1]]

    # Prepare arguments for multiprocessing
    args_list = [
        (
            compute_strategy_id, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
            multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
            all_devices, local_device, batching_method, 
            strategy_pool, sorted_len
        )
        for compute_strategy_id in compute_strategy_id_list
    ]

    # print(f"Simutaneously handle strategies {compute_strategy_id_list}")
    # Determine the number of threads to use
    num_threads = min(len(args_list), 32) 
    # Use ThreadPoolExecutor to parallelize the strategy processing
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_strategy, args_list))
    end_time = time.time()
    print(f"Find optimal seqs-assigning & seqs-batching strategy time cost: {end_time - start_time}s")
    
    # Unpack the results
    estimated_cost_1_list = [res[0] for res in results]
    batch_indices_list = [res[1] for res in results]
    estimated_cost_2_list = [res[2] for res in results]
    batching_option_matrix_list = [res[3] for res in results]
    
    # 依据estimated_cost_list中最小的值取出四个list对应的各个值
    min_cost_index = None
    if any(estimated_cost_2_list):
        print(f"compute_strategy_id_list = {compute_strategy_id_list}, estimated_cost_2_list = {estimated_cost_2_list} , estimated_cost_1_list = {estimated_cost_1_list}")
        min_cost_index = np.argmin([cost for cost in estimated_cost_2_list if cost is not None])
    else:
        print(f"compute_strategy_id_list = {compute_strategy_id_list}, estimated_cost_1_list = {estimated_cost_1_list}")
        min_cost_index = np.argmin(estimated_cost_1_list)
    
    optimal_estimated_cost_1 = estimated_cost_1_list[min_cost_index]
    optimal_batch_indices = batch_indices_list[min_cost_index]
    optimal_estimated_cost_2 = estimated_cost_2_list[min_cost_index]
    optimal_batching_option_matrix = batching_option_matrix_list[min_cost_index]
    
    return (compute_strategy_id_list[min_cost_index], optimal_estimated_cost_1, optimal_batch_indices, 
            optimal_estimated_cost_2, optimal_batching_option_matrix)
