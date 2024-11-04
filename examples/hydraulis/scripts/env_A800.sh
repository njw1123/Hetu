source /jizhicfs/hymiezhao/lhy/hydraulis_start.sh

# 务必要取消代理
# 不然grpc会走代理出现连接不上的问题
unset http_proxy
unset https_proxy

export PATH="/jizhicfs/pinxuezhao/miniconda3/envs/hetu-grpc/bin:${PATH}"

export HETU_P2P=SINGLE_COMMUNICATOR
export HETU_BRIDGE=SINGLE_COMMUNICATOR
# export HETU_OVERLAP_GRAD_REDUCE=FIRST_STAGE
export HETU_SHAPE_MISMATCH=BRIDGE_SUBGRAPH
export HETU_SWITCH_ALGORITHM=NEW_GREEDY
export HETU_SWITCH_PROFILE=TIME
export HETU_INTERNAL_LOG_LEVEL=INFO
export HETU_STRAGGLER=ANALYSIS

export HETU_MEMORY_PROFILE=WARN
# export HETU_MAX_SPLIT_SIZE_MB=200
# export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=20
export HETU_MAX_SPLIT_SIZE_MB=10240
export HETU_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
export HETU_PRE_ALLOCATE_SIZE_MB=27000

# Using multi-stream cuda event to watch time elaspe is inaccurate!
# export HETU_PARALLEL_ATTN=ANALYSIS
export HETU_PARALLEL_ATTN_SPLIT_PATTERN=NORMAL

# 设置网络接口相关的环境变量
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA="mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"

# 设置NCCL相关的环境变量
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_DEBUG=VERSION
export NCCL_NET_GDR_READ=1
export NCCL_SOCKET_NTHREADS=8
export NCCL_COLLNET_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0

# SHARP相关的环境变量
export SHARP_COLL_ENABLE_SAT=0

# CUDA相关的环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# OpenMPI相关配置
export OMPI_MCA_btl_tcp_if_include=bond1
export OMPI_MCA_oob_tcp_if_include=bond1

echo "env done"
