MODEL_SIZE=${1:-"32b"}
GLOBAL_BATCH_SIZE=-1 # 目前改用gtb代替gbs
GLOBAL_TOKEN_NUM=${2:--1}
GLOBAL_TOKEN_NUM=${2:-200000}
MAX_SEQ_LEN=${3:-32768}
# MAX_SEQ_LEN=${3:-16384}
# MAX_SEQ_LEN=${3:-65536}
# MAX_SEQ_LEN=${3:-131072}
SERVER_ADDR=${4:-"30.207.98.114"} # A800-0
# SERVER_ADDR=${4:-"30.207.99.202"} # A800-2
SERVER_PORT=${5:-"23333"}
HOST_FILE_PATH=${6:-"/jizhicfs/hymiezhao/lhy/hostfiles/host01234567.yaml"}
# HOST_FILE_PATH=${6:-"/jizhicfs/hymiezhao/lhy/hostfiles/host2367.yaml"}
ENV_FILE_PATH=${7:-"./scripts/env_A800.sh"}
STRATEGY_POOL_PATH=${8:-"./strategy/strategy_pool_32b.json"}

BEGIN_STEP=0
TORCH_PROFILE=0
CASE=5
WARM_UP=1
COMPUTE_ONLY=0
if [[ ${CASE} -eq 0 ]]; then
    # test
	NUM_GPUS=64
        MULTI_TP_PP_LIST="[[(16, 1), (16, 1), (16, 1), (16, 1)]]"
        # MULTI_TP_PP_LIST="[[(16, 1), (16, 1), (16, 1), (16, 1)], [(16, 1), (8, 1), (8, 1), (8, 4)], [(16, 1), (8, 3), (8, 3)], [(8, 4), (8, 4)], [(8, 2), (8, 2), (8, 2), (8, 2)], [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2)], [(4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)]]"
	# MULTI_TP_PP_LIST="[[(8, 1), (8, 1)], [(8, 1), (1, 2), (1, 2), (1, 2), (1, 2)], [(4, 1), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]]"
        BATCHING_METHOD=4
elif [[ ${CASE} -eq 1 ]]; then
    # homo + greedy packing with static shape
	NUM_GPUS=32
        MULTI_TP_PP_LIST="[[(4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)]]"
        # MULTI_TP_PP_LIST="[[(16, 1)], [(8, 1), (8, 1)], [(4, 1), (4, 1), (4, 1), (4, 1)]]"
        # MULTI_TP_PP_LIST="[[(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]]"
        # MULTI_TP_PP_LIST="[[(4, 1), (4, 1), (4, 1), (4, 1)], [(4, 1), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]]"
        # MULTI_TP_PP_LIST="[[(2, 1), (2, 1), (2, 1), (2, 1)]]"
	# MULTI_TP_PP_LIST="[[(4, 1), (4, 1)], [(4, 1), (1, 2), (1, 2)], [(2, 2), (2, 2)], [(2, 2), (1, 2), (1, 2)], [(2, 1), (2, 1), (2, 1), (2, 1)]]"
        BATCHING_METHOD=4
elif [[ ${CASE} -eq 2 ]]; then	
    # homo + greedy packing with dynamic shape
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=3
elif [[ ${CASE} -eq 3 ]]; then	
    # homo + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 4 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 8)], ]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 5 ]]; then	
    # hetero + hydraulis packing
	NUM_GPUS=64
        # MULTI_TP_PP_LIST="[[(16, 1), (16, 1), (16, 1), (16, 1)]]"
        MULTI_TP_PP_LIST="[[(16, 1), (16, 1), (16, 1), (16, 1)], [(16, 1), (8, 1), (8, 1), (8, 1), (8, 1), (8, 1), (8, 1)], [(16, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)], [(8, 3), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)], [(8, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)], [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2)]]"
	# MULTI_TP_PP_LIST="[[(16, 1), (16, 1), (16, 1), (16, 1)],  [(16, 1), (8, 1), (8, 1), (8, 4)], [(16, 1), (8, 1), (8, 1), (8, 1), (8, 1), (8, 1), (8, 1)], [(16, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1)], [(8, 2), (8, 2), (8, 2), (8, 2)], [(4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2), (4, 2)]]"
        BATCHING_METHOD=4
else
    echo unknown CASE
	exit 1
fi

echo num_gpus=${NUM_GPUS}, global_token_num = ${GLOBAL_TOKEN_NUM}, max_seq_len = ${MAX_SEQ_LEN}

if [ "${MODEL_SIZE}" = "7b" ]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    FFN_HIDDEN_SIZE=11008
    NUM_HEADS=32
elif [ "${MODEL_SIZE}" = "13b" ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    FFN_HIDDEN_SIZE=13824
    NUM_HEADS=40
elif [ "${MODEL_SIZE}" = "32b" ]; then
    NUM_LAYERS=60
    HIDDEN_SIZE=6656 #6672
    # HIDDEN_SIZE=512
    # FFN_HIDDEN_SIZE=2752
    FFN_HIDDEN_SIZE=17920
    NUM_HEADS=64
elif [ "${MODEL_SIZE}" = "70b" ]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192 #6672
    FFN_HIDDEN_SIZE=28672
    NUM_HEADS=64
else
    echo the model should be 7b/13b/32b/70b for test.
    exit 0
fi

# 请注意log编号目前并不等于rank编号
LOG_FOLDER=logs/case${CASE}/llama${MODEL_SIZE}_gpus${NUM_GPUS}_gtn${GLOBAL_TOKEN_NUM}_msl${MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/jizhicfs/hymiezhao/lhy/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
# JSON_FILE=${ROOT_FOLDER}/code/code.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u train_hetu.py \
--warm_up $WARM_UP \
--compute_only $COMPUTE_ONLY \
--torch_profile $TORCH_PROFILE \
--batching_method $BATCHING_METHOD \
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_token_num $GLOBAL_TOKEN_NUM \
--max_seq_len $MAX_SEQ_LEN \
--strategy_pool $STRATEGY_POOL_PATH \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 1 \
--steps 100 \
--begin_step $BEGIN_STEP \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
python3 ../../python_refactor/hetu/rpc/pssh_start.py \
        --hosts ${HOST_FILE_PATH} \
        --command "$CMD" \
        --server_port ${SERVER_PORT} \
        --ngpus ${NUM_GPUS} \
        --envs ${ENV_FILE_PATH} \
        --log_path ${LOG_FOLDER}
