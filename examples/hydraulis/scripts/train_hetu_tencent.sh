MODEL_SIZE=${1:-"32b"}
GLOBAL_BATCH_SIZE=-1 # 目前改用gtb代替gbs
GLOBAL_TOKEN_NUM=${2:-100000}
MAX_SEQ_LEN=${3:-8192}
SERVER_ADDR=${4:-"30.207.99.90"} # A800-0
# SERVER_ADDR=${4:-"30.207.96.91"} # A800-1
# SERVER_ADDR=${4:-"30.207.98.74"} # A800-2
# SERVER_ADDR=${4:-"30.207.98.114"} # A800-3
# SERVER_ADDR=${4:-"30.207.96.39"} # A800-4
# SERVER_ADDR=${4:-"30.207.98.70"} # A800-5
# SERVER_ADDR=${4:-"30.207.98.231"} # A800-6
# SERVER_ADDR=${4:-"30.207.98.69"} # A800-7
SERVER_PORT=${5:-"23333"}
HOST_FILE_PATH=${6:-"/jizhicfs/hymiezhao/hostfiles/host01.yaml"}
ENV_FILE_PATH=${7:-"./scripts/env_A800.sh"}

TORCH_PROFILE=0
CASE=0
if [[ ${CASE} -eq 0 ]]; then
	# test
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], [(4, 2), (1, 8)], [(4, 2), (1, 4), (1, 4)]]"
	BATCHING_METHOD=4
elif [[ ${CASE} -eq 1 ]]; then
	# homo + greedy packing with static shape
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (4, 2)], ]"
	BATCHING_METHOD=2
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
	NUM_GPUS=16
	MULTI_TP_PP_LIST="[[(4, 2), (1, 4), (1, 4)], ]"
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
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u train_hetu.py \
--torch_profile $TORCH_PROFILE \
--batching_method $BATCHING_METHOD \
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_token_num $GLOBAL_TOKEN_NUM \
--max_seq_len $MAX_SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
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
if [ ${NUM_GPUS} -gt 8 ]; then
python3 ../../python_refactor/hetu/rpc/pssh_start.py \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
else
python3 ../../python_refactor/hetu/rpc/pssh_start.py \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
fi
