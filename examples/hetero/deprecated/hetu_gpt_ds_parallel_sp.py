import hetu as ht
import numpy as np

from hetu.nn.modules.parallel_ds import parallel_data_provider, get_device_index


# self-attn
class GPTAttention(ht.nn.Module):
    def __init__(self, config, ds_parallel_config, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.use_flash_attn = config.use_flash_attn
        self.add_bias = False

        max_positions = config.max_position_embeddings
        # self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
        #             1, 1, max_positions, max_positions))
        self.masked_value = -1e4

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.qkv_dense = ht.nn.HtColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            ds_parallel_config['qkv'],
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
        )

        self.dense = ht.nn.HtRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            ds_parallel_config['dense'],
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        self.attn_dropout = ht.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = ht.nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key_t, value, attention_mask=None):
        # q*k^T, shape=[mbs_times_dp, num_heads, seq_len, seq_len]
        attn_weights = ht.bmm(query, key_t)
        mbs_times_dp, num_heads, seq_len, seq_len = attn_weights.global_shape

        # scale
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.global_shape[-1]) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # mask
        device_index = get_device_index(self.qkv_dense.device_group)
        causal_mask = ht.from_numpy_parallel(parallel_data_provider(np.tile(self.bias[:, :, :seq_len, :seq_len], 
                                                                            (mbs_times_dp, num_heads, 1, 1)),
                                                                    attn_weights.distributed_states, device_index),
                                             attn_weights.distributed_states, requires_grad=False,
                                             device_group=self.qkv_dense.device_group, name='causal_mask')
        mask = ht.from_numpy_parallel(parallel_data_provider(np.full(attn_weights.global_shape, self.masked_value, dtype=np.float32),
                                                             attn_weights.distributed_states, device_index), 
                                      attn_weights.distributed_states, requires_grad=False,
                                      device_group=self.qkv_dense.device_group, name='mask')
        attn_weights = ht.where(causal_mask, attn_weights, mask)
        if attention_mask is not None:
            # attn_weights: shape=[mbs_times_dp, num_heads, seq_len, seq_len]
            # attention_mask: shape=[mbs_times_dp, 1, 1, seq_len], 注意ds的设置
            # 被mask的<pad>位置上值为-1e4, 没有被mask的位置上值为0
            # todo: +-*/允许对应维度一个为n一个为1的情况下, n被切分
            # print(f'attn_weights global_shape={attn_weights.global_shape}, attention_mask.global_shape={attention_mask.global_shape}')
            # print(f'attn_weights shape={attn_weights.shape}, attention_mask.shape={attention_mask.shape}')
            attn_weights = attn_weights + attention_mask
        # softmax
        attn_weights = ht.softmax(attn_weights, 3)
        # weight sum, shape=[mbs_times_dp, num_heads, seq_len, head_dim]
        attn_output = ht.bmm(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        mbs_times_dp, # bsz dimension for flash-attn
        attention_mask=None,
    ):
        # column parallel, [mbs_times_dp*seq_len, 3*embed_dim]
        qkv = self.qkv_dense(hidden_states)

        # flash need bsz dimension!!!
        # [mbs_times_dp, seq_len, num_heads, 3*head_dim]
        qkv = qkv.reshape([mbs_times_dp, -1, self.num_heads, 3 * self.head_dim])
        # q,k,v shape=[mbs_times_dp, seq_len, num_heads, head_dim]
        query, key, value = ht.split(qkv, 3, qkv.ndim - 1)

        if self.use_flash_attn:
            attn_output = ht.attn(query, key, value, 0, -1, True)[0]
        else:
            # [mbs_times_dp, num_heads, seq_len, head_dim]
            query = query.transpose([0, 2, 1, 3], name="AttentionOp_query")
            value = value.transpose([0, 2, 1, 3], name="AttentionOp_value")
            # [mbs_times_dp, num_heads, head_dim, seq_len]
            key_t = key.transpose([0, 2, 3, 1], name="AttentionOp_key") # k^T

            # self-attn, shape=[mbs_times_dp, num_heads, seq_len, head_dim]
            attn_output, attn_weights = self._attn(query, key_t, value, attention_mask)

            # [mbs_times_dp, seq_len, num_heads, head_dim]
            attn_output = attn_output.transpose([0, 2, 1, 3])
        
        # [mbs_times_dp*seq_len, num_heads*head_dim]
        attn_output = attn_output.reshape([-1, self.num_heads * self.head_dim])
        # row parallel, shape=[mbs_times_dp*seq_len, num_heads*head_dim]
        attn_output = self.dense(attn_output)

        return attn_output



class ParallelMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_config, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.add_bias = False

        self.dense_h_to_4h = ht.nn.HtColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            ds_parallel_config['dense_h_to_4h'],
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        # self.activation_func = ht.nn.NewGeLU() # should be gelu

        self.dense_4h_to_h = ht.nn.HtRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            ds_parallel_config['dense_4h_to_h'],
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = ht.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b*seq_len, h] -> [b*seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        intermediate_parallel = ht.relu(intermediate_parallel)

        # [b*seq_len, 4h] -> [b*seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

class GPTMLP(ht.nn.Module):
    def __init__(self, config, ds_parallel_config, name='mlp'):
        super(GPTMLP, self).__init__()
        self.config = config
        self.parallel_mlp = ParallelMLP(config, ds_parallel_config, name)

    def forward(self, hidden_states):
        # [bsz*seq_len, hidden_size]
        hidden_states = self.parallel_mlp(hidden_states)
        return hidden_states

class GPTBlock(ht.nn.Module):
    def __init__(self, config, ds_parallel_config, layer_idx):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # sequence parallel: layernorm前做scatter(这一部分由row prallel的reduce-scatter完成); layernorm后做allgather
        self.ln_1 = ht.nn.HtParallelLayerNorm(hidden_size, ds_parallel_config['layernorm1'], sequence_parallel=True, eps=config.layer_norm_epsilon, name=f'ln1_block{layer_idx}')
        self.attn = GPTAttention(config, ds_parallel_config['attn'], layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.ln_2 = ht.nn.HtParallelLayerNorm(hidden_size, ds_parallel_config['layernorm2'], sequence_parallel=True, eps=config.layer_norm_epsilon, name=f'ln2_block{layer_idx}')
        self.mlp = GPTMLP(config, ds_parallel_config['mlp'], name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        mbs_times_dp,
        attention_mask=None,
    ):
        # [bsz*seq_len, embed_dim]
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [bsz*seq_len, hidden_size]
            mbs_times_dp,
            attention_mask=attention_mask, # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        # hidden_states =  feed_forward_hidden_states + residual
        hidden_states =  residual + feed_forward_hidden_states

        return hidden_states


class GPTModel(ht.nn.Module):
    def __init__(self, config, ds_parallel_config):
        super(GPTModel, self).__init__()
        self.config = config
        self.dtype = ht.float32

        self.embed_dim = config.hidden_size
        self.wte = ht.nn.HtVocabParallelEmbedding(config.vocab_size, self.embed_dim, ds_parallel_config['wte'], name='wte')
        self.wpe = ht.nn.HtParallelEmbedding(config.max_position_embeddings, self.embed_dim, ds_parallel_config['wpe'], name='wpe')

        self.drop = ht.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
            for _, block_config in ds_parallel_config['blocks'].items():
                if i >= block_config['range'][0] and i <= block_config['range'][1]:
                    blocks.append(GPTBlock(config, block_config, layer_idx=i))
                    break
        self.h = ht.nn.ModuleList(blocks)
        self.ln_f = ht.nn.HtParallelLayerNorm(self.embed_dim, ds_parallel_config['layernorm_final'], sequence_parallel=True, eps=config.layer_norm_epsilon, name='ln_final')

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b, seq_len]
        mbs_times_dp, seq_len = input_ids.global_shape
        
        # token_type_ids: [b, seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # position_ids: [1, seq_len]
        position_ids = np.arange(0, seq_len, dtype=np.int64) # pos: [0, 1, 2, ..., seq_len-1]
        position_ids = np.tile(position_ids, [mbs_times_dp, 1]) # shape: [b, seq_len]
        # position_ids = ht.from_numpy(position_ids)
        device_index = get_device_index(self.wpe.device_group)
        position_ids = ht.from_numpy_parallel(parallel_data_provider(position_ids, input_ids.distributed_states, device_index), 
                                              input_ids.distributed_states, device_group=self.wpe.device_group, name='position_ids')

        # attention_mask: [b, 1, 1, seq_len]
        if attention_mask is not None:
            assert attention_mask.global_shape == input_ids.global_shape \
                and attention_mask.distributed_states.check_equal(attention_mask.distributed_states), \
                'attention_mask global_shape and distributed_states should be equal to input_ids!'
            attention_mask = attention_mask.reshape([mbs_times_dp, 1, 1, -1])
            # 原attention_mask: 1为使用的值, 0为mask的值
            # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0 # 0为使用的值, -10000为mask的值

        # embeddding: [b, seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b, seq_len, embed_dim]
        position_embeds = self.wpe(position_ids) # [b, seq_len, embed_dim]
        # todo: fix backward grad tensor reduce bug for add(extension dims)
        hidden_states = inputs_embeds + position_embeds # [b, seq_len, embed_dim]
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b, seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds

        # [bsz, seq_len, embed_dim] -> [bsz*seq_len, embed_dim]
        _, _, embed_dim = hidden_states.global_shape
        hidden_states = hidden_states.reshape([-1, embed_dim])

        # for sequence parallel
        # todo: this is pretty hacky, find a better way
        sp = True
        if sp:
            ds_input = hidden_states.distributed_states
            ds_split0 = ht.DistributedStates(ds_input.device_num, {0: ds_input.device_num}, [0])
            assert ds_input.device_num == ds_input.get_dim(0) * ds_input.get_dim(-1), \
                'Workaround: sp assume input only split in dimension 0 for dp'
            if ds_input.get_dim(-1) > 1: # exists dup input for tp
                # [b*seq_len // tp, embed_dim]
                hidden_states = ht.comm(hidden_states, ds_split0)

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b*seq_len, embed_dim]
                mbs_times_dp,
                attention_mask=attention_mask, # [b, 1, 1, seq_len]
            )
        # layernorm
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class GPTLMHeadModel(ht.nn.Module):

    def __init__(self, config, ds_parallel_config):
        super(GPTLMHeadModel, self).__init__()
        self.transformer = GPTModel(config, ds_parallel_config['gpt'])
        self.lm_head = ht.nn.HtColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            ds_parallel_config['lm_head'],
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        self.lm_head.weight = self.transformer.wte.embedding_table # share embedding table
        self.config = config
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        mbs_times_dp, _ = input_ids.global_shape
        # [b, s] -> [b*s, h] -> [b*s//tp, h]
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # need allgather here: [b*s//tp, h] -> [b*s, h]
        if not hidden_states.distributed_states.check_equal(self.lm_head.ds_map['split0_dup']):
            hidden_states = ht.comm(hidden_states, self.lm_head.ds_map['split0_dup'])
        # [b*s, h] -> [b, s, h]
        _, hidden_size = hidden_states.global_shape
        hidden_states = hidden_states.reshape([mbs_times_dp, -1, hidden_size])
        # [b, s, h] -> [b, s-1, h]
        shift_hidden_states = ht.slice(hidden_states, [0,0,0], [hidden_states.shape[0], hidden_states.shape[1] - 1, hidden_states.shape[2]])
        # [b*(s-1), h]
        shift_hidden_states = shift_hidden_states.reshape([-1, hidden_size])

        # column parallel, [b*(s-1), h]->[b*(s-1), vocab_size], and splited in vocab dimension
        shift_lm_logits = self.lm_head(shift_hidden_states)

        loss = None
        if labels is not None:
            # lm_logits: [b*(s-1), vocab_size], labels: [b, s-1]
            shift_labels = ht.slice(labels, [0,1], [labels.shape[0], labels.shape[1] - 1])
            print(f'shift_lm_logits shape = {shift_lm_logits.shape}, ds = {shift_lm_logits.distributed_states}; shift_labels shape = {shift_labels.shape}, ds = {shift_labels.distributed_states}')

            if shift_lm_logits.distributed_states.get_dim(1) > 1:
                # print('use vocab parallel cross entropy')
                loss = ht.vocab_parallel_cross_entropy(shift_lm_logits,  
                    shift_labels, ignored_index = -1, reduction = "mean")
            else:
                # print('use commom cross entropy')
                loss = ht.softmax_cross_entropy_sparse(shift_lm_logits,
                    shift_labels, ignored_index = -1, reduction = "mean")
        # output = (shift_lm_logits,)
        # output = ((loss,) + output) if loss is not None else output
        return loss # ((loss), (shift_lm_logits))