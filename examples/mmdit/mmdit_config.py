import hetu

class MMDiTConfig(object):
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        ffn_hidden_size=-1,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="relu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        use_flash_attn = False,
        patch_size = 2,
        in_channels = 3,
        frequency_embedding_size = 64,
        block_out_channels = [128,256,512,512],
        layers_per_block = 2,
        lora_dtype="fp32",
        dqtype="fp32",
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        if ffn_hidden_size == -1:
            self.ffn_hidden_size = 4 * self.hidden_size
        else:
            self.ffn_hidden_size = ffn_hidden_size
        self.num_attention_heads = self.n_head
        self.num_hidden_layers = self.n_layer
        self.add_cross_attention = False
        self.use_return_dict = False
        self.output_attentions = False
        self.output_hidden_states= False
        self.use_flash_attn = use_flash_attn
        self.patch_size = patch_size
        self.in_channels = in_channels 
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.frequency_embedding_size = frequency_embedding_size
        if lora_dtype == "fp16":
            self.lora_dtype = hetu.float16
        elif lora_dtype == "bf16":
            self.lora_dtype = hetu.bfloat16
        elif lora_dtype == "fp4":
            self.lora_dtype = hetu.float4
        elif lora_dtype == "nf4":
            self.lora_dtype = hetu.nfloat4
        else:
            self.lora_dtype = hetu.float32
        if dqtype == "fp16":
            self.dqtype = hetu.float16
        elif dqtype == "bf16":
            self.dqtype = hetu.bfloat16
        else:
            self.dqtype = hetu.float32