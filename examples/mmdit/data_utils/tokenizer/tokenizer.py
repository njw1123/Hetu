"""tokenizers."""

from abc import ABC
from abc import abstractmethod

from .gpt2_tokenization import GPT2Tokenizer

import re


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))
    
    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class HetuTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))

class _GPT2BPETokenizer(HetuTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = 'GPT2 BPE'
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=['<pad>'], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']
        self.pad_id = self.tokenizer.special_tokens['<pad>']

        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.special_tokens = [self.image_token, self.video_token]

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):

        print("tokenizer ", text)

        special_tokens_regex = re.compile(
            "|".join(re.escape(token) for token in self.special_tokens)
        )
        segments = special_tokens_regex.split(text)
        matches = special_tokens_regex.findall(text)        

        segments = special_tokens_regex.split(text)
        matches = special_tokens_regex.findall(text)

        result = []
        for segment, match in zip(segments, matches + [""]):  # 添加空字符串以匹配最后一个 segment
            if segment.strip():  # 如果是普通文本
                result.extend(self.tokenizer.encode(segment))
            if match:  # 如果是特殊标记
                result.append(self.convert_tokens_to_ids(match))
        
        return result

    def convert_tokens_to_ids(self, match):
        if match == self.image_token:
            return -200
        elif match == self.video_token:
            return -201

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
    
    @property
    def pad(self):
        return self.pad_id