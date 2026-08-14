"""
Minimal GPT-2 tokenizer shim. GPT2Tokenizer/GPT2TokenizerFast both come back
with vocab_size=0 / empty output under this environment's transformers
version (5.10.2 system, 4.57.6 in .venv -- different errors, same root
symptom), even though the underlying tokenizer.json in the HF cache is fine
(verified directly via the raw `tokenizers` library). data/gpt2_programs.py
only ever calls tokenizer(...).input_ids and tokenizer.convert_ids_to_tokens,
so this shim covers exactly that surface using the raw tokenizers lib.
"""
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import torch


class _Encoding:
    def __init__(self, ids, as_tensor):
        self.input_ids = torch.tensor([ids], dtype=torch.long) if as_tensor else ids

    def __getitem__(self, key):
        if key == 'input_ids':
            return self.input_ids
        raise KeyError(key)

    def keys(self):
        return ['input_ids']

    def to(self, device):
        if torch.is_tensor(self.input_ids):
            self.input_ids = self.input_ids.to(device)
        return self


class GPT2TokShim:
    def __init__(self):
        path = hf_hub_download('gpt2', 'tokenizer.json')
        self._tok = Tokenizer.from_file(path)
        self.pad_token = None
        self.eos_token = '<|endoftext|>'

    def __call__(self, sentence, return_tensors=None, truncation=False, max_length=None):
        ids = self._tok.encode(sentence).ids
        if max_length is not None:
            ids = ids[:max_length]
        return _Encoding(ids, as_tensor=(return_tensors == 'pt'))

    def convert_ids_to_tokens(self, ids):
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        return [self._tok.id_to_token(i) for i in ids]

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        return self._tok.decode(ids)
