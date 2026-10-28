import json
import re
from pathlib import Path

class SimpleTokenizer:
    """
    Word-level tokenizer matching the vocabulary and preprocessing 
    defined in Stage 1 of the DNN Assignment.
    """
    def __init__(self, vocab=None, max_length=40):
        self.vocab = vocab or {}
        self.id_to_vocab = {idx: token for token, idx in self.vocab.items()}
        self.max_length = max_length
        self.pad_id = self.vocab.get("<pad>", 0)
        self.unk_id = self.vocab.get("<unk>", 1)
        self.bos_id = self.vocab.get("<bos>", 2)
        self.eos_id = self.vocab.get("<eos>", 3)

    @classmethod
    def load(cls, tokenizer_dir=None):
        if tokenizer_dir is None:
            tokenizer_dir = Path(__file__).resolve().parent
        else:
            tokenizer_dir = Path(tokenizer_dir)

        vocab_path = tokenizer_dir / "vocab.json"
        config_path = tokenizer_dir / "tokenizer_config.json"

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        max_length = 40
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                max_length = cfg.get("max_text_length", 40)

        return cls(vocab=vocab, max_length=max_length)

    def tokenize_text(self, text):
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())

    def encode(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_length
        tokens = self.tokenize_text(text)
        ids = [self.bos_id] + [self.vocab.get(t, self.unk_id) for t in tokens[:max_length-2]] + [self.eos_id]
        ids = ids[:max_length]
        return ids + [self.pad_id] * (max_length - len(ids))

    def decode(self, token_ids):
        tokens = [self.id_to_vocab.get(int(i), "<unk>") for i in token_ids]
        return " ".join([t for t in tokens if t not in ["<pad>", "<bos>", "<eos>"]])
