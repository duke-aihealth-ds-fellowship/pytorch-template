# type: ignore
from functools import partial

from datasets import DatasetDict
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from template.config import Config


def make_tokenizer(text, cfg: Config) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=[cfg.tokenizer.pad_token],
    )
    tokenizer.train_from_iterator(text, trainer=trainer, length=len(text))
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=cfg.tokenizer.pad_token
    )
    tokenizer.save_pretrained(cfg.tokenizer.path)
    return tokenizer


def tokenize_fn(batch, tokenizer: PreTrainedTokenizerFast, max_length: int):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def tokenize_dataset(tokenizer, splits: DatasetDict, cfg: Config) -> DatasetDict:
    tokenize_text = partial(
        tokenize_fn, tokenizer=tokenizer, max_length=cfg.tokenizer.max_length
    )
    splits = splits.map(tokenize_text, batched=True)
    columns = ["label", "input_ids", "attention_mask"]
    splits = splits.with_format("torch", columns=columns)
    return splits
