from functools import partial

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from template.config import Config


def make_tokenizer(dataset: Dataset, cfg: Config) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    unk_token, cls_token, sep_token, pad_token, mask_token = special_tokens
    trainer = BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,  # type: ignore
        special_tokens=special_tokens,  # type: ignore
    )
    tokenizer.train_from_iterator(dataset["text"], trainer, length=len(dataset))
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)  # type: ignore
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore
    cls_token_id = tokenizer.token_to_id(cls_token)
    sep_token_id = tokenizer.token_to_id(sep_token)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{cls_token} $A {sep_token}",
        special_tokens=[(cls_token, cls_token_id), (sep_token, sep_token_id)],
    )  # type: ignore
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=cls_token,
        eos_token=sep_token,
        unk_token=unk_token,
        pad_token=pad_token,
        cls_token=cls_token,
        sep_token=sep_token,
        mask_token=mask_token,
        padding_side="right",
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
    splits.save_to_disk(cfg.dataset.path)
    return splits
