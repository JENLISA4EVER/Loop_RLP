import glob
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import Dataset
import os
import json
import torch
import torch.nn.functional as F
from accelerate.utils import DataLoaderConfiguration
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from transformers import PreTrainedTokenizer,  AutoTokenizer
from dataclasses import dataclass
import time
import logging
logger = logging.getLogger(__name__)

class OmniMathGenerationDataset(Dataset):
    def __init__(
        self, data_source, tokenizer, max_length, apply_chat=False, prompt_key="problem",
        response_key="response",truncation="left",max_num=None
    ):
        """
        data_source: 可以是文件路径(str) 或 已经加载好的 list
        """
        self.data = []
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # RLP将整个语料视为序列 [cite: 6863-6866]
                        self.data.append(item)
                    except:
                        continue
        elif isinstance(data_source, list):
            for file_path in data_source:
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except:
                            continue
        if max_num is not None and len(self.data) > max_num:
            self.data = self.data[:max_num]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat = apply_chat
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.truncation = truncation
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #reference: verl/sft_dataset.py
        # 1. 分别根据prompt_key和response_key获取 prompt 和 response
        item = self.data[idx]
        prompt = item[self.prompt_key]
        response = item[self.response_key]

        # 2. string
        if self.apply_chat:
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_chat,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # 和之前没chat_template一样，只是分开编码prompt和response
            prompt_str = f"Problem: {prompt}\nSolution: "    
        response_str = response #FIXME:不是sft，不需要加eos_token
        
        # 3. tokenize
        prompt_ids_output = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        prompt_length = prompt_ids.shape[0]
        
        input_ids = prompt_ids #generation不需要response_ids
        attention_mask = prompt_attention_mask #generation不需要response_attention_mask
        
        # 4. padding to max_length
        sequence_length = prompt_length
        if self.truncation in ["right", "left"]:
            if sequence_length < self.max_length:
                padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                            dtype=input_ids.dtype) * self.tokenizer.pad_token_id
                padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
                input_ids = torch.cat((input_ids, padded_input_ids), dim=-1)
                attention_mask = torch.cat((attention_mask, padded_attention_mask), dim=-1)
            elif sequence_length > self.max_length:
                assert self.truncation == "right", "truncation must be right"
                if self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                elif self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                else:
                    raise ValueError(f"Invalid truncation: {self.truncation}")
        # 5. compute position_ids
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        
        position_ids = compute_position_id_with_mask(attention_mask)

        # 6. loss mak 
        loss_mask = attention_mask.clone() #和之前的保存一致，但是不适用

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": response_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
class OmniMathDataset(Dataset):
    def __init__(
        self, data_source, tokenizer, max_length, apply_chat=False, prompt_key="problem",
        response_key="solution",truncation="left",max_num=None
    ):
        """
        data_source: 可以是文件路径(str) 或 已经加载好的 list
        """
        self.data = []
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # RLP将整个语料视为序列 [cite: 6863-6866]
                        self.data.append(item)
                    except:
                        continue
        elif isinstance(data_source, list):
            for file_path in data_source:
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except:
                            continue
        if max_num is not None and len(self.data) > max_num:
            self.data = self.data[:max_num]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat = apply_chat
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.truncation = truncation
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #reference: verl/sft_dataset.py
        # 1. 分别根据prompt_key和response_key获取 prompt 和 response
        item = self.data[idx]
        prompt = item[self.prompt_key]
        response = item[self.response_key]

        # 2. string
        if self.apply_chat:
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_chat,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # 和之前没chat_template一样，只是分开编码prompt和response
            prompt_str = f"Problem: {prompt}\nSolution: "    
        response_str = response + self.tokenizer.eos_token #FIXME:ouro的tokenizer.eos_token和config.eos_token不一样
        
        # 3. tokenize
        prompt_ids_output = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        
        # 4. padding to max_length
        sequence_length = prompt_length + response_length
        if self.truncation in ["right", "left"]:
            if sequence_length < self.max_length:
                padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                            dtype=input_ids.dtype) * self.tokenizer.pad_token_id
                padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
                input_ids = torch.cat((input_ids, padded_input_ids), dim=-1)
                attention_mask = torch.cat((attention_mask, padded_attention_mask), dim=-1)
            elif sequence_length > self.max_length:
                assert self.truncation == "right", "truncation must be right"
                if self.truncation == "right":
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                elif self.truncation == "left":
                    input_ids = input_ids[-self.max_length:]
                    attention_mask = attention_mask[-self.max_length:]
                else:
                    raise ValueError(f"Invalid truncation: {self.truncation}")
        # 5. compute position_ids
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        
        position_ids = compute_position_id_with_mask(attention_mask)

        # 6. loss mak
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt loss.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0 #XXX:这里也要注意shift_labels
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0 #FIXME:不确定

        labels = input_ids.clone()
        # labels[labels == self.tokenizer.pad_token_id] = -100 #FIXME:有了loss_mask，这里还需要吗？
        #FIXME:另外用collate_fn处理，采用当前batch中最大的sequence_length作为max_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
class OmniMathNTPDataset(Dataset):
    '''
    OmniMathNTPDataset is used to evaluate the NTP mode on the OmniMath dataset.
    Eacy data instance is a prefix and a target token.
    '''
    
    def __init__(
        self, data_source, tokenizer, max_length, apply_chat=False, prompt_key="problem",
        response_key="solution",truncation="left"
    ):
        """
        data_source: 可以是文件路径(str) 或 已经加载好的 list
        """
        self.data = []
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # RLP将整个语料视为序列 [cite: 6863-6866]
                        self.data.append(item)
                    except:
                        continue
        elif isinstance(data_source, list):
            for file_path in data_source:
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except:
                            continue

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat = apply_chat
        self.prompt_key = prompt_key
        self.response_key = response_key
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #reference: verl/sft_dataset.py
        # 1. 分别根据prompt_key和response_key获取 prompt 和 response
        item = self.data[idx]
        prompt = item[self.prompt_key]
        response = item[self.response_key]

        # 2. string
        # 没有string过程，输入的数据是已经加入模板的字符串      
        # 3. tokenize
        prompt_ids_output = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        
        # 4. padding to max_length
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        
        position_ids = compute_position_id_with_mask(attention_mask)

        # 6. loss mak
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt loss.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0 #XXX:这里也要注意shift_labels
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0 #FIXME:不确定

        labels = input_ids.clone()
        # labels[labels == self.tokenizer.pad_token_id] = -100 #FIXME:有了loss_mask，这里还需要吗？
        #FIXME:另外用collate_fn处理，采用当前batch中最大的sequence_length作为max_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
class OpenThoughtsDataset(Dataset):
    def __init__(
        self, data_files, tokenizer, apply_chat=False,truncation="left", conversation_key="conversations", role_key="role", user_key="user", assistant_key="assistant", content_key="content"
    ):
        self.data = []
        if isinstance(data_files,list):
            for file in data_files:
                if file.endswith(".jsonl"):
                    with open(file, "r") as f:
                        from tqdm import tqdm
                        for line in tqdm(f):
                            try:
                                item = json.loads(line)
                                self.data.append(item)
                            except:
                                continue
                elif file.endswith(".parquet"):
                    df = pd.read_parquet(file)
                    self.data.extend(df.to_dict(orient="records"))
        self.tokenizer = tokenizer
        self.apply_chat = apply_chat
        self.truncation = truncation
        self.conversation_key = conversation_key
        self.role_key = role_key
        self.user_key = user_key
        self.assistant_key = assistant_key
        self.content_key = content_key
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.conversation_key in item:
            conversations = item[self.conversation_key]
            for conversation in conversations:
                if conversation[self.role_key] == self.user_key:
                    prompt = conversation[self.content_key]
                if conversation[self.role_key] == self.assistant_key:
                    response = conversation[self.content_key]
        else:
            prompt = item[self.prompt_key]
            response = item[self.response_key]
        if self.apply_chat:
            prompt = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = f"Problem: {prompt}\nSolution: "
        response_str = response + self.tokenizer.eos_token
        prompt_ids_output = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        # 5. compute position_ids
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        position_ids = compute_position_id_with_mask(attention_mask)
        # 6. loss mask
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt loss.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0 #XXX:这里也要注意shift_labels
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0 #FIXME:不确定
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

        
@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    reference: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizer
    max_length: int | None = None
    padding_side: str = "right"
    return_tensors: str = "pt"
    truncation: str = "right"
    max_length_method: str = "group_max"
    padding_side: str = "right"
    def __call__(self, features):
        batch_result = {}
        for key in features[0].keys(): #假设返回的features是dict类型，key是feature的名称
            if self.max_length_method == "fix":
                max_length = self.max_length
            elif self.max_length_method == "group_max":
                max_length = min(self.max_length, max(len(feature[key]) for feature in features))
            else:
                raise ValueError(f"Invalid max_length_method: {self.max_length_method}")
            if self.truncation == "right":
                value_list = [feature[key][:max_length] for feature in features]
            elif self.truncation == "left":
                value_list = [feature[key][-max_length:] for feature in features]
            else:
                raise ValueError(f"Invalid truncation: {self.truncation}")
            padding_dict = {
                    "input_ids": self.tokenizer.pad_token_id,
                    "mask": 0,
                    "position": 0,
                    "labels": -100,
                }
            pad_value = None
            for pad_key in padding_dict.keys():
                if pad_key in key: #例如key是input_ids，则padding_dict是input_ids，mask则对应attention_mask/loss_mask
                    pad_value = padding_dict[pad_key]
                    break
            if pad_value is None:
                pad_value = self.tokenizer.pad_token_id
            if self.padding_side == "right":
                value_tensor = torch.stack([F.pad(value, (0, max_length - value.shape[0]), value=pad_value) for value in value_list])
                batch_result[key] = value_tensor
            elif self.padding_side == "left":
                value_tensor = torch.stack([F.pad(value, (max_length - value.shape[0], 0), value=pad_value) for value in value_list])
                batch_result[key] = value_tensor
            else:
                raise ValueError(f"Invalid padding_side: {self.padding_side}")
        return batch_result
    
        
@dataclass
class DataCollatorWithPadding2:
    """
    Data collator that will dynamically pad the inputs received.
    reference: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizer
    max_length: int | None = None
    padding_side: str = "right"
    return_tensors: str = "pt"
    truncation: str = "right"
    max_length_method: str = "group_max"
    padding_side: str = "right"
    def __call__(self, features):
        batch_result = {}
        for key in features[0]:
            if self.max_length_method == "fix":
                max_length = self.max_length
            elif self.max_length_method == "group_max":
                max_length = min(self.max_length, max(len(feature["input_ids"]) for feature in features))
            else:
                raise ValueError(f"Invalid max_length_method: {self.max_length_method}")
            if self.truncation == "right":
                input_ids_list = [feature["input_ids"][:max_length] for feature in features]
                attention_mask_list = [feature["attention_mask"][:max_length] for feature in features]
                loss_mask_list = [feature["loss_mask"][:max_length] for feature in features]
                position_ids_list = [feature["position_ids"][:max_length] for feature in features]
                labels_list = [feature["labels"][:max_length] for feature in features]
            elif self.truncation == "left":
                input_ids_list = [feature["input_ids"][-max_length:] for feature in features]
                attention_mask_list = [feature["attention_mask"][-max_length:] for feature in features]
                loss_mask_list = [feature["loss_mask"][-max_length:] for feature in features]
                position_ids_list = [feature["position_ids"][-max_length:] for feature in features]
                labels_list = [feature["labels"][-max_length:] for feature in features]
            else:
                raise ValueError(f"Invalid truncation: {self.truncation}")
            if self.padding_side == "right":
                input_ids = torch.stack([F.pad(input_ids, (0, max_length - input_ids.shape[0]), value=self.tokenizer.pad_token_id) for input_ids in input_ids_list])
                attention_mask = torch.stack([F.pad(attention_mask, (0, max_length - attention_mask.shape[0]), value=0) for attention_mask in attention_mask_list])
                loss_mask = torch.stack([F.pad(loss_mask, (0, max_length - loss_mask.shape[0]), value=0) for loss_mask in loss_mask_list])
                position_ids = torch.stack([F.pad(position_ids, (0, max_length - position_ids.shape[0]), value=0) for position_ids in position_ids_list])
                labels = torch.stack([F.pad(labels, (0, max_length - labels.shape[0]), value=-100) for labels in labels_list])
            elif self.padding_side == "left":
                input_ids = torch.stack([F.pad(input_ids, (max_length - input_ids.shape[0], 0), value=self.tokenizer.pad_token_id) for input_ids in input_ids_list])
                attention_mask = torch.stack([F.pad(attention_mask, (max_length - attention_mask.shape[0], 0), value=0) for attention_mask in attention_mask_list])
                loss_mask = torch.stack([F.pad(loss_mask, (max_length - loss_mask.shape[0], 0), value=0) for loss_mask in loss_mask_list])
                position_ids = torch.stack([F.pad(position_ids, (max_length - position_ids.shape[0], 0), value=0) for position_ids in position_ids_list])
                labels = torch.stack([F.pad(labels, (max_length - labels.shape[0], 0), value=-100) for labels in labels_list])
            else:
                raise ValueError(f"Invalid padding_side: {self.padding_side}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
    
@dataclass
class Map:
    tokenizer: PreTrainedTokenizer
    prompt_key: str = "problem"
    response_key: str = "solution"
    conversation_key: str = "conversations"
    role_key: str = "role"
    user_key: str = "user"
    assistant_key: str = "assistant"
    content_key: str = "content"
    apply_chat: bool = False
    def __call__(self, item):
        if self.conversation_key in item:
            conversations = item[self.conversation_key]
            for conversation in conversations:
                if conversation[self.role_key] == self.user_key:
                    prompt = conversation[self.content_key]
                if conversation[self.role_key] == self.assistant_key:
                    response = conversation[self.content_key]
        else:
            prompt = item[self.prompt_key]
            response = item[self.response_key]
        if self.apply_chat:
            prompt = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        else:
            prompt_str = f"Problem: {prompt}\nSolution: "
        response_str = response + self.tokenizer.eos_token
        prompt_ids_output = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        # 5. compute position_ids
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        position_ids = compute_position_id_with_mask(attention_mask)
        # 6. loss mask
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt loss.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0 #XXX:这里也要注意shift_labels
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0 #FIXME:不确定
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "labels": labels,
        }
if __name__ == "__main__":
    data_dir = "/share/home/sxjiang/myproject/self-learn/processed_new/merged_5_3_2"
    data_files=glob.glob(os.path.join(data_dir, "*.parquet"))
    load_method = "1"
    tokenizer = AutoTokenizer.from_pretrained("/share/home/sxjiang/model/Ouro-1.4B")
    tokenizer.pad_token = tokenizer.eos_token
    apply_chat = True
    t0 = time.time()
    if apply_chat:
        tokenizer.eos_token = "<|im_end|>"
    if load_method == "2":
        # openthought_dataset = load_dataset("parquet",data_files=data_files,streaming=True)["train"]
        openthought_dataset = openthought_dataset.map(Map(tokenizer=tokenizer, apply_chat=True))
    elif load_method == "1":
        openthought_dataset = OpenThoughtsDataset(data_files=data_files, tokenizer=tokenizer, apply_chat=True)
    pass
    omni_math_dataset = OmniMathDataset(data_source="/share/home/sxjiang/myproject/self-learn/datasets/Omni-MATH/rlp_test_set.jsonl", tokenizer=tokenizer, max_length=1024)
    accelerator = Accelerator()
    dataloader = DataLoader(
        openthought_dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, max_length=15000,truncation="right"),num_workers=0)
    # dataloader = DataLoader(
    #     openthought_dataset, batch_size=16, num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    t1 = time.time()
    print(f"Load Dataset Time taken: {t1 - t0} seconds")
    for idx, batch in enumerate(dataloader):
        t2 = time.time()
        print(f"Load Batch {idx} Time taken: {t2 - t1} seconds")
        print(len(batch))
        t1 = time.time()
        if idx > 10:
            break