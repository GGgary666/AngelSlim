# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

import torch

__all__ = [
    "process_token_dict_to_mappings",
    "convert_sharegpt_data",
    "convert_ultrachat_data",
    "DataCollatorWithPadding",
]


def convert_sharegpt_data(row, dataset_column="conversations"):
    converted_messages = []

    role_mapping = {"human": "user", "gpt": "assistant"}
    messages = row[dataset_column]
    for message in messages:
        converted_messages.append(
            {"role": role_mapping[message["from"]], "content": message["value"]}
        )

    return {"conversations": converted_messages, "id": row["id"]}


def convert_ultrachat_data(row, dataset_column="messages"):
    converted_messages = []

    messages = row[dataset_column]
    for message in messages:
        converted_messages.append(
            {"role": message["role"], "content": message["content"]}
        )
    return {"conversations": converted_messages, "id": row["prompt_id"]}


# Copied from https://github.com/sgl-project/SpecForge/blob/main/specforge/data/preprocessing.py # noqa: E501
def process_token_dict_to_mappings(
    token_dict,
    draft_vocab_size: int,
    target_vocab_size: int,
):
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.

    Args:
        token_dict: A Counter object mapping token ids to their frequencies.
        draft_vocab_size: The size of the draft vocabulary.
        target_vocab_size: The size of the target vocabulary.

    Returns:
        A tuple containing:
            - d2t: A tensor mapping draft token ids to target token ids.
            - t2d: A tensor mapping target token ids to draft token ids.
    """
    if len(token_dict) < draft_vocab_size:
        existing_tokens = set(token_dict.keys())
        missing_tokens = set(range(draft_vocab_size)) - existing_tokens
        for token in missing_tokens:
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    print(f"Added missing tokens to reach draft vocab size: {draft_vocab_size}")
    print(f"Total tokens after addition: {len(token_dict)}")
    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)

    if total_frequency == 0:
        print(
            "Warning: Total token frequency is zero. All tokens will have zero ratio."
        )
        top_N_ratio = 0.0
    else:
        top_N_ratio = top_N_frequency_sum / total_frequency

    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()

    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    d2t = torch.tensor(d2t)
    t2d = torch.tensor(t2d)

    return d2t, t2d


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch
