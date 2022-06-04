import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class UDCInputExample:
    utterance: str
    label: float

    def to_dict(self):

        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


class UDCInputFeatures:

    def __init__(self, input_ids, attention_mask, token_type_ids, idx=None, to_tensor=False):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.idx = idx
        if to_tensor:
            self.input_ids = torch.tensor(self.input_ids)
            self.attention_mask = torch.tensor(self.attention_mask)
            self.token_type_ids = torch.tensor(self.token_type_ids)

    def cuda(self):
        self.input_ids = self.input_ids.cuda()
        self.token_type_ids = self.token_type_ids.cuda()
        self.attention_mask = self.attention_mask.cuda()
        return self


def convert_examples_to_features(
        examples: Union[List[str], List[List]],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        idx = (),
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if len(examples) == 0:
        return []

    if isinstance(examples[0], list):
        return [convert_examples_to_features(example, tokenizer, max_length, idx+(i,))
                            for i, example in enumerate(examples)]

    batch_encoding = tokenizer(
        examples,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = UDCInputFeatures(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"],
                                   attention_mask=inputs["attention_mask"], idx=idx+(i,), to_tensor=True)
        features.append(feature)
    return features


class UDCDataset(data.Dataset):
    def __init__(self, data: list, label_list: list, tokenizer: PreTrainedTokenizer, max_seq_length: int, use_cuda=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # transform:
        # utterances: List[str]
        # entities: List[List[str]]
        # entity_types: List[List[List[str]]]
        self.utterances, self.entities, self.entity_types, self.labels = \
            self.transform(self.data)
        self.labels = [label_list.index(label) for label in self.labels]
        # convert_features:
        # utterances: List[UDCInputFeatures]
        # entities: List[List[UDCInputFeatures]]
        # entity_types: List[List[List[UDCInputFeatures]]]
        self.utterances_feat = self.convert_features(self.utterances)
        self.entities_feat = self.convert_features(self.entities)
        self.entity_types_feat = self.convert_features(self.entity_types)
        self.cuda = use_cuda

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "utterance": self.utterances_feat[idx], # UDCInputFeatures
            "entity": self.entities_feat[idx],   # List[UDCInputFeatures]
            "entity_type": sum(self.entity_types_feat[idx], []), # List[UDCInputFeatures]
            "label": self.labels[idx],
        }
        return item

    def collate_fn(self, batchs):
        utterance_list = [b["utterance"] for b in batchs]
        u_input_ids = torch.stack([u.input_ids for u in utterance_list], dim=0)
        u_token_type_ids = torch.stack([u.token_type_ids for u in utterance_list], dim=0)
        u_attention_mask = torch.stack([u.attention_mask for u in utterance_list], dim=0)
        u_idx = [u.idx for u in utterance_list]
        batch_utterance = UDCInputFeatures(u_input_ids, u_attention_mask, u_token_type_ids, u_idx)

        entity_list = [b["entity"] for b in batchs]
        entity_list = sum(entity_list, [])
        e_input_ids = torch.stack([e.input_ids for e in entity_list], dim=0)
        e_token_type_ids = torch.stack([e.token_type_ids for e in entity_list], dim=0)
        e_attention_mask = torch.stack([e.attention_mask for e in entity_list], dim=0)
        e_idx = [e.idx for e in entity_list]
        batch_entity = UDCInputFeatures(e_input_ids, e_attention_mask, e_token_type_ids, e_idx)

        entity_type_list = [b["entity_type"] for b in batchs]
        entity_type_list = sum(entity_type_list, [])
        t_input_ids = torch.stack([t.input_ids for t in entity_type_list], dim=0)
        t_token_type_ids = torch.stack([t.token_type_ids for t in entity_type_list], dim=0)
        t_attention_mask = torch.stack([t.attention_mask for t in entity_type_list], dim=0)
        t_idx = [t.idx for t in entity_type_list]
        batch_entity_type = UDCInputFeatures(t_input_ids, t_attention_mask, t_token_type_ids, t_idx)

        batch_label = torch.tensor([b["label"] for b in batchs])

        if self.cuda:
            batch_utterance = batch_utterance.cuda()
            batch_entity = batch_entity.cuda()
            batch_entity_type = batch_entity_type.cuda()
            batch_label = batch_label.cuda()

        items = {
            "utterance": batch_utterance,       #idx: [4, 3, 5, 7]
            "entity": batch_entity,             #idx: [(4, 0), (4, 1), (5, 0), (7, 0), (7, 1)]
            "entity_type": batch_entity_type,   #idx: [(4, 0, 0), (4, 0, 1), ....]
            "label": batch_label,
        }
        return items

    def transform(self, data_list):
        utterances = []
        entities = []
        entity_types = []
        labels = []
        for data in data_list:
            utterances.append(data["utterance"])
            labels.append(data["label"])
            type_list = data['type_list']
            entity_list = []
            entity_type_list = []
            for entity, types in data["entity_types"].items():
                entity_list.append(entity)
                entity_type_list.append([type_list[type_id] for type_id in types])
            entities.append(entity_list)
            entity_types.append(entity_type_list)

        return utterances, entities, entity_types, labels

    def convert_features(self, examples):
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length,
        )

class UDCDataLoader(object):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None, use_cuda=False):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.use_cuda = use_cuda

    def get_dataloader(self, data: list, batch_size, labels, **kwargs):
        dataset = UDCDataset(data, labels, self.tokenizer, self.max_length, use_cuda=self.use_cuda)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, **kwargs)

