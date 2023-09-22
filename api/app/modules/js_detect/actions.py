import logging
import os
from enum import Enum
from typing import Sequence

import numpy as np
import torch
from src.codebert_bimodal.model import Model
from src.codebert_bimodal.utils import convert_examples_to_features
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

logger = logging.getLogger("js_detection")


class ModelConfig:
    n_gpu = torch.cuda.device_count()
    per_gpu_eval_batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "models"
    model_path = "models/pytorch_model.bin"
    local_rank = -1
    max_seq_length = 200


class TextDescription(str, Enum):
    MALICIOUS = "javascript perform malicious actions to trick users, steal data from users, \
        or otherwise cause harm."
    BENIGN = "javascript perform normal, non-harmful actions"


model_config = ModelConfig()


class JavaScriptDataset(Dataset):
    def __init__(self, tokenizer, args, data, type=None):
        self.examples = []
        self.type = type
        for js in data:
            if self.type == "test":
                js["label"] = 0
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

        # for idx, example in enumerate(self.examples[:3]):
        #     logger.debug("*** Example ***")
        #     logger.debug("idx: {}".format(idx))
        #     logger.debug("code_tokens: {}".format([x.replace("\u0120", "_") for x in example.code_tokens]))
        #     logger.debug("code_ids: {}".format(" ".join(map(str, example.code_ids))))
        #     logger.debug("nl_tokens: {}".format([x.replace("\u0120", "_") for x in example.nl_tokens]))
        #     logger.debug("nl_ids: {}".format(" ".join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """return both tokenized code ids and nl ids and label"""
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].nl_ids),
            torch.tensor(self.examples[i].label),
        )


class JavaScriptClassifier:
    def __init__(self):
        self.device = model_config.device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_config.model_dir)
        self.args = model_config
        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = 2
        model = RobertaModel.from_pretrained(
            "microsoft/codebert-base",
            from_tf=False,
            config=config,
        )
        self.model = Model(model, config, self.tokenizer, model_config)
        self.model.load_state_dict(torch.load(model_config.model_path, map_location=self.device))
        self.model.to(self.args.device)

    def predict(self, input_data):
        eval_dataset = JavaScriptDataset(self.tokenizer, self.args, input_data, "test")

        eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset) if self.args.local_rank == -1 else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        # if self.args.n_gpu > 1:
        #     model = torch.nn.DataParallel(self.model)
        model = self.model
        # model.eval()
        # Eval!
        logger.info("***** Running Test *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        nb_eval_steps = 0
        all_predictions = []
        for batch in eval_dataloader:
            code_inputs = batch[0].to(self.args.device)
            nl_inputs = batch[1].to(self.args.device)
            labels = batch[2].to(self.args.device)
            with torch.no_grad():
                _, predictions = model(code_inputs, nl_inputs, labels)
                all_predictions.append(predictions.cpu())
            nb_eval_steps += 1
        all_predictions = torch.cat(all_predictions, 0).squeeze().numpy()

        # if isinstance(all_predictions, np.array):
        #     all_predictions = np.array(all_predictions)
        logger.debug(all_predictions)
        results = []

        for example, pred in zip(input_data, all_predictions.tolist()):
            if example["doc"] == TextDescription.MALICIOUS.value and pred == 1:
                results.append({"idx": example["idx"], "label": "malicious"})
            elif example["doc"] == TextDescription.MALICIOUS.value and pred == 0:
                results.append({"idx": example["idx"], "label": "benign"})
            elif example["doc"] == TextDescription.BENIGN.value and pred == 1:
                results.append({"idx": example["idx"], "label": "benign"})
            elif example["doc"] == TextDescription.BENIGN.value and pred == 0:
                results.append({"idx": example["idx"], "label": "malicious"})

        return results


cls = JavaScriptClassifier()


def get_cls():
    return cls
