import os
import torch
import sys

from models.semantic_matchers.colbert.utils.utils import torch_load_dnn

from transformers import AutoTokenizer
from models.semantic_matchers.colbert.modeling.hf_colbert import class_factory
from models.semantic_matchers.colbert.infra.config import ColBERTConfig


class BaseColBERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """

    def __init__(self, name_or_path, colbert_config=None):
        super().__init__()

        self.colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(name_or_path), colbert_config)
        self.name = self.colbert_config.model_name or name_or_path

        try:
            HF_ColBERT = class_factory(self.name)
        except:
            self.name = 'bert-base-uncased' # TODO: Double check that this is appropriate here in all cases
            HF_ColBERT = class_factory(self.name)

        # assert self.name is not None
        # HF_ColBERT = class_factory(self.name)

        self.model = HF_ColBERT.from_pretrained(name_or_path, colbert_config=self.colbert_config)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(name_or_path)

        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.LM

    @property
    def linear(self):
        return self.model.linear

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        assert not path.endswith('.dnn'), f"{path}: We reserve *.dnn names for the deprecated checkpoint format."

        self.model.save_pretrained(path)
        self.raw_tokenizer.save_pretrained(path)

        self.colbert_config.save_for_checkpoint(path)


