import argparse
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMForTokenClassification
import torch
from tqdm import tqdm
import numpy as np
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


def metrics(y_true, y):
    """Calculate precision score, recall and f1 score.

        Args:
            y_true (list): true sequences.
            y (list): predicted sequences.

        Returns:
            results: precision score, recall and f1 score.
        """
    results = {
        "precision": precision_score(y_true, y),
        "recall": recall_score(y_true, y),
        "f1": f1_score(y_true, y)
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results
