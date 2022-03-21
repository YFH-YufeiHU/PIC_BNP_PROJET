from PIL import Image, ImageDraw, ImageFont
import json
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMTokenizer,LayoutLMv2Tokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import logging
import os
import random
import torch
from data_loading.read_txt_utils import convert_examples_to_features, read_examples_from_file
from torch.utils.data import random_split
from torchvision import transforms

# if there are questions about path, change line 102 and 113.

logger = logging.getLogger(__name__)

# In fact, the codes of FunsdDataset is also included in the library layoutlm.data.funsd, we can use these functions directly by importing this library.
# Here we define the FunsdDataset by ourself and we create the torch type dataset for funsd which can be directly called in main.py to train.
def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class FunsdDataset(Dataset):
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        if args.local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            features = convert_examples_to_features(

                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                logger=logger,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                pad_token_label_id=pad_token_label_id,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

             
        if args.local_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)
        self.images = []
        transform_ = transforms.Compose([transforms.ToTensor(),])
        for path in features:
          # print(path.file_name)
          if mode == "train":
            path = os.path.join('/content/PIC_BNP_PROJET/data_loading/FUNSD/training_data/images',path.file_name)
          else:
            path = os.path.join('/content/PIC_BNP_PROJET/data_loading/FUNSD/testing_data/images',path.file_name)
          img = Image.open(path).convert("RGB")
          img = transform_(img)
          self.images.append(img)
        # exit()
        # print(len(self.images))
      
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
            # for layout lm v2
            self.images[index],
        )


# dataset uses the BIOES annotation scheme,
# a given token is at the beginning (B), inside (I), outside (O), at the end (E) or start (S) of a given entity.
# Entities include ANSWER, QUESTION, HEADER and OTHER
def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

labels = get_labels("./data_loading/FUNSD/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}

# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index
# print(labels)

# Create a PyTorch dataset and corresponding dataloader
args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': './data_loading/FUNSD',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict(args)
# for layoutlm v1
# tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
# for layoutlm v2
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

train_dataset_all = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
train_size = round(train_dataset_all.__len__()*0.75)
val_size = train_dataset_all.__len__() - train_size

train_val_test_split = [train_size, val_size]
train_dataset, val_dataset = random_split(
                train_dataset_all, train_val_test_split)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=2)

val_sampler = RandomSampler(val_dataset)
val_dataloader = DataLoader(val_dataset,
                              sampler=val_sampler,
                              batch_size=2)

eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset,
                              sampler=eval_sampler,
                            batch_size=2)

# print(len(train_dataloader))
# print(len(eval_dataloader))
# print(eval_dataloader)
print(val_dataloader)
