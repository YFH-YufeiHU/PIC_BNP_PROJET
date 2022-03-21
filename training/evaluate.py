import numpy as np
from transformers import LayoutLMForTokenClassification
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from data_loading.funsd import eval_dataloader,pad_token_label_id,label_map
#from data_loading.funsd_sroie import eval_dataloader,pad_token_label_id,label_map

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def evaluate(model, device, eval_dataloader,labels,save_result=False,version_v2=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.to(device)
    # put model in evaluation mode
    model.eval()
    # print(len(eval_dataloader))
    # exit()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
    # for batch in eval_dataloader:
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)
            if version_v2 == True:
              imgs = batch[5].to(device)

            # forward pass
            if version_v2:
              outputs = model(image=imgs,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              labels=labels)
            else:
              outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              labels=labels)
            if save_result:
                pass
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    # label_map = {i: label for i, label in enumerate(labels)}
    pad_token_label_id = CrossEntropyLoss().ignore_index
    # print(pad_token_label_id)
    # print(label_map)
    # exit()

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                # print(label_map[out_label_ids[i][j]])
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    if save_result:
        with open("./predictions.txt", "w") as f:
            with open("./data_loading/FUNSD/test.txt", "r") as f1:
            # with open("./data_loading/SROIE/test.txt", "r") as f1:
                example_id = 0
                for line in f1:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        print(line)
                        f.write(line)
                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        output_line = (line.split()[0] + " " + preds_list[example_id].pop(0) + "\n")
                        f.write(output_line)
    # print(results)
    return results
