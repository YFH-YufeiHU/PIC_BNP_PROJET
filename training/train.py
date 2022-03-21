import shutil
from transformers import LayoutLMForTokenClassification
import torch
from transformers import AdamW
from tqdm import tqdm
from data_loading.funsd import train_dataloader,eval_dataloader
from .evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

if os.path.exists('./tensorboard_log'):
    shutil.rmtree('./tensorboard_log')
else:
    os.mkdir('./tensorboard_log')
writer = SummaryWriter('./tensorboard_log')

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(model, device, train_dataloader, val_dataloader, optimizer,labels,num_train_epochs,version_v2=False):
    model.to(device)
    global_step = 0
    num_train_epochs = num_train_epochs
    t_total = len(train_dataloader) * num_train_epochs # total number of training steps

    #put the model in training mode
    f1 = 0
    model.train()
    for epoch in range(num_train_epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)
            if version_v2 == True:
              imgs = batch[5].to(device)

            # forward pass
            if version_v2:
              outputs = model(image = imgs,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          labels=labels)
            else:
              outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          labels=labels)
            loss = outputs.loss

            # print loss every 100 steps
            if global_step % 100 == 0:
                print(f"Train loss after {global_step} steps: {loss.item()}")

            # backward pass to get the gradients
            loss.backward()

            #print("Gradients on classification head:")
            #print(model.classifier.weight.grad[6,:].sum())

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        print("Evaluation for the epoch {}".format(epoch+1))
        results_val = evaluate(model,device, val_dataloader,labels)
        results_train = evaluate(model,device, train_dataloader,labels)
        model.train()
        writer.add_scalars('loss', {'train loss':results_train['loss'],'val loss':results_val['loss']}, epoch)
        writer.add_scalars('precision', {'train precision':results_train['precision'],'val precision':results_val['precision']}, epoch)
        writer.add_scalars('recall', {'train recall':results_train['recall'],'val recall':results_val['recall']}, epoch)
        writer.add_scalars('f1', {'train f1':results_train['f1'],'val f1':results_val['f1']}, epoch)
        if results_val['f1']>f1:
            print("Save the best model of epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), './checkpoint_LayoutLMF_best.pth')
            f1 = results_val['f1']
        print(results_val)


    torch.save(model.state_dict(), './checkpoint_LayoutLMF_final.pth')


    return global_step, loss / global_step
