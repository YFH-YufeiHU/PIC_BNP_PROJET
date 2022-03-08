from transformers import LayoutLMForTokenClassification
import torch
from transformers import AdamW
from tqdm import tqdm
from data_loading.funsd import train_dataloader,eval_dataloader
from .evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
import os

if os.path.exists('./tensorboard_log'):
    pass
else:
    os.mkdir('./tensorboard_log')
writer = SummaryWriter('./tensorboard_log')

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(model, device, train_dataloader, eval_dataloader,optimizer,labels,num_train_epochs):
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

            # forward pass
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
            results = evaluate(model,device,eval_dataloader)
            model.train()
            writer.add_scalar('loss', results['loss'], epoch)
            writer.add_scalar('precision', results['precision'], epoch)
            writer.add_scalar('recall', results['recall'], epoch)
            writer.add_scalar('f1', results['f1'], epoch)
            if results['f1']>f1:
                torch.save(model.state_dict(), './checkpoint_LayoutLMF_{}_best.pth'.format(epoch))
                f1 = results['f1']
            print(results)



    torch.save(model.state_dict(), './checkpoint_LayoutLMF_final.pth')


    return global_step, loss / global_step
