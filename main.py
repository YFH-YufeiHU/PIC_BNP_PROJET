from pickle import FALSE
from transformers import LayoutLMForTokenClassification
from transformers import LayoutLMv2ForTokenClassification
import torch
from transformers import AdamW
from tqdm import tqdm
import os
import numpy as np
import random
from training.train import train
from training.evaluate import evaluate
import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    # Datset Options
    parser.add_argument("--data_root", type=str, default='./data_loading/FUNSD',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='funsd',
                        choices=['funsd', 'sroie'], help='Name of dataset')
    # Models Options
    parser.add_argument("--model", type=str, default='LayoutLM',
                        choices=['LayoutLM', 'LayoutLMv2'], help='model name')
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save results to \"./results\"")
    parser.add_argument("--num_train_epochs", type=int, default=5,
                        help="epoch number (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate (default: 0.00005)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    return parser
def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    # define number of labels refer to different dataset
    if opts.dataset.lower() == 'funsd':
        from data_loading.funsd import train_dataloader, eval_dataloader,val_dataloader
        def get_labels(path):
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        labels = get_labels(os.path.join(opts.data_root,"labels.txt"))
        num_labels = len(labels)
    else:
        from data_loading.funsd_sroie import train_dataloader, eval_dataloader,val_dataloader
        def get_labels(path):
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        labels = get_labels(os.path.join(opts.data_root, "labels.txt"))
        num_labels = len(labels)
        print(num_labels)
    # define model
    if opts.model == 'LayoutLM':
        model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
        version_v2 = False
    else:
        model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased", num_labels=num_labels)
        version_v2 = True
    model.to(device)
    # define optimization
    optimizer = AdamW(model.parameters(), lr=opts.lr)
    if opts.test_only:
        model.load_state_dict(torch.load('./checkpoint_LayoutLMF_best.pth'))
        results = evaluate(model=model, device=device, eval_dataloader=eval_dataloader,labels=labels,save_result=True,version_v2=version_v2)
        print(results)
        return
    else:
        globel_step, loss = train(model=model, device=device, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, labels=labels,num_train_epochs = opts.num_train_epochs,version_v2=version_v2)
        print('the globel step is {} and the loss is {}'.format(globel_step,loss))
        return
if __name__ == '__main__':
    main()
