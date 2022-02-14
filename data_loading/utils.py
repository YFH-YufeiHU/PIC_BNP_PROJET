from sroie.py import dataloader_train_sroie, dataloader_test_sroie
from funsd.py import train_dataloader, eval_dataloader

def dataloader_train(Dataset):
  if Dataset=='sroie':
    return(dataloader_train_sroie)
  elif Dataset=='funsd':
    return(train_dataloader)
  
def dataloader_test(Dataset):
  if Dataset=='sroie':
    return(dataloader_test_sroie)
  elif Dataset=='funsd':
    return(eval_dataloader)
  

