import argparse
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
#from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import os
import shutil

shutil.rmtree("confusion_matrix")
os.makedirs("confusion_matrix", exist_ok=True)

import clip
from tqdm import tqdm
from random import sample

#for confusion matrix
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

#user define
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # If using GPU then use mixed precision training.
clip_backbone = 'ViT-B/16' #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

def draw_confusion_matrix(args, all_preds, all_label, epoch, Val_acc):
    label_name = args.class_name
    num_class = len(label_name)
    all_preds = all_preds.tolist()
    all_label = all_label.tolist()
    cf_matrix = confusion_matrix(all_label, all_preds)
    cm = np.zeros((num_class, num_class),dtype=np.float64)
    for i in range(num_class):
        sum = 0
        for num in cf_matrix[i]:
            sum += num
        for j in range(num_class):
            cm[i][j] = float(float(cf_matrix[i][j])/float(sum))
        cm[i] = np.around(cm[i], decimals=3)
    df_cm = pd.DataFrame(cm, label_name, label_name)
    plt.figure(figsize = (18,14))
    sns.heatmap(df_cm, annot=True, fmt=".3", cmap='BuGn')
    plt.xlabel("Prediction")
    plt.ylabel("Label")
    plt.savefig("./confusion_matrix/" + "acc_" + str(round(Val_acc, 2)) + "_" + str(epoch) + "_confusion_matrix.png")
    
    num_pred = dict((i, all_preds.count(i)) for i in range(6))

class image_title_dataset():
    def __init__(self, preprocess, list_image_path, list_label):
        self.preprocess = preprocess
        self.image_path = list_image_path
        self.label = torch.tensor(list_label,dtype=torch.long)

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]))
        label = self.label[idx]
        return image, label

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()

class CLIP_with_meta(nn.Module):
    def __init__(self, clip_backbone, device):
        super(CLIP_with_meta, self).__init__()
        self.device = device
        #CLIP
        model, _ = clip.load(clip_backbone, device = device, jit = False)
        self.model_CLIP = model
        if device == "cpu":
            self.model_CLIP.float()
        else :
            clip.model.convert_weights(self.model_CLIP)

        #GPT-2
        self.model_GPT = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.tokenizer_GPT = GPT2Tokenizer.from_pretrained("gpt2")

    def forward(self, img, list_text):
        input_text = "A picture seems to express some feelings like "
        input_ids = self.tokenizer_GPT.encode(input_text, return_tensors="pt").to(self.device)
        output = self.model_GPT.generate(input_ids=input_ids, max_length=30, do_sample=True, temperature=0.7, top_p=0.9, top_k=0, num_return_sequences=5)
        generated_prompt = self.tokenizer_GPT.decode(output[0], skip_special_tokens=True)
        if len(generated_prompt.split()) > 60:
            new_prompt = ""
            for i in range(60):
                new_prompt += generated_prompt.split()[i]
                new_prompt += " "
            generated_prompt = new_prompt
        class_tokenized_seen = clip.tokenize([generated_prompt + ' ' + i + '.' for i in list_text]).to(self.device)
        logits, _ = self.model_CLIP(img, class_tokenized_seen)
        probs = logits.softmax(dim=-1)

        return probs

def train_epoch(args, model, train_dataloader, optimizer, loss_img, epoch, epoch_size_train):
    model.train()

    print('Start Training')
    train_loss = 0
    train_correct = 0
    train_num = 0
    Train_loss = 0
    Train_acc = 0

    with tqdm(total = epoch_size_train, desc=f'Epoch {epoch + 1}/{args.EPOCH}',postfix=dict,mininterval=0.3) as pbar:
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            probs = model(inputs, args.class_text)
            _, preds_image = torch.max(probs, 1)

            correct = torch.sum(preds_image == labels)
            train_correct += correct.item()
            train_num += labels.shape[0]

            loss = loss_img(probs, labels)
            loss.backward()
            train_loss += loss.item()

            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model.model_CLIP)
                optimizer.step()
                clip.model.convert_weights(model.model_CLIP)
            
            Train_acc = (train_correct / train_num) * 100
            Train_loss = train_loss / train_num
            pbar.set_postfix(**{'train_loss': Train_loss, 
                                'train_acc' : Train_acc, 
                                'lr'        : optimizer.param_groups[0]['lr']})
            pbar.update(1)

def val_epoch(args, model, val_dataloader, optimizer, loss_img, epoch, epoch_size_val):
    model.eval()

    print('Start Validation')
    val_loss = 0
    val_correct = 0
    val_num = 0
    Val_loss = 0
    Val_acc = 0

    all_preds, all_label = [], []

    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{args.EPOCH}',postfix=dict,mininterval=0.3) as pbar:
        for i, batch in enumerate(val_dataloader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            val_num += labels.shape[0]
            
            probs = model(inputs, args.class_text)
            _, preds_image = torch.max(probs, 1)

            #for confusion matrix
            if len(all_preds) == 0:
                all_preds.append(preds_image.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds_image.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], labels.detach().cpu().numpy(), axis=0)

            correct = torch.sum(preds_image == labels)
            val_correct += correct.item()
            
            loss = loss_img(probs, labels)
            val_loss += loss.item()
            
            Val_acc = (val_correct / val_num) * 100
            Val_loss = val_loss / val_num
            
            pbar.set_postfix(**{'val_loss': Val_loss, 
                                'val_acc' : Val_acc,
                                'lr'      : optimizer.param_groups[0]['lr']})
            pbar.update(1)
    
    all_preds, all_label = all_preds[0], all_label[0]
    
    return all_preds, all_label, Val_loss, Val_acc

def main(args):
    #user define
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # If using GPU then use mixed precision training.
    clip_backbone = 'ViT-B/16' #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    _, preprocess = clip.load(clip_backbone, device = device, jit = False)
    model = CLIP_with_meta(clip_backbone, device)
    model = torch.load("model/maml_" + args.task_name + ".pth")

    # use your own data
    train_list_image_path = [] 
    train_list_label = []

    for i, emotion in enumerate(args.class_name):
        pic_list = sample(os.listdir(os.path.join(args.dataset_base, "train", emotion)), args.SHOT)
        for pic in pic_list:
            train_list_image_path.append(os.path.join(args.dataset_base, "train", emotion, pic))
            train_list_label.append(i)
    train_dataset = image_title_dataset(preprocess, train_list_image_path, train_list_label)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size = args.batch_size) #Define your own dataloader
    epoch_size_train = len(train_list_image_path) // args.batch_size

    val_list_image_path = [] 
    val_list_label = []

    for i, emotion in enumerate(args.class_name):
        for pic in os.listdir(os.path.join(args.dataset_base, "val", emotion)):
            val_list_image_path.append(os.path.join(args.dataset_base, "val", emotion, pic))
            val_list_label.append(i)
    val_dataset = image_title_dataset(preprocess, val_list_image_path, val_list_label)
    val_dataloader = DataLoader(val_dataset,shuffle=True,batch_size = args.batch_size) #Define your own dataloader
    epoch_size_val = len(val_list_image_path) // args.batch_size

    #loss and optimizer
    loss_img = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    # Now training
    best_val_acc = 0

    for epoch in range(args.EPOCH):
        if args.TRAIN:
            train_epoch(args, model, train_dataloader, optimizer, loss_img, epoch, epoch_size_train)
        all_preds, all_label, val_loss, val_acc = val_epoch(args, model, val_dataloader, optimizer, loss_img, epoch, epoch_size_val)
        print("Result of", epoch, "/", args.EPOCH, ": ")
        print("SHOT:", args.SHOT)
        print("TRAIN:", args.TRAIN)
        print("val_loss:", val_loss)
        print("val_acc:", val_acc)
        print()
        if val_acc > best_val_acc:
            print("best_update!!!")
            best_val_loss = val_loss
            best_val_acc = val_acc
            draw_confusion_matrix(args, all_preds, all_label, epoch, val_acc)
        print("Best acc:")
        print("Best_val_loss:", best_val_loss)
        print("Best_val_acc:", str(round(best_val_acc, 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", type=str, default="../Dataset/Emotion6", help="path to dataset")
    parser.add_argument("--task_name", type=str, default="Emotion642", help="output file name")
    parser.add_argument("--class_name", nargs='+', type=str, default=['sadness', 'surprise'], help="emotion class name")
    parser.add_argument("--class_text", nargs='+', type=str, default=['sadness', 'surprise'], help="emotion class text")

    parser.add_argument("--lr", type=float, default=1e-7, help="learner learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--EPOCH", type=int, default=5000, help="")
    
    parser.add_argument("--TRAIN", type=bool, default=True, help="")
    parser.add_argument("--SHOT", type=int, default=1, help="")
    args = parser.parse_args()
    main(args)