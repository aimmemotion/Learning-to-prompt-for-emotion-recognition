import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import math
import clip
from PIL import Image

#for l2l
import learn2learn as l2l

class MAML_Dataset(Dataset):
    def __init__(self, preprocess, class_name_seen, dataset_base):
        super().__init__()
        self.preprocess = preprocess
        self.train_list_image = []
        self.train_list_label = []
        self.__getdata__(class_name_seen, dataset_base)
    
    def __getdata__(self, class_name_seen, dataset_base):
        for i, emotion in enumerate(class_name_seen):
            for pic in os.listdir(os.path.join(dataset_base, "train", emotion)):
                self.train_list_image.append(os.path.join(dataset_base, "train", emotion, pic))
                self.train_list_label.append(i)
        print(f'x-len: {len(self.train_list_image)}, y-len: {len(self.train_list_label)}')

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.train_list_image[idx]))
        label = self.train_list_label[idx]
        return image, label
    
    def __len__(self):
        return len(self.train_list_label)


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

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def main(args):
    #edit
    adapt_steps=args.adapt_steps
    lr=args.lr
    meta_lr=args.meta_lr
    WAYS = args.ways
    SHOTS = args.shots
    ITERATION = args.iteration

    task_name = args.task_name
    label_seen = args.label_seen
    class_name = args.class_name
    class_text = args.class_text
    class_name_seen = [class_name[i] for i in label_seen]

    #user define
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # If using GPU then use mixed precision training.
    clip_backbone = 'ViT-B/16' #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    _, preprocess = clip.load(clip_backbone, device = device, jit = False)
    model = CLIP_with_meta(clip_backbone, device)

    # use your own data
    dataset_base = args.dataset_base
    train_dataset = MAML_Dataset(preprocess, class_name_seen, dataset_base)

    #loss and optimizer
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    maml = l2l.algorithms.MAML(model, lr=lr, first_order=False)
    maml = maml.cuda()
    opt = Adam(maml.parameters(),lr=meta_lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
    #dataset
    dataset = l2l.data.MetaDataset(train_dataset)
    transforms = [
        l2l.data.transforms.FusedNWaysKShots(dataset, n=WAYS, k=SHOTS),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    taskset = l2l.data.TaskDataset(dataset, task_transforms=transforms, num_tasks=args.num_tasks)

    # Now training
    min_loss = 10.0
    os.makedirs("model",exist_ok=True)
    f = open("model/result_" + task_name + ".txt", 'w')

    for iteration in range(ITERATION):
        learner = maml.clone()  # Creates a clone of model
        adaptation_task = taskset.sample()

        # Fast adapt to support tasks
        for step in range(adapt_steps):
            class_text_list = [class_text[adaptation_task[1][SHOTS*i].item()] for i in range(WAYS)]
            for i in range(WAYS*SHOTS):
                adaptation_task[1][i] = math.floor(i/SHOTS)
            adaptation_predictions = learner(adaptation_task[0].cuda(), class_text_list)
            evaluation_error = loss_func(adaptation_predictions, adaptation_task[1].cuda())
            learner.adapt(evaluation_error, allow_unused=True)

        # Compute evaluation loss for that task
        evaluation_task = taskset.sample()
        class_text_list = [class_text[adaptation_task[1][SHOTS*i].item()] for i in range(WAYS)]
        for i in range(WAYS*SHOTS):
            evaluation_task[1][i] = math.floor(i/SHOTS)
        evaluation_predictions = learner(evaluation_task[0].cuda(), class_text_list)
        evaluation_error = loss_func(evaluation_predictions, evaluation_task[1].cuda())

        # Meta-update the model parameters
        opt.zero_grad() ##update maml not learner
        evaluation_error.backward()
        opt.step()

        # Validation
        with torch.no_grad():
            predictions = learner(evaluation_task[0].cuda(), class_text_list)
            valid_error = loss_func(predictions, evaluation_task[1].cuda())
            valid_accuracy = accuracy(predictions,evaluation_task[1].cuda())
            print(f"iter_{iteration+1}, val loss:{valid_error:.4f}, val acc:{valid_accuracy:.4f}")
            if valid_error < min_loss:
                min_loss = valid_error
                f.write("iter: ")
                f.write(str(iteration+1))
                f.write(" loss: ")
                f.write(str(round(valid_error.item(), 4)))
                f.write(" acc: ")
                f.write(str(round(valid_accuracy.item(), 4)))
                f.write("\n")
                torch.save(maml,"model/maml_" + task_name + ".pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", type=str, default="../Dataset/Emotion6", help="path to dataset")
    parser.add_argument("--task_name", type=str, default="Emotion642", help="output file name")
    parser.add_argument("--label_seen", nargs='+', type=int, default=[0, 1, 2, 3], help="emotion class name")
    parser.add_argument("--class_name", nargs='+', type=str, default=['anger',  'disgust',  'fear',  'joy', 'sadness', 'surprise'], help="emotion class name")
    parser.add_argument("--class_text", nargs='+', type=str, default=['anger',  'disgust',  'fear',  'joy', 'sadness', 'surprise'], help="emotion class text")

    parser.add_argument("--adapt_steps", type=int, default=2, help="")
    parser.add_argument("--lr", type=float, default=1e-7, help="learner learning rate")
    parser.add_argument("--meta_lr", type=float, default=1e-8, help="meta-learner learning rate")
    parser.add_argument("--ways", type=int, default=4, help="meta-learning ways")
    parser.add_argument("--shots", type=int, default=2, help="meta-learning shots")
    parser.add_argument("--iteration", type=int, default=10000, help="")
    parser.add_argument("--num_tasks", type=int, default=20000, help="")
    args = parser.parse_args()
    main(args)