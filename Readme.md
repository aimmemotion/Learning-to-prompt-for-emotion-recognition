## This is the official code for the paper "Learning to Prompt for Vision-Language Emotion Recognition" 

### Environment installation 

```
Create a conda virtual environment and activate it:
conda create --name myenv python=3.8.16
conda activate myenv

Install PyTorch>=1.10.1 and torchvision>=0.11.2 with CUDA>=11.3:
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

Install required package:
pip install timm==0.4.12
pip install learn2learn
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install learn2learn
pip install matplotlib
pip install pandas
pip install seaborn
pip install -U scikit-learn
pip install transformers==4.15.0
```
### Dataset Preparation
```
For standard folder dataset, move images to labeled sub-folders. The file structure should look like:
Emotion6
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```
### Training
```
python meta_training.py --dataset_base ../Dataset/Emotion6/ \
--task_name Emotion642 \
--label_seen 0 1 2 3 \
--class_name anger disgust fear joy sadness surprise \
--class_text anger disgust fear joy sadness surprise
```

### Testing
```
python meta_testing.py --dataset_base ../Dataset/Emotion6/ \
--task_name Emotion642 \
--class_name sadness surprise \
--class_text sadness surprise \
--EPOCH 10000 \
--SHOT 1
```
