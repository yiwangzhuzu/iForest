import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import shutil
import os.path as osp
from tqdm import tqdm
import torch.nn as nn
import timm
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name='resnet50', out_features=6):  #类别总数#resnet50     vit_tiny_patch16_224    vgg16
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)  # 从预训练的库中加载模型
        # self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="pretrained/resnet50d_ra2-464e36ba.pth")  # 从预训练的库中加载模型
        # classifier
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features  # 修改全连接层数目
            self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features  # 修改全连接层数目
            self.model.head = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vgg":
            n_features = self.model.head.fc.in_features  # 修改全连接层数目
            self.model.head.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)
        # resnet修改最后的全链接层
        print(self.model)  # 返回模型

    def forward(self, x):  # 前向传播
        x = self.model(x)
        return x

model_path = r'D:\2022\TongZhou\Test\tSNE\resnet50_36epochs_accuracy0.96909_weights.pth'

# 加载预训练的VIT模型
# model = SELFMODEL(model_name='resnet50d', out_features=31)
model = SELFMODEL(model_name='resnet50', out_features=6)
weights = torch.load(model_path, map_location='cuda:0')
# weights = torch.load(model_path)
model.load_state_dict(weights)
model.cuda()
model.eval()



#耕地\工矿用地\交通运输用地\林地\水域用地\住宅用地

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.image_filenames = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                                filename.endswith('.jpg') or filename.endswith('.png')or filename.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filepath = self.image_filenames[idx]
        with open(filepath, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img).cuda()
        return img_tensor




x = r'D:\2022\TongZhou\Test\T1样本回归'
folder_path_list = os.listdir(x)

z1 = r'D:\2022\TongZhou\Test\T1样本回归1' + '/resnet50_iso'
if osp.isdir(z1):
    shutil.rmtree(z1)
os.mkdir(z1)

for folder_path in folder_path_list:

    y = x + '/' + folder_path
    # 定义数据集和数据加载器
    dataset = CustomDataset(y, transform)
    dataloader = DataLoader(dataset, batch_size=64)

    # 使用VIT模型提取图像特征
    features_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            batch_features = model(batch).cpu().numpy()
            features_list.append(batch_features)
    features = np.concatenate(features_list)

    a = os.listdir(y)

    # 使用 IsolationForest 进行去噪，其中 n_estimators 表示使用多少个随机树进行去噪
    clf = IsolationForest(n_estimators=100, contamination=0.05)
    clf.fit(features)
    labels = clf.predict(features)

    # 将噪声样本和正常样本分别存储到两个列表中
    normal_samples = []
    noise_samples = []
    for i, label in enumerate(labels):
        if label == 1:
            normal_samples.append(os.path.join(folder_path, a[i]))
        else:
            noise_samples.append(os.path.join(folder_path, a[i]))


    z2 = z1 +'/'+ folder_path

    os.mkdir(z2)

    # 将正常样本和噪声样本分别存储到对应的文件夹中
    normal_folder = z2 + "/normal"
    if osp.isdir(normal_folder):
        print("normal folder 已存在， 正在删除...")
        shutil.rmtree(normal_folder)
    os.mkdir(normal_folder)
    for img_path in normal_samples:
        z3 =x+ '/' +img_path
        shutil.copy2(z3 , normal_folder)

    noise_folder = z2 + "/noise"
    if osp.isdir(noise_folder):
        print("noise folder 已存在， 正在删除...")
        shutil.rmtree(noise_folder)
    os.mkdir(noise_folder)
    for img_path in noise_samples:
        z4 = x + '/' + img_path
        shutil.copy2(z4 , noise_folder)
