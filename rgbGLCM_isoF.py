
import shutil
import os.path as osp
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import os
from sklearn.decomposition import PCA
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import cv2
from sklearn.preprocessing import MinMaxScaler

distances = [1]  # 距离
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 方向角度
levels = 256  # 灰度级别数

def rgb_features(image):
    # 打开图片并转换为RGB模式


    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 提取RGB三个波段
    red_band = image_array[:, :, 0]
    green_band = image_array[:, :, 1]
    blue_band = image_array[:, :, 2]

    # 计算每个波段的最大值、最小值和中位数
    red_max = np.max(red_band)
    red_min = np.min(red_band)
    red_median = np.median(red_band)

    green_max = np.max(green_band)
    green_min = np.min(green_band)
    green_median = np.median(green_band)

    blue_max = np.max(blue_band)
    blue_min = np.min(blue_band)
    blue_median = np.median(blue_band)

    # 组合九个特征成一个序列
    features = [red_max, red_min, red_median, green_max, green_min, green_median, blue_max, blue_min, blue_median]

    return features


def calculate_glcm_features(image_gray, distances, angles, levels):

    # 计算灰度共生矩阵
    glcm = greycomatrix(image_gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # 计算灰度共生矩阵的统计特征
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    features = np.concatenate([contrast, dissimilarity])
    # features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    features = features.reshape(-1)
    return features

def hdgsrgb(image):

    image_cv1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    rgb_value = rgb_features(image_cv1)
    hdgs_value = calculate_glcm_features(image_cv2, distances, angles, levels)

    rgb_value.extend(hdgs_value)

    features = rgb_value
    # 将哈希值添加到列表

    return features


# 定义图像文件夹路径和文件后缀
suffix = ['.jpg', '.png', '.tif']
folders_path = r'D:\2022\TongZhou\Test\T1样本回归'
save_path = r'D:\2022\TongZhou\Test\T1样本回归去噪\孤立森林去噪'
folder_path_list = os.listdir(folders_path)


if osp.isdir(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)

for folder_path in folder_path_list:
    x=folder_path
    folder_path = folders_path + '/' + folder_path

    # 遍历文件夹下所有的图片文件，并使用phash函数计算图片的哈希值
    hash_list = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if any(f.endswith(suffix) for suffix in suffix):
                img_path = osp.join(root, f)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                hash_value = hdgsrgb(img)
                hash_list.append(hash_value)

    pca = PCA(n_components=8)  # 假设选择个主成分

    hash_list = pca.fit_transform(hash_list)

    scaler = MinMaxScaler()
    hash_list = scaler.fit_transform(hash_list)

    a = os.listdir(folder_path)

    # 使用 IsolationForest 进行去噪，其中 n_estimators 表示使用多少个随机树进行去噪
    clf = IsolationForest(n_estimators=100, contamination=0.1)
    clf.fit(hash_list)
    labels = clf.predict(hash_list)

    # 将噪声样本和正常样本分别存储到两个列表中
    normal_samples = []
    noise_samples = []
    for i, label in enumerate(labels):
        if label == 1:
            normal_samples.append(os.path.join(x, a[i]))
        else:
            noise_samples.append(os.path.join(x, a[i]))


    z2 = save_path +'/'+ x

    os.mkdir(z2)

    # 将正常样本和噪声样本分别存储到对应的文件夹中
    normal_folder = z2 + "/normal"
    if osp.isdir(normal_folder):
        print("normal folder 已存在， 正在删除...")
        shutil.rmtree(normal_folder)
    os.mkdir(normal_folder)
    for img_path in normal_samples:
        z3 =folders_path+ '/' +img_path
        shutil.copy2(z3 , normal_folder)

    noise_folder = z2 + "/noise"
    if osp.isdir(noise_folder):
        print("noise folder 已存在， 正在删除...")
        shutil.rmtree(noise_folder)
    os.mkdir(noise_folder)
    for img_path in noise_samples:
        z4 = folders_path + '/' + img_path
        shutil.copy2(z4 , noise_folder)

