import os
import scipy.io
import scipy.ndimage
import numpy as np
from random import shuffle
from sklearn.decomposition import PCA
from collections import Counter
from HSIdataset import DatasetInfo
import random

# 数据归一化
def ZNorm(data):
    h, w, bands = data.shape
    Ndata = data.reshape((h * w, bands))
    mu = np.mean(Ndata, axis=0)
    sigma = np.std(Ndata, axis=0)
    Ndata = (Ndata - mu)
    Ndata = Ndata / sigma
    data = Ndata.reshape((h, w, bands))
    return data


# 按数据量
def data_generator(datasetName, FolderPath, sampleperclass=0, rate=0, pca= None, del_list=[], seed=468):
    # 拼接路径
    info = DatasetInfo.info[datasetName]
    CLASSNUM = info['CLASSES_NUM']
    BAND = info['BAND']
    DATA_FILE_PATH = os.path.join(FolderPath, DatasetInfo.info[datasetName]['data_name'])
    LABEL_FILE_PATH = os.path.join(FolderPath, DatasetInfo.info[datasetName]['label_name'])

    # 载入数据
    label = scipy.io.loadmat(LABEL_FILE_PATH)[info['label_key']]
    data = scipy.io.loadmat(DATA_FILE_PATH)[info['data_key']]
    print("shape", data.shape)

    # 数据降维
    if pca:
        newX = np.reshape(data, (-1, data.shape[2]))
        transformer = PCA(n_components=pca, whiten=True)
        newX = transformer.fit_transform(newX)
        data = np.reshape(newX, ( data.shape[0], data.shape[1], pca))

    # index 记录重组
    INDEX_DICT = {"label_list": []}
    class_dict = {c: [] for c in np.unique(label).tolist()}  # 类别计数器

    print(datasetName + " INDEX_DICT:", INDEX_DICT)
    print(datasetName + " class_list:", class_dict)

    # 遍历 label 数组 取出各像素值的标签
    for h in range(label.shape[0]):
        for w in range(label.shape[1]):
            class_dict[label[h, w]].append((label[h, w],h,w))
    
    # 删除无用部分
    for i in del_list:
        del(class_dict[i])

    # 打乱
    random.seed(seed)
    for k,v in class_dict.items():
        random.shuffle(v)

    # 计数
    counter = {key: len(value) for key,value in class_dict.items()}  # 分类
            
                
    # random.shuffle(INDEX_DICT["label_list"])

    print("counter:", counter)

    
    # 分为test + train
    train_label, test_label = [],[]
  
        

    if sampleperclass:
        for key, value in class_dict.items():
            # print("partition:", key)
            train_label.extend(value[:sampleperclass])
            test_label.extend(value[sampleperclass:])
    elif rate:
        for key, value in class_dict.items():
            # print("partition:", key)
            RANGE = round(len(value) * rate)
            train_label.extend(value[:RANGE])
            test_label.extend(value[RANGE:])
    else:
        raise Exception("division error")
        
    return data, train_label, test_label, CLASSNUM, BAND