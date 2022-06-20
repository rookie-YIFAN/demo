import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
from einops.layers.torch import Rearrange
from einops import repeat


class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt',
        'data_name': 'PaviaU.mat',
        'label_name': 'PaviaU_gt.mat',
        'CLASSES_NUM': 9,
        'BAND': 103
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt',
            'data_name': 'salinas_corrected.mat',
            'label_name': 'salinas_gt.mat',
            'permute': [2, 0, 1],
            'CLASSES_NUM': 16,
            'BAND': 204
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'data_name': 'KSC.mat',
            'label_name': 'KSC_gt.mat',
            'BAND': 224,
            'CLASSES_NUM': 13,
        }, 'Houston': {
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
        }, 'Indian': {
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt',
            'data_name': 'indian_pines_corrected.mat',
            'label_name': 'indian_pines_gt.mat',
            'CLASSES_NUM': 16,
            'BAND':200
        }, 'Pavia': {
            'data_key': 'pavia',
            'label_key': 'pavia_gt',
            'data_name': 'Pavia.mat',
            'label_name': 'Pavia_gt.mat'
        }, 'Houston2013': {
            'data_key': 'Houston',
            'label_key': 'Houston_gt',
            'data_name': 'Houston2013.mat',
            'label_name': 'Houston2013_gt.mat',
            'resolution': (349, 1905, 144),
            'BAND': 144,
            'CLASSES_NUM': 15,
        }, 'Houston2018': {
            'data_key': 'Houston2018',
            'label_key': 'Houston2018_gt',
            'data_name': 'Houston2018_new.mat',
            'label_name': 'Houston2018_new_gt.mat',
            'resolution': (2384, 601, 50),
            'BAND': 50
        }, 'Xiongan': {
            'data_key': 'xiongan',
            'label_key': 'xiongan_gt',
            'data_name': 'Xiongan.mat',
            'label_name': 'Xiongan_gt.mat',
            'CLASSES_NUM': 20,
            'resolution': (1580, 3750, 256),
            'BAND': 256

        }, 'GF5': {
            'data_key': 'gf5',
            'label_key': 'gf5_gt',
            'data_name': 'gf5.mat',
            'label_name': 'gf5_gt.mat',
            'CLASSES_NUM': 20,
            'resolutino': (1400, 1400, 280),
            'BAND': 280
        }, 'Yancheng': {
            'data_key': 'yancheng',
            'label_key': 'yancheng_gt',
            'data_name': 'Yancheng.mat',
            'label_name': 'Yancheng_gt.mat',
            'CLASSES_NUM': 20,
            'resolution': (585, 1175, 266),
            'BAND': 266
        }
    }


class Pre_HSIDataset(Dataset):
    def __init__(self, dataset, labelset, len=0, patchsize=11, pca=None, is_train=True):
        '''
        :param data: [Salina:data, KSC:data, Indiana:data]
        :param label: [Salina, KSC, Indiana]  [datasetName, n_label, ori_label, (h,w)]
        :param patchsz: scale
        '''

        # 数据类型转换
        # if data.dtype != np.float32: data = data.astype(np.float32)
        # if label.dtype != np.int32: label = label.astype(np.int32)

        super(Dataset, self).__init__()
        self.PCA = pca
        self.patchsz = patchsize
        self.dataset = self.addMirror(dataset.astype(np.float32))
        print("hsi ori size {} padding size {}".format(dataset.shape, self.dataset.shape))

        # label_list 格式转换
        if is_train and len != 0:
            self.labelset = labelset[:len]

        else:
            self.labelset = labelset
        print("data_len:", self.labelset.__len__())

    def __len__(self):
        return self.labelset.__len__()

    def addMirror(self, data):
        dx = self.patchsz // 2
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def __getitem__(self, index):
        '''
        :param index:
        :return: 光谱信息， 标签
        '''
        move = self.patchsz // 2
        label, x, y = self.labelset[index]
        x += move
        y += move
        # print("d_shpe ", self.dataset.shape)
        # print("x,y : (%d,%d) range: x (%d,%d) y (%d,%d)" %(x,y,x - move,x + move,y - move,y + move))

        patch = self.dataset[x - move:x + move + 1, y - move: y + move + 1, :]
        Tar = torch.tensor(self.dataset[x, y, :], dtype=torch.float32)
        # if patch.shape[1] != 5 or patch.shape[2] != 5:
        #     print(self.dataset.shape)
        #     print("item  shape", patch.shape, "x,y", x,y)
        return torch.tensor(repeat(patch, 'a b c -> B a b c', B=1), dtype=torch.float32), torch.tensor(label - 1,
                                                                                                       dtype=torch.long), Tar


class HSIDataset(Dataset):
    def __init__(self, dataset, labelset, len=0, patchsize=11, pca=None, is_train=True):
        '''
        :param data: [Salina:data, KSC:data, Indiana:data]
        :param label: [Salina, KSC, Indiana]  [datasetName, n_label, ori_label, (h,w)]
        :param patchsz: scale
        '''

        # 数据类型转换
        # if data.dtype != np.float32: data = data.astype(np.float32)
        # if label.dtype != np.int32: label = label.astype(np.int32)

        super(Dataset, self).__init__()
        self.PCA = pca
        self.patchsz = patchsize
        self.dataset = self.addMirror(dataset.astype(np.float32))
        print("hsi ori size {} padding size {}".format(dataset.shape, self.dataset.shape))

        # label_list 格式转换
        if is_train and len != 0:
            self.labelset = labelset[:len]

        else:
            self.labelset = labelset
        print("data_len:", self.labelset.__len__())

    def __len__(self):
        return self.labelset.__len__()

    def addMirror(self, data):
        dx = self.patchsz // 2
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror

    def __getitem__(self, index):
        '''
        :param index:
        :return: 光谱信息， 标签
        '''
        move = self.patchsz // 2
        label, x, y = self.labelset[index]
        x += move
        y += move
        # print("d_shpe ", self.dataset.shape)
        # print("x,y : (%d,%d) range: x (%d,%d) y (%d,%d)" %(x,y,x - move,x + move,y - move,y + move))

        patch = self.dataset[x - move:x + move + 1, y - move: y + move + 1, :]
        # if patch.shape[1] != 5 or patch.shape[2] != 5:
        #     print(self.dataset.shape)
        #     print("item  shape", patch.shape, "x,y", x,y)
        return torch.tensor(repeat(patch, 'a b c -> B a b c', B=1), dtype=torch.float32), torch.tensor(label - 1, dtype=torch.long)


# input = torch.rand(10,64,49)
# res = repeat(input, 'a b c -> d a b c', d=2)
# print(res.shape)
class Linear_HSIDataset(Dataset):
    def __init__(self, dataset, labelset, len=0, pca=None, is_train=True):
        '''
        :param data: [Salina:data, KSC:data, Indiana:data]
        :param label: [Salina, KSC, Indiana]  [datasetName, n_label, ori_label, (h,w)]
        :param patchsz: scale
        '''

        # 数据类型转换
        # if data.dtype != np.float32: data = data.astype(np.float32)
        # if label.dtype != np.int32: label = label.astype(np.int32)

        super(Dataset, self).__init__()
        self.PCA = pca
        # self.patchsz = patchsize
        self.dataset = dataset.astype(np.float32)
        print("hsi ori size {} padding size {}".format(dataset.shape, self.dataset.shape))

        # label_list 格式转换
        if is_train and len != 0:
            self.labelset = labelset[:len]

        else:
            self.labelset = labelset
        print("data_len:", self.labelset.__len__())

    def __len__(self):
        return self.labelset.__len__()

    def __getitem__(self, index):
        '''
        :param index:
        :return: 光谱信息， 标签
        '''

        label, x, y = self.labelset[index]

        # print("d_shpe ", self.dataset.shape)
        # print("x,y : (%d,%d) range: x (%d,%d) y (%d,%d)" %(x,y,x - move,x + move,y - move,y + move))

        patch = self.dataset[x, y, :]
        # if patch.shape[1] != 5 or patch.shape[2] != 5:
        #     print(self.dataset.shape)
        #     print("item  shape", patch.shape, "x,y", x,y)
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label - 1, dtype=torch.long)