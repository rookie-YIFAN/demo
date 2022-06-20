'''模型最终训练'''
import torch
from torch import nn, optim
from scipy.io import loadmat, savemat
from utils import train, test, traint, model_eval
from HSIdataset import HSIDataset, DatasetInfo
from sklearn.preprocessing import scale
# from M_encoder import SP_T
from model.FGB_MST import SP_T

from torch.utils.data import DataLoader
from Datagenerator import data_generator
import pandas as pd
import time


# from visdom import Visdom

# ------------------- 基础设置 --------------------

NUM_WORKERS = 1
SEED = 468
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# ---------------------- 数据集路径配置 --------------------
datasetName = 'PaviaU'
FolderPath = '/home/wyf/PublicFolder/wyf/data/ori_data/'
FolderPath = '/mnt/ori_data/'

# sampleperclass = 50  # 读取样本数
rate = 0.10
del_list = [0]  # 删除类别
BATCHSZ = 8
PatchSZ = 13
RUN = 10
SAMPLE_PER_CLASS = [5]
# SAMPLE_PER_CLASS = [0.05, 0.08, 0.1, 0.15, 0.2]
modelname = "MST"

PRE_MODEL = "/home/wyf/CODE/20211210/Base_SPT/Ft_model_zoo/PaviaU/PaviaU best epoch 460 los 0.22460.pkl"
PRE_MODEL = '/mnt/WYF/MST/pretrain_res/PaviaU/PaviaU best epoch 189 los 0.21272.pkl'


train_type = 'normal'
band = 80


def main(datasetName, run, sample=0, train_type= train_type):
    # ------------------------ 数据集读取 ------------------------
    if sample:
        data, train_label, test_label, CLASSES_NUM, BAND  = data_generator(datasetName, FolderPath,sampleperclass=sample, del_list=del_list, pca=band)
        print("run {} batch size {} sampleperclass {}".format(run, BATCHSZ, sample))

    elif sample<1 and sample>0:
        data, train_label, test_label, CLASSES_NUM, BAND  = data_generator(datasetName, FolderPath, rate=sample, del_list=del_list, pca=band)
        print("run {} batch size {}  rate {} ".format(run, BATCHSZ, rate))
    else:
        raise("input sample err ")

    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = scale(data)
    data = data.reshape((h, w, c))


    trainDataset = HSIDataset(data, train_label, patchsize=PatchSZ)
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)

    testDataset = HSIDataset(data, test_label, patchsize=PatchSZ, is_train=False)
    testLoader = DataLoader(testDataset, batch_size=512, shuffle=True, num_workers=NUM_WORKERS)

    # ------------------- 模型及超参数 --------------------
    # Patchsz=13, emb_dim=8*16, band=80, group=8, group_out=16
    TDFE_hyper_params = (13, 8 * 16, 80, 8, 16)
    # SP_T(TDFE_hyper_params, 9, 13, 8*16, 4*16, 4).cuda(DEVICE)
    model = SP_T(TDFE_hyper_params, CLASSES_NUM, 13, 8 * 16, 4 * 16, 4).cuda(DEVICE)

    EPOCHS = 20

    if 'pretrain' in train_type:
        if PRE_MODEL:
            # model.load_state_dict(torch.load(PRE_MODEL, map_location='cpu'))
            model_dict = model.state_dict()
            pretext_model = torch.load(PRE_MODEL)
            state_dict = {k: v for k, v in pretext_model.items() if 'spectral_cls_layer' not in k}
            print(state_dict)
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print("{} load success".format(PRE_MODEL))
        else:
            print("pre train model path no found")

    # ------------ 损失函数 优化器 学习率下架管理器 ---------------
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

    # ------------ 训练 1---------------
    B_AA, B_OA, B_KA, B_RC = 0.0, 0.0, 0.0,[]
    B_AA, B_OA, B_KA, B_RC, model_dict = traint(model, trainLoader, testLoader, criterion, optimizer, DEVICE, 40, early_stopping=True, early_num=50)

    # ------------ 训练 2 ---------------
    # 训练参数记载 3(trainloss, evalloss, acc) * epochs
    # res = torch.zeros((3, EPOCHS))
    best_acc = 0

    # for epoch in range(EPOCHS):
    #     print('*' * 5 + 'Epoch:{}'.format(epoch) + '*' * 5)
    #     model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader, device=DEVICE)
    #     # acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader, device=DEVICE)
    #     class_recall, AA, OA, Ka, evalLoss= model_eval(testLoader, model, criterion, DEVICE)
    #
    #     print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, OA))
    #     print('*' * 18)
    #     if OA > B_OA:
    #        B_AA, B_OA, B_KA, B_RC = AA, OA, Ka, class_recall

    print("best acc AA {} OA {} Ka {}".format(B_AA, B_OA, B_KA))
    return B_AA, B_OA, B_KA, B_RC




def res2excel(res,path):
    avg = []
    writer = pd.ExcelWriter(path)
    for idx, i in enumerate(['OA_r', 'AA_r', 'Ka_r', 'class_recall_r']):
        res[idx] = pd.DataFrame(res[idx])
        avg.append(res[idx].mean())
        res[idx].to_excel(excel_writer=writer,sheet_name=i,index=False)
    print("totol ave res: ", avg)
    pd.DataFrame(avg).to_excel(excel_writer=writer, sheet_name="avg", index=False)
    writer.save()
    writer.close()



if __name__ == '__main__':
    for i, sample in enumerate(SAMPLE_PER_CLASS):
        class_recall_r, AA_r, OA_r, Ka_r = [], [], [],[]
        print('*' * 5 + 'sample:{}'.format(rate) + '*' * 5)
        for r in range(RUN):
            print('*' * 5 + 'run:{}'.format(r) + '*' * 5)

            AA, OA, KA, class_recall = main(datasetName, r, sample=sample, train_type=train_type)

            AA_r.append(AA)
            OA_r.append(OA)
            Ka_r.append(KA)
            class_recall_r.append(class_recall)

            if sample > 0:
                print("num {} run {} OA {:.4f} AA {:.4f} ka {:.4f}".format(sample, r, OA, AA, KA))
            else:
                print("rate {} run {} OA {:.4f} AA {:.4f} ka {:.4f}".format(sample, r, OA, AA, KA))

        times = time.strftime("%Y-%m-%d %X")

        res = [OA_r, AA_r, Ka_r, class_recall_r]
        if sample > 1:
            path = "res/{}/{} num {}[{}].xlsx".format(datasetName, times, sample, train_type)
        else:
            path = "res/{}/{} rate {}[{}].xlsx".format(datasetName, times, sample, train_type)

        res2excel(res, path)

    print("finish time : {}".format(times))
