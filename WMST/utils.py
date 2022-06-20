import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import random
import numpy as np
import time
# import record
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score

# 模型训练
def train(model, criterion, optimizer, dataLoader, device):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(device)
    trainLoss = []
    for input, target in tqdm(dataLoader):
        input, target = input.to(device), target.to(device)
        
        out = model(input)
        
        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, float(np.mean(trainLoss))


# 模型测试
def test(model, criterion, dataLoader, device):
    model.eval()
    model.to(device)
    evalLoss, correct = [], 0
    for input, target in tqdm(dataLoader):
        input, target = input.to(device), target.to(device)
       
        logits = model(input)
       
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)


def model_eval(data_iter, net, criterion, device):
    i = 0
    pred_li = []
    gt_li = []
    evalLoss = []
    net.eval()

    with torch.no_grad():
        for X, y in tqdm(data_iter):
            # if i == 0:
            #     print("valida_iter", len(data_iter) * len(y))
            #     i += 1
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)

            y_hat = net(X)
            loss = criterion(y_hat, y)
            evalLoss.append(loss.item())

            y_pred = y_hat.argmax(dim=1).cpu().numpy()
            # print(y_pred,y.cpu().numpy())
            pred_li.extend(y_pred)
            gt_li.extend(y.cpu().numpy())
    net.train()
    # 计算类别 recall 值
    class_recall = recall_score(gt_li, pred_li, average=None)
    # 计算平均 recall
    AA = class_recall.mean()
    # 计算准确率
    OA = accuracy_score(gt_li, pred_li)
    # 计算kappa
    Ka =  cohen_kappa_score(gt_li, pred_li)

    return class_recall, AA, OA, Ka, np.mean(evalLoss)


# 模型训练2
def traint( net, train_iter, valida_iter, loss, optimizer, device, epochs, early_stopping=True, early_num=50):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)

    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    i = 0
    B_AA, B_OA, B_KA, B_RC = 0., 0., 0., None

    for epoch in range(epochs):
        i = 0
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        for X, y in tqdm(train_iter):
            # if i < 1:
            #     print("tar ", y)
            #     i += 1
            #     print("train iter", len(train_iter) * len(y))

            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step()
        class_recall, AA, OA, Ka, valida_loss = model_eval(valida_iter, net, loss, device)
        # valida_acc, valida_loss = test(net, criterion=loss, dataLoader=valida_iter, device=device)
        # valida_acc, valida_loss = record.evaluate_accuracy(valida_iter, net, loss, device)
        if OA > B_OA:
            B_AA, B_OA, B_KA, B_RC = AA, OA, Ka, class_recall
            model_state_dict = net.state_dict()

        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(OA)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               valida_loss, OA, time.time() - time_epoch))
        # print(class_recall)
        print("best acc AA {} OA {} Ka {}".format(B_AA, B_OA, B_KA))
        print("branch Weight: ", net.localFE.branchWeight)

    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - start))

    return B_AA, B_OA, B_KA, B_RC, model_state_dict



# 数据预处理
def preprocess(data, n_components=None):
    h, w, _ = data.shape
    if not data.dtype==np.float32:
        data = data.astype(np.float32)
    # 数据归一化
    data = data.reshape((h*w, -1))
    data -= np.min(data, axis=0)
    data /= np.max(data, axis=0)
    # PCA降维
    pca = PCA(0.99, whiten=True) if n_components is None else PCA(n_components)
    pca.fit(data)
    if len(pca.singular_values_) < 10:
        pca = PCA(10)
        data = pca.fit_transform(data)
    else:
        data = pca.transform(data)
    return data.reshape((h, w, -1))