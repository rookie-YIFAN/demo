import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torchsummary import summary


# have np_maxpooling
class TransEncoder(nn.Module):
    def __init__(self, emb_size=60, proj_size=32, nhead=4):
        super(TransEncoder, self).__init__()

        self.emb_size = emb_size

        self.mulitheadAtt1 = nn.MultiheadAttention(emb_size, nhead, dropout=0.25)

        self.layernorm = nn.LayerNorm(emb_size)

        self.ff1_1 = nn.Linear(emb_size, proj_size)
        self.ff1_2 = nn.Linear(proj_size, emb_size)

        self.dropout = nn.Dropout(0.25)

        self.FF1 = nn.Sequential(
            self.ff1_1,
            nn.ReLU(inplace=True),
            self.ff1_2,
            self.dropout
        )

    def forward(self, x):
        src1, _ = self.mulitheadAtt1(x, x, x)
        src1_n1 = self.layernorm(src1 + x)
        src1_n2 = self.layernorm(self.FF1(src1_n1) + src1_n1)

        return src1_n2


class TDFE_G(nn.Module):
    def __init__(self, Patchsz, emb_dim, band, group, group_out):
        '''
        :param Patchsz:  patchSize of input sample
        :param emb_dim:  embedding dimension of single pixel
        :param band: band of input sample
        :param group:  grouping num of band
        :param group_out: embedding dimension for every group feature
        '''

        super(TDFE_G, self).__init__()

        print("TDFE branch running")
        self.name = 'TDFE'

        self.G = group
        self.G_dim = int(band / group)

        self.group_conv_num = 4

        self.group_emb_in = self.G_dim - 2 - 2
        self.group_emb_out = group_out

        self.batchNum1 = Patchsz*Patchsz
        self.batchNum2 = (Patchsz-4)*(Patchsz-4)

        self.conv1 = nn.Sequential(
            Rearrange("B t w h (C D) -> B (t C) w h D", C=self.G, D=self.G_dim),
            nn.Conv3d(
                in_channels=self.G,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv_m = nn.Sequential(
            Rearrange("B t w h (C D) -> B (t C) w h D", C=self.G, D=self.G_dim),
            nn.Conv3d(
                in_channels=self.G,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.G * self.group_conv_num,
                out_channels=self.G * self.group_conv_num,
                kernel_size=(3, 3, 3),
                groups=self.G),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(self.G * self.group_conv_num, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
        )

        self.conv3 = nn.Sequential(
            Rearrange("B t w h b -> B (t b) w h "),
            nn.Conv2d(in_channels= self.G * self.group_conv_num * self.group_emb_in, out_channels=group_out * self.G, kernel_size=1, padding=0, stride=1, groups=self.G),
            nn.ReLU(inplace=True),
            Rearrange("B a b c -> B  (b c) a"),
        )

        self.conv4 = nn.Sequential(
            nn.Linear(group_out * group, emb_dim),
            # Rearrange("B a b  -> B b a"),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            Rearrange("B t w h b -> B (w h) (t b) "),
            nn.BatchNorm1d(self.batchNum2),
            nn.Linear(band, emb_dim),
            # Rearrange("B a b  -> B b a"),
            nn.ReLU(inplace=True)
        )


        self.Linear = nn.Sequential(
            Rearrange("B t w h b -> B (t b) w h"),
            nn.BatchNorm2d(band),
            nn.Conv2d(band, emb_dim, groups=8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            Rearrange("B o w h  -> B (w h) o")

        )

        self.convMedia = nn.Sequential(
            Rearrange("B (a b) c d e -> B b a (c d) e ", a=8, b=4),
            Rearrange("B a b c d -> B (c a) (b d)"),
            nn.Conv1d(in_channels=81 * 4, out_channels=81 * 2, kernel_size=3, groups=81, padding=1),
            Rearrange("B (a b) c -> B a (b c)", a=81, b=2)
        )

    def forward(self, X):


        x = self.conv1(X)

        # z = self.convMedia(x);
        z = x[:,:,1:-1,1:-1,:]
        middleScale = self.convMedia(z)

        x = self.conv2(x)

        # 全连接特征融合
        x = self.conv3(x)

        # 波段特征统一
        x = self.conv4(x)

        y = X[:,:,2:-2,2:-2,:]

        # print("z shape", z.shape)
        # residual = self.conv5(y)
        residual = self.Linear(y)

        # middleScale = self.co

        return x + residual + middleScale
        # print("x shape", x.shape)

        # return x




class SP_T(nn.Module):
    # model = SP_T(TDFE_hyper_params, 9, 13, 8*16, 4*16, 4).cuda()
    def __init__(self, TDFE_hyper_params, class_num, patchSZ, emb_size=60, proj_size=32, nhead=4):
        super(SP_T,self).__init__()
        print("MergeT  running")
        self.name = 'MergeT'
        self.NornNum = (patchSZ-4)*(patchSZ-4)
        self.localFE = TDFE_G(*TDFE_hyper_params)


        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))  # 用于分类的token - 创新点

        self.spectral = nn.Sequential(
            # Rearrange("B a b  -> B b a "),
            TransEncoder(emb_size, proj_size, nhead),
            TransEncoder(emb_size, proj_size, nhead)
        )

        self.spectral_cls_layer = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.2),
            nn.Linear(emb_size, class_num)
        )

        self.spectral_cls_layer1 = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 64),
            nn.Dropout(0.3),
            nn.Linear(64, class_num),
        )

        self.spectral_cls_layer2 = nn.Sequential(
            nn.AvgPool2d((self.NornNum, 1), stride=1),
            Rearrange("B a b -> B (a b)"),
            nn.Dropout(0.3),
            nn.Linear(emb_size, emb_size),
            nn.Dropout(0.3),
            nn.Linear(emb_size, 64),
            nn.Dropout(0.3),
            nn.Linear(64, class_num),
        )

    def forward(self, x):
        local_res = self.localFE(x)
        local_out = self.spectral(local_res)
        cls_res = self.spectral_cls_layer(local_out)

        return cls_res

    def encoder(self, x):
        local_res = self.localFE(x)
        local_out = self.spectral(local_res)
        return local_out








if __name__ == '__main__':
    # model = SP_T(9).cuda()
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # model = TDFE_G(Patchsz=13, emb_dim=8*16, band=80, group=8, group_out=16).cuda()
    # summary(model, (1, 13, 13, 80))
    #

    # input = torch.rand((1, 1, 11, 11, 60)).cuda()
    # print(model(input).shape)

    TDFE_hyper_params = (13, 8*16, 80, 8, 16)
    # # class_num, patchSZ, emb_size = 60, proj_size = 32, nhead = 4
    model = SP_T(TDFE_hyper_params, 9, 13, 8*16, 4*16, 4).cuda()
    summary(model, (1, 13, 13, 80))

    # summary(model, (1, 13, 13, 80))
    # input = torch.rand((1, 1, 13, 13, 80)).cuda(DEVICE)
    #     # encoder_out = model.encoder(input)
    #     # print(encoder_out.shape)
    #     #
    #     # decoder = M_Decoder(
    #     #     emb_num = 9*9,
    #     #     input_dim=8 * 16,
    #     #     emb_dim=256,
    #     #     depth=2,  # encoder 堆叠个数
    #     #     heads=8,  # 多少个头
    #     #     hidden_dim=128,  # mlp分类
    #     #     BH_dim=256,
    #     #     target_dim=80,
    #     #     dim_head=int(256/8),
    #     #     dropout=0,  # encoder用的
    #     #     emb_dropout=0,  # 位置嵌入用的
    #     #     DEVICE=DEVICE
    #     # ).cuda(DEVICE)
    #     #
    #     # print(decoder(encoder_out).shape)

    # model = TDFE(*TDFE_hyper_params).cuda()
    # # summary(model, (1, 13, 13, 80))
    #
    #
    # conv1 = nn.Conv1d(in_channels=81*4, out_channels=81*2, kernel_size=3,groups=81,padding=1)
    # input = torch.randn(1, 64, 81*4)
    # # batch_size x entity_len x embedding_size -> batch_size x embedding_size x text_len
    # input = input.permute(0, 2, 1)
    # out = conv1(input)
    # print(out.size())