from collections import deque
import sys
import numpy as np
from torch import torch, cat, nn
# import torch.nn.functional as F
import torchvision.models as models
# import torchvision.transforms as transforms




#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_init_layer(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    # layer.bias.data.fill_(0.01)

def kaiming_init(m):
    # print(m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # m.bias.data.fill_(0.01)

"""
class DeConvBlk(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=0, final=False):
        super(DeConvBlk, self).__init__()
        if final:
            self.deconv = nn.ConvTranspose2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
            self.act = nn.Softmax(dim=1)
        else:
            self.deconv = nn.ConvTranspose2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
            self.act = nn.Sequential(
                nn.BatchNorm2d(channelx[1]),
                nn.ReLU(),
            )

        #weights initialization
        # kaiming_w_init(self.conv)
    
    def forward(self, x):
        x = self.deconv(x) 
        y = self.act(x)
        return y
"""


class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        #weights initialization
        # kaiming_w_init(self.conv)
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, final=False): #up, 
        super(ConvBlock, self).__init__()
        #conv block
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
        #init
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)
 
    def forward(self, x):
        #convolutional block
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y




class huang(nn.Module): #
    #default input channel adalah 4 untuk RGBD
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(huang, self).__init__()
        self.config = config
        self.gpu_device = device

        #------------------------------------------------------------------------------------------------
        #RGBD
        # self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGBD_encoder = models.resnet50(pretrained=False) #resnet18
        #input conv pertama diganti untuk menerima 4 channel RGBD
        self.RGBD_encoder.conv1 = nn.Conv2d(4, config.n_fmap_r50[0], kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.RGBD_encoder.fc = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGBD_encoder.avgpool = nn.Sequential() 
        #SS
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_r50[4]+config.n_fmap_r50[3], config.n_fmap_r50[3]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[config.n_fmap_r50[3]+config.n_fmap_r50[2], config.n_fmap_r50[2]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[config.n_fmap_r50[2]+config.n_fmap_r50[1], config.n_fmap_r50[1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[config.n_fmap_r50[1]+config.n_fmap_r50[0], config.n_fmap_r50[0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[config.n_fmap_r50[0], config.n_class], final=True)#, up=False)
        # self.dconv3_ss_f = DeConvBlk(channelx=[config.n_fmap_r50[4], config.n_fmap_huang[0]], stridex=config.stride_huang[0])
        # self.dconv2_ss_f = DeConvBlk(channelx=[config.n_fmap_r50[3]+config.n_fmap_huang[0], config.n_fmap_huang[1]], stridex=config.stride_huang[1])
        # self.dconv1_ss_f = DeConvBlk(channelx=[config.n_fmap_r50[2]+config.n_fmap_huang[1], config.n_fmap_huang[2]], stridex=config.stride_huang[2])
        # self.dconv0_ss_f = DeConvBlk(channelx=[config.n_fmap_r50[1]+config.n_fmap_huang[2], config.n_fmap_huang[3]], stridex=config.stride_huang[3])
        # self.final_ss_f = DeConvBlk(channelx=[config.n_fmap_r50[0]+config.n_fmap_huang[3], config.n_fmap_huang[4]], stridex=config.stride_huang[4], final=True)

        #------------------------------------------------------------------------------------------------
        self.global_pool = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        #------------------------------------------------------------------------------------------------
        #controller ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        self.ctrl_branch = nn.ModuleList([nn.Sequential( 
            nn.Linear(config.n_fmap_r50[4], config.control_huang[0]),
            nn.Linear(config.control_huang[0], config.control_huang[1]),
            nn.Linear(config.control_huang[1], config.control_huang[2]),
            nn.Linear(config.control_huang[2], 2),
        ) for _ in range(config.n_cmd)]) #.to(self.gpu_device, dtype=torch.float)


    def forward(self, rgbs, deps, cmd):#, velo_in):
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGBD_features_sum = 0
        segs_f = []
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgbd = cat([rgbs[i]/255.0, deps[i]], dim=1) #normalisasi 0-1 saja, tidak perlu self.rgb_normalizer(
            x_RGBD = self.RGBD_encoder.conv1(in_rgbd)
            x_RGBD = self.RGBD_encoder.bn1(x_RGBD)
            RGB_features0 = self.RGBD_encoder.relu(x_RGBD)
            RGB_features1 = self.RGBD_encoder.layer1(self.RGBD_encoder.maxpool(RGB_features0))
            RGB_features2 = self.RGBD_encoder.layer2(RGB_features1)
            RGB_features3 = self.RGBD_encoder.layer3(RGB_features2)
            RGB_features4 = self.RGBD_encoder.layer4(RGB_features3)
            RGBD_features_sum += RGB_features4

            #bagian segmentation
            ss_f_3 = self.conv3_ss_f(cat([self.up(RGB_features4), RGB_features3], dim=1))
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features2], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features1], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features0], dim=1))
            ss_f = self.final_ss_f(self.up(ss_f_0))
            segs_f.append(ss_f)
            # ss_f_3 = self.dconv3_ss_f(RGB_features4)
            # ss_f_2 = self.dconv2_ss_f(cat([ss_f_3, RGB_features3], dim=1))
            # ss_f_1 = self.dconv1_ss_f(cat([ss_f_2, RGB_features2], dim=1))
            # ss_f_0 = self.dconv0_ss_f(cat([ss_f_1, RGB_features1], dim=1))
            # ss_f = self.final_ss_f(cat([ss_f_0, RGB_features0], dim=1))
            # segs_f.append(ss_f)
        
        latent_features = self.global_pool(RGBD_features_sum)
        # print(latent_features.shape)

        #------------------------------------------------------------------------------------------------
        #control decoder #cmd ada 3, 0 = lurus, 1 = belok kiri, 2 = belok kanan
        #sementara ini terpaksa loop sepanjang batch dulu, ga tau caranya supaya langsung
        # print(cmd)
        # print(len(cmd))
        # print(cmd.shape)
        control_pred = self.ctrl_branch[cmd[0].item()](latent_features[0:1,:])
        for i in range(1, len(cmd)): 
            # print("-----------")
            # print(cmd[i].item())
            # print(latent_features[i:i+1,:].shape)
            control_pred = cat([control_pred, self.ctrl_branch[cmd[i].item()](latent_features[i:i+1,:])], dim=0) #concat di axis batch
            # print(control_pred.shape)
        #denormalisasi
        # print(control_pred.shape)
        steering = control_pred[:,0] * 2 - 1.0 # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        #------------------------------------------------------------------------------------------------

        return segs_f, steering, throttle
