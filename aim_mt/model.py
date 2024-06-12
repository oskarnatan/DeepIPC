from collections import deque
import sys
import numpy as np
from torch import torch, cat, nn
# import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms




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
            nn.Conv2d(channel[0], channel[1], kernel_size=1)
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


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control



class aim_mt(nn.Module): #
    #default input channel adalah 3 untuk RGB, 2 untuk DVS, 1 untuk LiDAR
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(aim_mt, self).__init__()
        self.config = config
        self.gpu_device = device
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.resnet34(pretrained=True) #resnet18
        self.RGB_encoder.fc = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential() 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        #SS
        self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_r34[4]+config.n_fmap_r34[3], config.n_fmap_r34[3]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[config.n_fmap_r34[3]+config.n_fmap_r34[2], config.n_fmap_r34[2]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[config.n_fmap_r34[2]+config.n_fmap_r34[1], config.n_fmap_r34[1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[config.n_fmap_r34[1]+config.n_fmap_r34[0], config.n_fmap_r34[0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[config.n_fmap_r34[0], config.n_class], final=True)#, up=False)
        #DEPTH
        self.conv3_dep_f = ConvBlock(channel=[config.n_fmap_r34[4]+config.n_fmap_r34[3], config.n_fmap_r34[3]])#, up=True)
        self.conv2_dep_f = ConvBlock(channel=[config.n_fmap_r34[3]+config.n_fmap_r34[2], config.n_fmap_r34[2]])#, up=True)
        self.conv1_dep_f = ConvBlock(channel=[config.n_fmap_r34[2]+config.n_fmap_r34[1], config.n_fmap_r34[1]])#, up=True)
        self.conv0_dep_f = ConvBlock(channel=[config.n_fmap_r34[1]+config.n_fmap_r34[0], config.n_fmap_r34[0]])#, up=True)
        self.final_dep_f = ConvBlock(channel=[config.n_fmap_r34[0], 1], final=True)#, up=False)
        #------------------------------------------------------------------------------------------------
        #feature fusion
        self.necks_net = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_r34[4], config.n_fmap_r34[3]), #512 - 256
            nn.Linear(config.n_fmap_r34[3], config.n_fmap_r34[2]), #256 - 128
            nn.Linear(config.n_fmap_r34[2], config.n_fmap_r34[1]), #128 - 64
        )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 6 karena concat dari wp xy dan rp1 xy rp2 xy
        self.gru = nn.GRUCell(input_size=6, hidden_size=config.n_fmap_r34[1])
        self.pred_dwp = nn.Linear(config.n_fmap_r34[1], 2)
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        #------------------------------------------------------------------------------------------------


    def forward(self, rgbs, rp1, rp2):#, velo_in):
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGB_features_sum = 0
        segs_f = []
        deps_f = []
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgb = self.rgb_normalizer(rgbs[i]) #
            x_RGB = self.RGB_encoder.conv1(in_rgb)
            x_RGB = self.RGB_encoder.bn1(x_RGB)
            RGB_features0 = self.RGB_encoder.relu(x_RGB)
            RGB_features1 = self.RGB_encoder.layer1(self.RGB_encoder.maxpool(RGB_features0))
            RGB_features2 = self.RGB_encoder.layer2(RGB_features1)
            RGB_features3 = self.RGB_encoder.layer3(RGB_features2)
            RGB_features4 = self.RGB_encoder.layer4(RGB_features3)
            RGB_features_sum += RGB_features4
            up_RGB_features4 = self.up(RGB_features4)
            #bagian segmentation
            ss_f_3 = self.conv3_ss_f(cat([up_RGB_features4, RGB_features3], dim=1))
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features2], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features1], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features0], dim=1))
            ss_f = self.sigmoid(self.final_ss_f(self.up(ss_f_0)))
            segs_f.append(ss_f)
            #bagian depth estimation
            dep_f_3 = self.conv3_dep_f(cat([up_RGB_features4, RGB_features3], dim=1))
            dep_f_2 = self.conv2_dep_f(cat([self.up(dep_f_3), RGB_features2], dim=1))
            dep_f_1 = self.conv1_dep_f(cat([self.up(dep_f_2), RGB_features1], dim=1))
            dep_f_0 = self.conv0_dep_f(cat([self.up(dep_f_1), RGB_features0], dim=1))
            dep_f = self.relu(self.final_dep_f(self.up(dep_f_0)))
            deps_f.append(dep_f)

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(RGB_features_sum)
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.gpu_device, dtype=hx.dtype)
        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)
        #------------------------------------------------------------------------------------------------

        return segs_f, deps_f, pred_wp

    def pid_control(self, pwaypoints, linear_velo): #angular_velo
        assert(pwaypoints.size(0)==1)
        waypoints = pwaypoints[0].data.cpu().numpy()
        
        #vehicular controls dari PID
        aim_point = (waypoints[1] + waypoints[0]) / 2.0 #tengah2nya wp0 dan wp1
        #90 deg ke kanan adalah 0 radian, 90 deg ke kiri adalah 1*pi radian
        angle_rad = np.clip(np.arctan2(aim_point[1], aim_point[0]), 0, np.pi) #arctan y/x
        angle_deg = np.degrees(angle_rad)
        #ke kiri adalah 0 -> +1 == 90 -> 180, ke kanan adalah 0 -> -1 == 90 -> 0
        error_angle = (angle_deg - 90.0) * self.config.err_angle_mul
        pid_steering = self.turn_controller.step(error_angle)
        pid_steering = np.clip(pid_steering, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * self.config.des_speed_mul
        pid_throttle = self.speed_controller.step(desired_speed - linear_velo)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)

        
        if pid_throttle < self.config.min_act_thrt:
            steering = 0.0 #dinetralkan
            throttle = 0.0
        else:
            steering = pid_steering
            throttle = pid_throttle

        
        metadata = {
            # 'lr_velo': [float(angular_velo[0].astype(np.float64)), float(angular_velo[1].astype(np.float64))],
            'linear_velo' : float(linear_velo),
            'steering': float(steering),
            'throttle': float(throttle),
            'pid_steering': float(pid_steering),
            'pid_throttle': float(pid_throttle),
            'wp_4': [float(waypoints[3][0].astype(np.float64)), float(waypoints[3][1].astype(np.float64))],
            'wp_3': [float(waypoints[2][0].astype(np.float64)), float(waypoints[2][1].astype(np.float64))], #tambahan
            'wp_2': [float(waypoints[1][0].astype(np.float64)), float(waypoints[1][1].astype(np.float64))],
            'wp_1': [float(waypoints[0][0].astype(np.float64)), float(waypoints[0][1].astype(np.float64))],
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle_deg.astype(np.float64)),
            'aim': [float(aim_point[0].astype(np.float64)), float(aim_point[1].astype(np.float64))],
            # 'delta': float(delta.astype(np.float64)),
            'robot_pos': None, #akan direplace nanti
            'robot_bearing': None,
            'rp1': None, #akan direplace nanti
            'rp2': None, #akan direplace nanti
            'fps': None,
            'model_fps': None,
            'intervention': False,
        }
        return steering, throttle, metadata


