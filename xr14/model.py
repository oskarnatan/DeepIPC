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



class xr14(nn.Module): #
    #default input channel adalah 3 untuk RGB, 2 untuk DVS, 1 untuk LiDAR
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(xr14, self).__init__()
        self.config = config
        self.gpu_device = device
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(pretrained=True) #efficientnet_b4
        self.RGB_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan 
        #SS
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_b3[4][-1]+config.n_fmap_b3[3][-1], config.n_fmap_b3[3][-1]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[config.n_fmap_b3[3][-1]+config.n_fmap_b3[2][-1], config.n_fmap_b3[2][-1]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[config.n_fmap_b3[2][-1]+config.n_fmap_b3[1][-1], config.n_fmap_b3[1][-1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[config.n_fmap_b3[1][-1]+config.n_fmap_b3[0][-1], config.n_fmap_b3[0][0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[config.n_fmap_b3[0][0], config.n_class], final=True)#, up=False)
        #------------------------------------------------------------------------------------------------

        #untuk semantic cloud generator
        self.cover_area = config.coverage_area
        self.n_class = config.n_class
        self.h, self.w = int(config.crop_roi[0]/config.scale), int(config.crop_roi[1]/config.scale)
        #SC
        self.SC_encoder = models.efficientnet_b1(pretrained=False) #efficientnet_b0
        self.SC_encoder.features[0][0] = nn.Conv2d(config.n_class, config.n_fmap_b1[0][0], kernel_size=3, stride=2, padding=1, bias=False) #ganti input channel conv pertamanya, buat SC cloud
        self.SC_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.SC_encoder.avgpool = nn.Sequential()
        self.SC_encoder.apply(kaiming_init)
        #------------------------------------------------------------------------------------------------
        #feature fusion
        self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(config.n_fmap_b3[4][-1]+config.n_fmap_b1[4][-1], config.n_fmap_b3[4][1], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][1], config.n_fmap_b3[4][0])
        )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 8 karena concat dari wp xy, rp1 xy, rp2 xy, dan velocity lr
        self.gru = nn.GRUCell(input_size=8, hidden_size=config.n_fmap_b3[4][0])
        self.pred_dwp = nn.Linear(config.n_fmap_b3[4][0], 2)
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        #------------------------------------------------------------------------------------------------
        #controller
        #MLP Controller
        self.controller = nn.Sequential(
            nn.Linear(config.n_fmap_b3[4][0], config.n_fmap_b3[3][-1]),
            nn.Linear(config.n_fmap_b3[3][-1], 2), #str dan thrt
            nn.ReLU()
        )

    def forward(self, rgbs, pt_cloud_xs, pt_cloud_zs, rp1, rp2, velo_in):#, gt_ss):
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGB_features_sum = 0
        SC_features_sum = 0
        segs_f = []
        sdcs = []
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgb = self.rgb_normalizer(rgbs[i]) #
            RGB_features0 = self.RGB_encoder.features[0](in_rgb)
            RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
            RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
            RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
            RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
            RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
            RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
            RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
            RGB_features8 = self.RGB_encoder.features[8](RGB_features7)
            RGB_features_sum += RGB_features8
            #bagian upsampling
            ss_f_3 = self.conv3_ss_f(cat([self.up(RGB_features8), RGB_features5], dim=1))
            ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features3], dim=1))
            ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features2], dim=1))
            ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features1], dim=1))
            ss_f = self.final_ss_f(self.up(ss_f_0))
            segs_f.append(ss_f)
            #------------------------------------------------------------------------------------------------
            #buat semantic cloud
            top_view_sc = self.gen_top_view_sc_ptcloud(pt_cloud_xs[i], pt_cloud_zs[i], ss_f)
            sdcs.append(top_view_sc)
            #bagian downsampling
            SC_features0 = self.SC_encoder.features[0](top_view_sc)
            SC_features1 = self.SC_encoder.features[1](SC_features0)
            SC_features2 = self.SC_encoder.features[2](SC_features1)
            SC_features3 = self.SC_encoder.features[3](SC_features2)
            SC_features4 = self.SC_encoder.features[4](SC_features3)
            SC_features5 = self.SC_encoder.features[5](SC_features4)
            SC_features6 = self.SC_encoder.features[6](SC_features5)
            SC_features7 = self.SC_encoder.features[7](SC_features6)
            SC_features8 = self.SC_encoder.features[8](SC_features7)
            SC_features_sum += SC_features8

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hx = self.necks_net(cat([RGB_features_sum, SC_features_sum], dim=1))
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.gpu_device, dtype=hx.dtype)
        #predict delta wp
        out_wp = list()
        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)
            # if nwp == 1: #ambil hidden state ketika sampai pada wp ke 2, karena 3, 4, dan 5 sebenarnya tidak dipakai
            #     hx_mlp = torch.clone(hx)
        pred_wp = torch.stack(out_wp, dim=1)
        #------------------------------------------------------------------------------------------------
        #control decoder
        control_pred = self.controller(hx) #cat([hid_states, hid_state_nxr, hid_state_vel], dim=1)
        steering = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle

        return segs_f, pred_wp, steering, throttle, sdcs


    def gen_top_view_sc_ptcloud(self, pt_cloud_x, pt_cloud_z, semseg):
        #proses awal
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*self.w)] for n in range(semseg.shape[0])])).to(self.gpu_device, dtype=semseg.dtype)

        #normalize ke frame 
        cloud_data_x = torch.round((pt_cloud_x + self.cover_area) * (self.w-1) / (2*self.cover_area)).ravel()
        cloud_data_z = torch.round((pt_cloud_z * (1-self.h) / self.cover_area) + (self.h-1)).ravel()

        #cari index interest
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        #stack n x z cls dan plot
        # print(cloud_data_n.shape)
        # print(label_img.ravel().shape)
        # print(cloud_data_z.shape)
        # print(cloud_data_x.shape)
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW

        return top_view_sc


    def mlp_pid_control(self, pwaypoints, angular_velo, psteer, pthrottle):
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
        linear_velo = np.mean(angular_velo) * self.config.wheel_radius
        #delta = np.clip(desired_speed - linear_velo, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(desired_speed - linear_velo)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)

        #proses vehicular controls dari MLP
        mlp_steering = np.clip(psteer.cpu().data.numpy(), -1.0, 1.0)
        mlp_throttle = np.clip(pthrottle.cpu().data.numpy(), 0.0, self.config.max_throttle)

        #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
        act_pid_throttle = pid_throttle >= self.config.min_act_thrt
        act_mlp_throttle = mlp_throttle >= self.config.min_act_thrt
        if act_pid_throttle and act_mlp_throttle:
            act_pid_steering = np.abs(pid_steering) >= self.config.min_act_thrt
            act_mlp_steering = np.abs(mlp_steering) >= self.config.min_act_thrt
            if act_pid_steering and not act_mlp_steering:
                steering = pid_steering
            elif act_mlp_steering and not act_pid_steering:
                steering = mlp_steering
            else: #keduanya sama2 kurang dari threshold atau sama2 lebih dari threshold
                steering = self.config.cw_pid[0]*pid_steering + self.config.cw_mlp[0]*mlp_steering
            throttle = self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle
        elif act_pid_throttle and not act_mlp_throttle:
            steering = pid_steering
            throttle = pid_throttle
        elif act_mlp_throttle and not act_pid_throttle:
            steering = mlp_steering
            throttle = mlp_throttle
        else: # (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
            steering = 0.0 #dinetralkan
            throttle = 0.0
        

        # print(waypoints[2])
        
        metadata = {
            'control_option': self.config.ctrl_opt,
            'lr_velo': [float(angular_velo[0]), float(angular_velo[1])],
            'linear_velo' : float(linear_velo),
            'steering': float(steering),
            'throttle': float(throttle),
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1])],
            'pid_steering': float(pid_steering),
            'pid_throttle': float(pid_throttle),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1])],
            'mlp_steering': float(mlp_steering),
            'mlp_throttle': float(mlp_throttle),
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
        return float(steering), float(throttle), metadata



"""
#final decision
        if self.config.ctrl_opt == "one_of":
            #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
            steering = np.clip(self.config.cw_pid[0]*pid_steering + self.config.cw_mlp[0]*mlp_steering, -1.0, 1.0)
            throttle = np.clip(self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            if (pid_throttle >= self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                steering = pid_steering
                throttle = pid_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle >= self.config.min_act_thrt):
                steering = mlp_steering
                throttle = mlp_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                steering = 0.0 #dinetralkan
                throttle = 0.0
        elif self.config.ctrl_opt == "both_must":
            #opsi 2: vehicle jalan jika dan hanya jika kedua controller aktif. jika salah satu saja non aktif, maka vehicle berhenti
            steering = np.clip(self.config.cw_pid[0]*pid_steering + self.config.cw_mlp[0]*mlp_steering, -1.0, 1.0)
            throttle = np.clip(self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            if (pid_throttle < self.config.min_act_thrt) or (mlp_throttle < self.config.min_act_thrt):
                steering = 0.0 #dinetralkan
                throttle = 0.0
        elif self.config.ctrl_opt == "pid_only":
            #opsi 3: PID only
            steering = pid_steering
            throttle = pid_throttle
            #MLP full off
            # mlp_steering = 0.0
            # mlp_throttle = 0.0
            if pid_throttle < self.config.min_act_thrt:
                steering = 0.0 #dinetralkan
                throttle = 0.0
        elif self.config.ctrl_opt == "mlp_only":
            #opsi 4: MLP only
            steering = mlp_steering
            throttle = mlp_throttle
            #PID full off
            # pid_steering = 0.0
            # pid_throttle = 0.0
            if mlp_throttle < self.config.min_act_thrt:
                steering = 0.0 #dinetralkan
                throttle = 0.0
        else:
            sys.exit("ERROR, FALSE CONTROL OPTION")
"""