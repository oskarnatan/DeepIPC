import os

class GlobalConfig:
    gpu_id = '0'
    model = 'huang'
    logdir = 'log/'+model+'_mix_mix'
    init_stop_counter = 30

    batch_size = 8
    rp1_close = 4 #ganti rp jika mendekati ...meter
    bearing_bias = 7.5 #dalam derajat, pastikan sama dengan yang ada di plot_wprp.py
    n_buffer = 0 #buffer untuk MAF dalam second
    data_rate = 4 # 1 detik ada berapa data?
    #settingan zed camera
    max_depth = 39.99999 #dalam meter
    min_depth = 0.2 #dalam meter


	# Data
    seq_len = 1 # jumlah input seq
    pred_len = 0 # future waypoints predicted
    logdir = logdir+"_seq"+str(seq_len) #update direktori name

    # root_dir = '/home/aisl/WHILL/ros-whill-robot/main/dataset' 
    root_dir = '/home/aisl/OSKAR/WHILL/ros-whill-robot/main/dataset/dataset'
    train_dir = root_dir+'/train_routes'
    val_dir = root_dir+'/val_routes'
    test_dir = root_dir+'/test_routes'
    #train: sunny0,2,4,6,8,11 sunset1,3,5,7,9,10
    train_conditions = ['sunny', 'sunset'] #sunny route 2,4,6,11 ada sedikit adversarial
    #val: sunny1,3,5,7,9,10 sunset0,2,4,6,8,11 
    val_conditions = ['sunny', 'sunset'] #pokoknya kebalikannya train
    test_conditions = ['sunny3'] 
    # train_data, val_data, test_data = [], [], []
    # for weather in weathers:
    #     train_data.append(os.path.join(root_dir+'/train_routes', weather))
    #     val_data.append(os.path.join(root_dir+'/val_routes', weather))
    # test_weathers = ['cloudy']
    # for weather in test_weathers:
    #     test_data.append(os.path.join(root_dir+'/test_routes', weather))


    crop_roi = [512, 1024] #HxW
    scale = 2 #buat resizinig diawal load data

    lr = 0.0003 # learning rate initial NAdam
    lw = [2, 10, 1] #bobot loss seg str thr, dari papernya gitu https://ieeexplore.ieee.org/document/9119447

    n_cmd = 3 #jumlah command yang ada: 0 lurus, 1 kiri, 2 kanan
    max_throttle = 1.0 # upper limit on throttle signal value in dataset
    wheel_radius = 0.15#radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1 #minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    #BACA https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/core/evaluation/class_names.html#get_palette
    #HANYA ADA 19 CLASS?? + #tambahan 0,0,0 hitam untuk area kosong pada SDC nantinya
    SEG_CLASSES = {
        'colors'        :[[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],  
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
                        [0, 80, 100], [0, 0, 230], [119, 11, 32]],  
        'classes'       : ['None', 'road', 'sidewalk', 'building', 'wall',
                            'fence', 'pole', 'traffic light', 'traffic sign', 
                            'vegetation', 'terrain', 'sky', 'person', 
                            'rider', 'car', 'truck', 'bus',
                            'train', 'motorcycle', 'bicycle']
    }
    n_class = len(SEG_CLASSES['colors'])

    # n_fmap_r18 = [64, 64, 128, 256, 512]
    # n_fmap_r34 = [64, 64, 128, 256, 512] #sama dengan resnet18
    n_fmap_r50 = [64, 256, 512, 1024, 2048]
    # n_fmap_huang = [512, 128, 64, 16, n_class]#untuk seg decoder
    # stride_huang = [4, 2, 2, 2, 1] #yang terakhir point-wise
    control_huang = [256, 64, 16]
    #jangan lupa untuk mengganti model torchvision di init model.py

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
