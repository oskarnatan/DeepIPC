import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
import cv2
from torch import torch
import yaml

from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from model import huang
from data import WHILL_Data, swap_RGB2BGR
from log.huang_mix_mix_seq1.config import GlobalConfig #pakai config.py yang dicopykan ke log
# import random
# random.seed(0)
# torch.manual_seed(0)


#Class untuk penyimpanan dan perhitungan update metric
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def IOU(Yp, Yt):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    output = Yp.view(-1) > 0.5 #maksudnya yang lebih dari 0.5 adalah true
    target = Yt.view(-1) > 0.5 #dan yang kurang dari 0.5 adalah false
    intersection = (output & target).sum() #irisan
    union = (output | target).sum() #union
    #rumus IoU
    iou = intersection / union
    return iou


def color_pred(config, seg):
    seg = seg.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((seg.shape[2], seg.shape[3], 3))

    #ambil tensor output segmentationnya
    inx = np.argmax(seg[0], axis=0)
    for cmap in config.SEG_CLASSES['colors']:
        cmap_id = config.SEG_CLASSES['colors'].index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap
    
    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = swap_RGB2BGR(imgx)
    return imgx

#FUNGSI test
def test(data_loader, model, config, device):
    #buat variabel untuk menyimpan kalkulasi metric, dan iou
    score = {'total_metric': AverageMeter(),
            'ss_metric': AverageMeter(),
            'str_metric': AverageMeter(),
            'thr_metric': AverageMeter()}

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
        ('batch', []),
        ('test_metric', []),
        ('test_ss_metric', []),
        ('test_str_metric', []),
        ('test_thr_metric', []),
        ('elapsed_time', []),
    ])

    #buat save direktori
    save_dir = config.logdir + "/offline_test/" 
    os.makedirs(save_dir, exist_ok=True)

    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress validasi dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #validasi....
        batch_ke = 1
        for data in data_loader:
            #load IO dan pindah ke GPU
            rgbs = []
            segs = []
            deps = []

            for i in range(0, config.seq_len): #append data untuk input sequence
                rgbs.append(data['rgbs'][i].to(device, dtype=torch.float))
                segs.append(data['segs'][i].to(device, dtype=torch.float))
                deps.append(data['deps'][i].to(device, dtype=torch.float))
                # check_gt_seg(config, segs[-1])
            rp1 = torch.stack(data['rp1'], dim=1).to(device, dtype=torch.float)
            rp2 = torch.stack(data['rp2'], dim=1).to(device, dtype=torch.float)
            gt_steering = data['steering'].to(device, dtype=torch.float)
            gt_throttle = data['throttle'].to(device, dtype=torch.float)

            #forward pass
            start_time = time.time() #waktu mulai
            pred_segs, steering, throttle = model(rgbs, deps, data['cmd'])
            elapsed_time = time.time() - start_time #hitung elapsedtime



            #compute metric
            metric_seg = 0
            for i in range(0, config.seq_len):
                metric_seg = metric_seg + IOU(pred_segs[i], segs[i])
            metric_seg = metric_seg / config.seq_len #dirata-rata
            metric_str = F.l1_loss(steering, gt_steering)
            metric_thr = F.l1_loss(throttle, gt_throttle)
            total_metric = (1-metric_seg) + metric_str + metric_thr

            #hitung rata-rata (avg) metric, dan metric untuk batch-batch yang telah diproses
            score['total_metric'].update(total_metric.item())
            score['ss_metric'].update(metric_seg.item()) 
            score['str_metric'].update(metric_str.item())
            score['thr_metric'].update(metric_thr.item())

            #update visualisasi progress bar
            postfix = OrderedDict([('te_total_m', score['total_metric'].avg),
                                ('te_ss_m', score['ss_metric'].avg),
                                ('te_str_m', score['str_metric'].avg),
                                ('te_thr_m', score['thr_metric'].avg)])

            #simpan history test ke file csv, ambil dari hasil kalkulasi metric langsung, jangan dari averagemeter
            log['batch'].append(batch_ke)
            log['test_metric'].append(total_metric.item())
            log['test_ss_metric'].append(metric_seg.item())
            log['test_str_metric'].append(metric_str.item())
            log['test_thr_metric'].append(metric_thr.item())
            log['elapsed_time'].append(elapsed_time)
            
            #save outputnya
            condition = data['condition'][-1]
            route = data['route'][-1]
            filename = data['filename'][-1]

            #paste ke csv file
            save_dir_log = save_dir+'outputs/'+condition
            os.makedirs(save_dir_log, exist_ok=True)
            pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)
            
            #save outputnya
            meta_output = dict()
            save_dir_seg = save_dir+'outputs/'+condition+'/'+route+'/pred_seg/'
            save_dir_meta = save_dir+'outputs/'+condition+'/'+route+'/meta/'
            os.makedirs(save_dir_seg, exist_ok=True)
            os.makedirs(save_dir_meta, exist_ok=True)

            meta_output['mlp_steering'] = float(steering.cpu().detach().numpy())
            meta_output['mlp_throttle'] = float(throttle.cpu().detach().numpy())
            meta_output['robot_bearing'] = data['bearing_robot'].item()
            meta_output['robot_pos'] = [data['lat_robot'].item(), data['lon_robot'].item()]
            meta_output['intervention'] = True
            rp1 = rp1[0].cpu().detach().numpy()
            rp2 = rp2[0].cpu().detach().numpy()
            meta_output['rp1'] = [float(rp1[0]), float(rp1[1])]
            meta_output['rp2'] = [float(rp2[0]), float(rp2[1])]
            meta_output['cmd'] = data['cmd'].item()
            meta_output['model_fps'] = float(1/elapsed_time)
            

            colored_seg = color_pred(config, pred_segs[-1]) #ambil yg terakhir sja
            cv2.imwrite(save_dir_seg+filename, colored_seg)
            with open(save_dir_meta+filename[:-3]+"yml", 'w') as dict_file:
                yaml.dump(meta_output, dict_file)

            batch_ke += 1  
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()

        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['test_metric'].append(np.mean(log['test_metric']))
        log['test_ss_metric'].append(np.mean(log['test_ss_metric']))
        log['test_str_metric'].append(np.mean(log['test_str_metric']))
        log['test_thr_metric'].append(np.mean(log['test_thr_metric']))
        log['elapsed_time'].append(np.mean(log['elapsed_time']))

        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['test_metric'].append(np.std(log['test_metric'][:-1]))
        log['test_ss_metric'].append(np.std(log['test_ss_metric'][:-1]))
        log['test_str_metric'].append(np.std(log['test_str_metric'][:-1]))
        log['test_thr_metric'].append(np.std(log['test_thr_metric'][:-1]))
        log['elapsed_time'].append(np.std(log['elapsed_time'][:-1]))

        #paste ke csv file
        pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)


    #return value
    return log



# Load config
config = GlobalConfig()

#SET GPU YANG AKTIF
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

#IMPORT MODEL dan load bobot
print("IMPORT ARSITEKTUR DL DAN COMPILE")
model = huang(config, device).to(device, dtype=torch.float)
model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

#BUAT DATA BATCH
test_set = WHILL_Data(data_root=config.test_dir, conditions=config.test_conditions, config=config)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) #BS selalu 1

#test
test_log = test(dataloader_test, model, config, device)


#kosongkan cuda chace
torch.cuda.empty_cache()

