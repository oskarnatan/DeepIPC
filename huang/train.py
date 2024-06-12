import pandas as pd
import os
import cv2
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from torch import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import shutil
from model import huang
from data import WHILL_Data, swap_RGB2BGR
from config import GlobalConfig
from torch.utils.tensorboard import SummaryWriter
# import random
# random.seed(0)
# torch.manual_seed(0)


#buat ngecek GT SEG aja
def check_gt_seg(config, gt_seg):
    gt_seg = gt_seg.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((gt_seg.shape[2], gt_seg.shape[3], 3))
    #ambil tensor segmentationnya
    inx = np.argmax(gt_seg[0], axis=0)
    for cmap in config.SEG_CLASSES['colors']:
        cmap_id = config.SEG_CLASSES['colors'].index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap
    
    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = swap_RGB2BGR(imgx)
    cv2.imwrite(config.logdir+"/check_gt_seg.png", imgx) #cetak gt segmentation


#Class untuk penyimpanan dan perhitungan update loss
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

"""
#Class NN Module untuk Perhitungan BCE Dice Loss
def BCEDice(Yp, Yt, smooth=1e-7):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = Yp.view(-1)
    Yt = Yt.view(-1)
    #hitung BCE
    bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
    #hitung dice loss
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
    #kalkulasi lossnya
    bce_dice_loss = bce + dice_loss
    return bce_dice_loss
"""


#FUNGSI TRAINING
def train(data_loader, model, config, writer, cur_epoch, device, optimizer):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'ss_loss': AverageMeter(),
            'str_loss': AverageMeter(),
            'thr_loss': AverageMeter()}
    
    #masuk ke mode training, pytorch
    model.train()

    #visualisasi progress training dengan tqdm
    prog_bar = tqdm(total=len(data_loader))

    #training....
    total_batch = len(data_loader)
    batch_ke = 0
    for data in data_loader:
        cur_step = cur_epoch*total_batch + batch_ke

        #load IO dan pindah ke GPU
        rgbs = []
        segs = []
        deps = []

        for i in range(0, config.seq_len): #append data untuk input sequence
            rgbs.append(data['rgbs'][i].to(device, dtype=torch.float))
            segs.append(data['segs'][i].to(device, dtype=torch.float))
            deps.append(data['deps'][i].to(device, dtype=torch.float))
            # check_gt_seg(config, segs[-1])

        gt_steering = data['steering'].to(device, dtype=torch.float)
        gt_throttle = data['throttle'].to(device, dtype=torch.float)


        #forward pass
        # print(data['cmd'])
        pred_segs, steering, throttle = model(rgbs, deps, data['cmd'])#, , gt_velocity

        #compute loss
        loss_seg = 0
        for i in range(0, config.seq_len):
            loss_seg = loss_seg + F.binary_cross_entropy(pred_segs[i], segs[i], reduction='mean')
        loss_seg = loss_seg / config.seq_len #dirata-rata
        loss_str = F.mse_loss(steering, gt_steering)
        loss_thr = F.mse_loss(throttle, gt_throttle)
        total_loss = config.lw[0]*loss_seg + config.lw[1]*loss_str + config.lw[2]*loss_thr #pembobotan dari papernya gitu https://ieeexplore.ieee.org/document/9119447

        #backpro, kalkulasi gradient, dan optimasi
        optimizer.zero_grad()
        total_loss.backward() #ga usah retain graph
        optimizer.step() #dan update bobot2 pada network model

        #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
        score['total_loss'].update(total_loss.item())
        score['ss_loss'].update(loss_seg.item()) 
        score['str_loss'].update(loss_str.item()) 
        score['thr_loss'].update(loss_thr.item())


        #update visualisasi progress bar
        postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
                            ('t_ss_l', score['ss_loss'].avg),
                            ('t_str_l', score['str_loss'].avg),
                            ('t_thr_l', score['thr_loss'].avg)])
        
        #tambahkan ke summary writer
        writer.add_scalar('t_total_l', total_loss.item(), cur_step)
        writer.add_scalar('t_ss_l', loss_seg.item(), cur_step)
        writer.add_scalar('t_str_l', loss_str.item(), cur_step)
        writer.add_scalar('t_thr_l', loss_thr.item(), cur_step)

        prog_bar.set_postfix(postfix)
        prog_bar.update(1)
        batch_ke += 1
    prog_bar.close()    

    #return value
    return postfix


#FUNGSI VALIDATION
def validate(data_loader, model, config, writer, cur_epoch, device):
    #buat variabel untuk menyimpan kalkulasi loss, dan iou
    score = {'total_loss': AverageMeter(),
            'ss_loss': AverageMeter(),
            'str_loss': AverageMeter(),
            'thr_loss': AverageMeter()}
    
    #masuk ke mode eval, pytorch
    model.eval()

    with torch.no_grad():
        #visualisasi progress eval dengan tqdm
        prog_bar = tqdm(total=len(data_loader))

        #eval....
        total_batch = len(data_loader)
        batch_ke = 0
        for data in data_loader:
            cur_step = cur_epoch*total_batch + batch_ke

            #load IO dan pindah ke GPU
            rgbs = []
            segs = []
            deps = []

            for i in range(0, config.seq_len): #append data untuk input sequence
                rgbs.append(data['rgbs'][i].to(device, dtype=torch.float))
                segs.append(data['segs'][i].to(device, dtype=torch.float))
                deps.append(data['deps'][i].to(device, dtype=torch.float))
                # check_gt_seg(config, segs[-1])

            gt_steering = data['steering'].to(device, dtype=torch.float)
            gt_throttle = data['throttle'].to(device, dtype=torch.float)


            #forward pass
            pred_segs, steering, throttle = model(rgbs, deps, data['cmd'])#, , gt_velocity

            #compute loss
            loss_seg = 0
            for i in range(0, config.seq_len):
                loss_seg = loss_seg + F.binary_cross_entropy(pred_segs[i], segs[i], reduction='mean')
            loss_seg = loss_seg / config.seq_len #dirata-rata
            loss_str = F.mse_loss(steering, gt_steering)
            loss_thr = F.mse_loss(throttle, gt_throttle)
            total_loss = loss_seg + loss_str + loss_thr 

            #backpro, kalkulasi gradient, dan optimasi
            # optimizer.zero_grad()
            # total_loss.backward() #ga usah retain graph
            # optimizer.step() #dan update bobot2 pada network model

            #hitung rata-rata (avg) loss, dan metric untuk batch-batch yang telah diproses
            score['total_loss'].update(total_loss.item())
            score['ss_loss'].update(loss_seg.item()) 
            score['str_loss'].update(loss_str.item()) 
            score['thr_loss'].update(loss_thr.item())


            #update visualisasi progress bar
            postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
                                ('v_ss_l', score['ss_loss'].avg),
                                ('v_str_l', score['str_loss'].avg),
                                ('v_thr_l', score['thr_loss'].avg)])
            
            #tambahkan ke summary writer
            writer.add_scalar('v_total_l', total_loss.item(), cur_step)
            writer.add_scalar('v_ss_l', loss_seg.item(), cur_step)
            writer.add_scalar('v_str_l', loss_str.item(), cur_step)
            writer.add_scalar('v_thr_l', loss_thr.item(), cur_step)

            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
            batch_ke += 1
        prog_bar.close()    

    #return value
    return postfix



#MAIN FUNCTION
def main():
    # Load config
    config = GlobalConfig()
    

    #SET GPU YANG AKTIF
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id#visible_gpu #"0" "1" "0,1"

    #IMPORT MODEL UNTUK DITRAIN
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    model = huang(config, device).to(device, dtype=torch.float)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    #KONFIGURASI OPTIMIZER
    optima = optim.NAdam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    #BUAT DATA BATCH
    train_set = WHILL_Data(data_root=config.train_dir, conditions=config.train_conditions, config=config)
    val_set = WHILL_Data(data_root=config.val_dir, conditions=config.val_conditions, config=config)
    # train_set = WHILL_Data(root=config.train_data, config=config)
    # val_set = WHILL_Data(root=config.val_data, config=config)
    # print(len(train_set))
    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # print(len(dataloader_train))
    
    #cek retrain atau tidak
    if not os.path.exists(config.logdir+"/trainval_log.csv"):
        print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)
        curr_ep = 0
        lowest_score = float('inf')
        stop_count = config.init_stop_counter
    else:
        print('Continue training!!!!!!!!!!!!!!!!')
        print('Loading checkpoint from ' + config.logdir)
        #baca log history training sebelumnya
        log_trainval = pd.read_csv(config.logdir+"/trainval_log.csv")
        # replace variable2 ini
        # print(log_trainval['epoch'][-1:])
        curr_ep = int(log_trainval['epoch'][-1:]) + 1
        lowest_score = float(np.min(log_trainval['val_loss']))
        stop_count = int(log_trainval['stop_counter'][-1:])
        # Load checkpoint
        model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
        optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))

        #update direktori dan buat tempat penyimpanan baru
        config.logdir += "/retrain"
        os.makedirs(config.logdir, exist_ok=True)
        print('Created new retrain dir:', config.logdir)
    
    #copykan config file
    shutil.copyfile('config.py', config.logdir+'/config.py')

    #buat dictionary log untuk menyimpan training log di CSV
    log = OrderedDict([
            ('epoch', []),
            ('best_model', []),
            ('val_loss', []),
            ('val_ss_loss', []),
            ('val_str_loss', []),
            ('val_thr_loss', []),
            ('train_loss', []), 
            ('train_ss_loss', []),
            ('train_str_loss', []),
            ('train_thr_loss', []),
            ('lrate', []),
            ('stop_counter', []), 
            ('elapsed_time', []),
        ])
    writer = SummaryWriter(log_dir=config.logdir)
    
    #proses iterasi tiap epoch
    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        #cetak lr 
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        #training validation
        start_time = time.time() #waktu mulai
        train_log = train(dataloader_train, model, config, writer, epoch, device, optima)
        val_log = validate(dataloader_val, model, config, writer, epoch, device)
        #update learning rate untuk training process
        scheduler.step(val_log['v_total_l']) #parameter acuan reduce LR adalah val_total_metric
        elapsed_time = time.time() - start_time #hitung elapsedtime

        #simpan history training ke file csv
        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['t_total_l'])
        log['val_loss'].append(val_log['v_total_l'])
        log['train_ss_loss'].append(train_log['t_ss_l'])
        log['val_ss_loss'].append(val_log['v_ss_l'])
        log['train_str_loss'].append(train_log['t_str_l'])
        log['val_str_loss'].append(val_log['v_str_l'])
        log['train_thr_loss'].append(train_log['t_thr_l'])
        log['val_thr_loss'].append(val_log['v_thr_l'])
        log['elapsed_time'].append(elapsed_time)
        print('| t_total_l: %.4f | t_ss_l: %.4f | t_str_l: %.4f | t_thr_l: %.4f |' % (train_log['t_total_l'], train_log['t_ss_l'], train_log['t_str_l'], train_log['t_thr_l']))
        print('| v_total_l: %.4f | v_ss_l: %.4f | v_str_l: %.4f | v_thr_l: %.4f |' % (val_log['v_total_l'], val_log['v_ss_l'], val_log['v_str_l'], val_log['v_thr_l']))
        print('elapsed time: %.4f sec' % (elapsed_time))
        
        #save recent model dan optimizernya
        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))

        #save model best only
        if val_log['v_total_l'] < lowest_score:
            print("v_total_l: %.4f < lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model terbaik disave!")
            torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
            torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
            # torch.save(optima_lw.state_dict(), os.path.join(config.logdir, 'best_optim_lw.pth'))
            #v_total_l sekarang menjadi lowest_score
            lowest_score = val_log['v_total_l']
            #reset stop counter
            stop_count = config.init_stop_counter
            print("stop counter direset ke: ", stop_count)
            #catat sebagai best model
            log['best_model'].append("BEST")
        else:
            print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model tidak disave!")
            stop_count -= 1
            print("stop counter : ", stop_count)
            log['best_model'].append("")

        #update stop counter
        log['stop_counter'].append(stop_count)
        #paste ke csv file
        pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

        #kosongkan cuda chace
        torch.cuda.empty_cache()
        epoch += 1

        # early stopping jika stop counter sudah mencapai 0 dan early stop true
        if stop_count==0:
            print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
            break #loop
        

#RUN PROGRAM
if __name__ == "__main__":
    main()


