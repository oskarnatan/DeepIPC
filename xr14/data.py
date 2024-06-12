import os
import yaml
import cv2
# from PIL import Image, ImageFile
from collections import deque
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch 
from torch.utils.data import Dataset


class WHILL_Data(Dataset):

    def __init__(self, data_root, conditions, config):
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.data_rate = config.data_rate
        self.rp1_close = config.rp1_close

        self.condition = [] #buat offline test nantinya
        self.route = []
        self.filename = []
        self.rgb = []
        self.seg = []
        self.pt_cloud = []
        self.lon = []
        self.lat = []
        self.loc_x = []
        self.loc_y = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.loc_heading = []
        self.steering = []
        self.throttle = []
        self.velocity_l = []
        self.velocity_r = []
        
        for condition in conditions:
            sub_root = os.path.join(data_root, condition)
            preload_file = os.path.join(sub_root, 'xr14_seq'+str(self.seq_len)+'_pred'+str(self.pred_len)+'_rp1'+str(self.rp1_close)+'_maf'+str(self.config.n_buffer*self.data_rate)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_condition = []
                preload_route = []
                preload_filename = []
                preload_rgb = []
                preload_seg = []
                preload_pt_cloud = []
                preload_lon = []
                preload_lat = []
                preload_loc_x = []
                preload_loc_y = []
                preload_rp1_lon = []
                preload_rp1_lat = []
                preload_rp2_lon = []
                preload_rp2_lat = []
                preload_bearing = []
                preload_loc_heading = []
                preload_steering = []
                preload_throttle = []
                preload_velocity_l = []
                preload_velocity_r = []
                
                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                root_files.sort() #nanti sudah diacak oleh torch dataloader
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    
                    #load route list nya
                    with open(route_dir+"/"+route+"_routepoint_list.yml", 'r') as rp_listx:
                    # with open(route_dir+"/gmaps"+route[-2:]+"_routepoint_list.yml", 'r') as rp_listx:
                        rp_list = yaml.load(rp_listx)
                        #assign end point sebagai route terakhir
                        rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
                        rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])
                    
                    #list dan sort file, slah satu saja
                    files = os.listdir(route_dir+"/rgb/")
                    files.sort() #nanti sudah diacak oleh torch dataloader

                    # buat MAF mean avg filter
                    sin_angle_buff = deque()
                    if self.config.n_buffer!=0:
                        with open(route_dir+"/meta/"+files[0][:-3]+"yml", 'r') as first_metafile:
                            first_meta = yaml.load(first_metafile)
                        for _ in range(0, self.config.n_buffer*self.data_rate-1): #-1 karena nanti akan diappend dulu dengan data baru
                            sin_angle_buff.append(np.sin(np.radians(first_meta['wit_EKF_rpy'][2])))

                    for i in range(0, len(files)-(self.seq_len-1)-(self.pred_len*self.data_rate)): #kurangi sesuai dengan jumlah sequence dan wp yang akan diprediksi
                        #ini yang buat yg disequence kan
                        rgbs = []
                        segs = []
                        pt_clouds = []
                        loc_xs = []
                        loc_ys = []
                        loc_headings = []
                        
                        # read files sequentially (past and current frames)
                        for j in range(0, self.seq_len):
                            filename = files[i+j]
                            rgbs.append(route_dir+"/rgb/"+filename)
                            segs.append(route_dir+"/segmentation_GT/"+filename)
                            pt_clouds.append(route_dir+"/point_cloud/"+filename[:-3]+"npy")

                        #appendkan
                        preload_rgb.append(rgbs)
                        preload_seg.append(segs)
                        preload_pt_cloud.append(pt_clouds)

                        #metadata buat testing nantinya
                        preload_condition.append(condition)
                        preload_route.append(route)
                        preload_filename.append(filename)

                        # ambil local loc, heading, vehicular controls, gps loc, dan bearing pada seq terakhir saja (current)
                        with open(route_dir+"/meta/"+filename[:-3]+"yml", "r") as read_meta_current:
                            meta_current = yaml.load(read_meta_current)
                        loc_xs.append(meta_current['whill_local_position_xyz'][0])
                        loc_ys.append(meta_current['whill_local_position_xyz'][1])
                        loc_headings.append(np.radians(meta_current['whill_local_orientation_rpy'][2]))
                        preload_lon.append(meta_current['ublox_longitude'])
                        preload_lat.append(meta_current['ublox_latitude'])

                        #apply MAF ke bearing
                        angle_deg = meta_current['wit_EKF_rpy'][2]
                        sin_angle_buff.append(np.sin(np.radians(angle_deg)))
                        sin_angle_buff_mean = np.array(sin_angle_buff).mean()
                        #cek kuadran
                        if 0 < angle_deg <= 90: #Q1
                            angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif 90 < angle_deg <= 180: #Q2
                            angle_deg_maf = 180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -180 < angle_deg <= -90: #Q3 180 - 270
                            angle_deg_maf = -180 - np.degrees(np.arcsin(sin_angle_buff_mean))
                        elif -90 < angle_deg <= 0: #Q4 270 - 360
                            angle_deg_maf = np.degrees(np.arcsin(sin_angle_buff_mean))
                        sin_angle_buff.popleft() #hilangkan 1 untuk diisilagi dengan next data nantinya
                        bearing_robot_deg = angle_deg_maf+self.config.bearing_bias

                        if bearing_robot_deg > 180: #buat jadi -180 ke 0
                            bearing_robot_deg = bearing_robot_deg - 360
                        elif bearing_robot_deg < -180: #buat jadi 180 ke 0
                            bearing_robot_deg = bearing_robot_deg + 360
                        preload_bearing.append(np.radians(bearing_robot_deg))

                        #vehicular controls
                        preload_steering.append(meta_current['whill_steering'])
                        preload_throttle.append(meta_current['whill_throttle'])
                        preload_velocity_l.append(np.abs(meta_current['whill_LR_wheel_angular_velo'][0])) #kecepatan LR dibuat positif semua
                        preload_velocity_r.append(np.abs(meta_current['whill_LR_wheel_angular_velo'][1])) #kecepatan LR dibuat positif semua

                        
                        #assign next route lat lon
                        about_to_finish = False
                        for r in range(2): #ada 2 route point
                            next_lat = rp_list['route_point']['latitude'][r]
                            next_lon = rp_list['route_point']['longitude'][r]
                            dLat_m = (next_lat-meta_current['ublox_latitude']) * 40008000 / 360 #111320 #Y
                            dLon_m = (next_lon-meta_current['ublox_longitude']) * 40075000 * np.cos(np.radians(meta_current['ublox_latitude'])) / 360 #X
                            
                            if r==0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close and not about_to_finish: #jika jarak euclidian rp1 <= jarak min, hapus route dan loncat ke next route
                                if len(rp_list['route_point']['latitude']) > 2: #jika jumlah route list masih > 2
                                    rp_list['route_point']['latitude'].pop(0)
                                    rp_list['route_point']['longitude'].pop(0)
                                else: #berarti mendekati finish
                                    about_to_finish = True
                                    rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                                    rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]

                                next_lat = rp_list['route_point']['latitude'][r]
                                next_lon = rp_list['route_point']['longitude'][r]
                            
                            if r==0:
                                preload_rp1_lon.append(next_lon)
                                preload_rp1_lat.append(next_lat)
                            else: #r==1
                                preload_rp2_lon.append(next_lon)
                                preload_rp2_lat.append(next_lat)


                        # read files sequentially (future frames)
                        for k in range(1, self.pred_len+1):
                            filenamef = files[(i+self.seq_len-1) + (k*self.data_rate)] #future seconds, makanya dikali data rate
                            # meta
                            with open(route_dir+"/meta/"+filenamef[:-3]+"yml", "r") as read_meta_future:
                                meta_future = yaml.load(read_meta_future)
                            loc_xs.append(meta_future['whill_local_position_xyz'][0])
                            loc_ys.append(meta_future['whill_local_position_xyz'][1])
                            loc_headings.append(np.radians(meta_future['whill_local_orientation_rpy'][2]))

                        #append sisanya
                        preload_loc_x.append(loc_xs)
                        preload_loc_y.append(loc_ys)
                        preload_loc_heading.append(loc_headings)


                # dump ke npy
                preload_dict = {}
                preload_dict['condition'] = preload_condition
                preload_dict['route'] = preload_route
                preload_dict['filename'] = preload_filename
                preload_dict['rgb'] = preload_rgb
                preload_dict['seg'] = preload_seg
                preload_dict['pt_cloud'] = preload_pt_cloud
                preload_dict['lon'] = preload_lon
                preload_dict['lat'] = preload_lat
                preload_dict['loc_x'] = preload_loc_x
                preload_dict['loc_y'] = preload_loc_y
                preload_dict['rp1_lon'] = preload_rp1_lon
                preload_dict['rp1_lat'] = preload_rp1_lat
                preload_dict['rp2_lon'] = preload_rp2_lon
                preload_dict['rp2_lat'] = preload_rp2_lat
                preload_dict['bearing'] = preload_bearing
                preload_dict['loc_heading'] = preload_loc_heading
                preload_dict['steering'] = preload_steering
                preload_dict['throttle'] = preload_throttle
                preload_dict['velocity_l'] = preload_velocity_l
                preload_dict['velocity_r'] = preload_velocity_r
                np.save(preload_file, preload_dict)


            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.condition += preload_dict.item()['condition']
            self.route += preload_dict.item()['route']
            self.filename += preload_dict.item()['filename']
            self.rgb += preload_dict.item()['rgb']
            self.seg += preload_dict.item()['seg']
            self.pt_cloud += preload_dict.item()['pt_cloud']
            self.lon += preload_dict.item()['lon']
            self.lat += preload_dict.item()['lat']
            self.loc_x += preload_dict.item()['loc_x']
            self.loc_y += preload_dict.item()['loc_y']
            self.rp1_lon += preload_dict.item()['rp1_lon']
            self.rp1_lat += preload_dict.item()['rp1_lat']
            self.rp2_lon += preload_dict.item()['rp2_lon']
            self.rp2_lat += preload_dict.item()['rp2_lat']
            self.bearing += preload_dict.item()['bearing']
            self.loc_heading += preload_dict.item()['loc_heading']
            self.steering += preload_dict.item()['steering']
            self.throttle += preload_dict.item()['throttle']
            self.velocity_l += preload_dict.item()['velocity_l']
            self.velocity_r += preload_dict.item()['velocity_r']
            print("Preloading " + str(len(preload_dict.item()['rgb'])) + " sequences from " + preload_file)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        data = dict()
        #metadata buat testing nantinya
        data['condition'] = self.condition[index]
        data['route'] = self.route[index]
        data['filename'] = self.filename[index]

        data['rgbs'] = []
        data['segs'] = []
        data['pt_cloud_xs'] = []
        data['pt_cloud_zs'] = []
        seq_rgbs = self.rgb[index]
        seq_segs = self.seg[index]
        seq_pt_clouds = self.pt_cloud[index]
        seq_loc_xs = self.loc_x[index]
        seq_loc_ys = self.loc_y[index]
        seq_loc_headings = self.loc_heading[index]

        for i in range(0, self.seq_len):
            data['rgbs'].append(torch.from_numpy(np.array(crop_matrix(cv2.imread(seq_rgbs[i]), resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1))))
            data['segs'].append(torch.from_numpy(np.array(cls2one_hot(crop_matrix(cv2.imread(seq_segs[i]), resize=self.config.scale, crop=self.config.crop_roi), n_class=self.config.n_class))))

            pt_cloud = np.nan_to_num(crop_matrix(np.load(seq_pt_clouds[i])[:,:,0:3], resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1), nan=0.0, posinf=39.99999, neginf=0.2) #min_d, max_d, -max_d, ambil xyz-nya saja 0:3, baca https://www.stereolabs.com/docs/depth-sensing/depth-settings/
            data['pt_cloud_xs'].append(torch.from_numpy(np.array(pt_cloud[0:1,:,:])))
            data['pt_cloud_zs'].append(torch.from_numpy(np.array(pt_cloud[2:3,:,:])))


        #current ego robot position dan heading di index 0
        ego_loc_x = seq_loc_xs[0]
        ego_loc_y = seq_loc_ys[0]
        ego_loc_heading = seq_loc_headings[0]   

        # waypoint processing to local coordinates
        data['waypoints'] = [] #wp dalam local coordinate
        for j in range(1, self.pred_len+1):
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_loc_headings[j], seq_loc_xs[j], seq_loc_ys[j], np.pi/2-ego_loc_heading, ego_loc_x, ego_loc_y)
            data['waypoints'].append(tuple(local_waypoint[0,:2]))
      

        # convert rp1_lon, rp1_lat rp2_lon, rp2_lat ke local coordinates
        #komputasi dari global ke local
        #https://gamedev.stackexchange.com/questions/79765/how-do-i-convert-from-the-global-coordinate-space-to-a-local-space
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([[np.cos(bearing_robot), -np.sin(bearing_robot)],
                            [np.sin(bearing_robot),  np.cos(bearing_robot)]])
        dLat1_m = (self.rp1_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon1_m = (self.rp1_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        dLat2_m = (self.rp2_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon2_m = (self.rp2_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        # print("rp1_lat "+str(self.rp1_lat[index]))
        # print("rp2_lat "+str(self.rp2_lat[index]))
        # print("rp1_lon "+str(self.rp1_lon[index]))
        # print("rp2_lon "+str(self.rp2_lon[index]))

        #vehicular controls dan velocity jadikan satu LR
        data['steering'] = self.steering[index]
        data['throttle'] = self.throttle[index]
        data['lr_velo'] = tuple(np.array([self.velocity_l[index], self.velocity_r[index]]))

        #metadata buat testing nantinya
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data



def swap_RGB2BGR(matrix):
    red = matrix[:,:,0].copy()
    blue = matrix[:,:,2].copy()
    matrix[:,:,0] = blue
    matrix[:,:,2] = red
    return matrix



def crop_matrix(image, resize=1, D3=True, crop=[512, 1024]):
    
    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else: #buat matrix 2d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]

    #resize image
    WH_resized = (int(cropped_im.shape[1]/resize), int(cropped_im.shape[0]/resize))
    resized_image = cv2.resize(cropped_im, WH_resized, interpolation=cv2.INTER_NEAREST)

    return resized_image



def cls2one_hot(ss_gt, n_class):
    #inputnya adalah HWC baca cv2 secara biasanya, ambil salah satu channel saja
    ss_gt = np.transpose(ss_gt, (2,0,1)) #GANTI CHANNEL FIRST
    ss_gt = ss_gt[:1,:,:].reshape(ss_gt.shape[1], ss_gt.shape[2])
    result = (np.arange(n_class) == ss_gt[...,None]).astype(int) # jumlah class di cityscape pallete
    result = np.transpose(result, (2, 0, 1))   # (H, W, C) --> (C, H, W)
    # np.save("00009_ss.npy", result) #SUDAH BENAR!
    # print(result)
    # print(result.shape)
    return result


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out




"""
def scale_and_crop_image(image, scale=1, crop=[512, 1024]):

    #Scale and crop a PIL image, returning a channels-first numpy array.

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_y = height//2 - crop[0]//2
    start_x = width//2 - crop[1]//2
    cropped_image = image[start_y:start_y+crop[0], start_x:start_x+crop[1]]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image
"""