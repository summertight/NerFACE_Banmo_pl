import os, glob
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import json


def read_mat(mat_file_path):
    """
    Mat file keys:
      - dict_keys(['id', 'exp', 'tex', 'angle', 'gamma', 'trans'])
    """
    mat_file = loadmat(mat_file_path)        
    exp = mat_file['exp'][0]
    angle = mat_file['angle'][0]
    trans = mat_file['trans'][0]
    
    return exp, angle, trans


def fov_to_intrinsic(fov, W):
    camera_angle_x = fov
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    return focal


def make_probmap_from_mask(sampling_map):
    sampling_map = sampling_map.numpy()
    probs = (1 / (sampling_map.sum())) * sampling_map

    return probs.reshape(-1)


def rotvec_to_rotmatrix(rotvec, trans):
    rotvec = rotvec[[2, 0, 1]] # [3DMM] pitch, yaw, roll (radian) -> [rotvec] roll, pitch, yaw
    rotmatrix = R.from_rotvec(rotvec)
    rotmatrix = rotmatrix.as_matrix()
    trans = trans.reshape((-1, 1))
    pose_matrix = np.hstack((rotmatrix, trans))

    return pose_matrix


class Vox_3dmm_Dataset_pervideo(Dataset):
    def __init__(self, basedir, mode='train', img_size=[256, 256], trans_set_0=False):
        super().__init__()
        videodir = '/home/nas4_user/jaeseonglee/sandbox/'##XXX This is fixed for beta-test

        video_list = os.listdir(videodir)

        self.trans_set_0 = trans_set_0
        self.basedir     = basedir
        self.mode        = mode
        self.H, self.W   = img_size[0], img_size[1]
        self.meta_chunk   = []
        self.image_chunk  = []
        self.video_frame_range = [0]
        # fov = 12
        for v_idx, v_name in enumerate(video_list):
            with open(os.path.join(videodir, v_name, "aux_data/flame_opted_params.json"), "r") as fp:
                meta = json.load(fp)
                self.meta_chunk.append(meta['frames'])
            unit_video = sorted(glob.glob(os.path.join(videodir, v_name,'*.jpg')))
            self.image_chunk.append(unit_video)
            self.video_frame_range.append(len(unit_video)+self.video_frame_range[v_idx])#XXX follow python convention
        #wwwimport pdb; pdb.set_trace()
        #with open(os.path.join(basedir, "aux_data/flame_opted_params.json"), "r") as fp:
        #    self.meta = json.load(fp)
        # Dataset Path
        #self.image_dir = os.path.join(basedir)
        #with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
        #   self.meta = json.load(fp)
        #self.data_list = self.meta['frames']

        #self.pose_dir = os.path.join(basedir.replace('images','3dmm'))
        #import pdb;pdb.set_trace()
        
        
        # Preprocess Intrinsic parameters
        #self.focal = fov_to_intrinsic(fov, self.W)
        self.focal = 1500
        self.intrinsics = np.array([-self.focal, self.focal, .5, .5])
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.intrinsics]

        

    def __len__(self):
        return self.video_frame_range[-1]-1

    def __getitem__(self, idx):
        #data_idx = idx
        video_label = 0
        for i in range(len(self.video_frame_range)-1):
            if self.video_frame_range[i]<=idx<self.video_frame_range[i+1]:
                invideo_idx = idx - self.video_frame_range[i]
                video_label = i
                image_path = self.image_chunk[i][invideo_idx]
                meta_data  = self.meta_chunk[i][invideo_idx]

        

        #image_path = os.path.join(self.image_list[idx])
        #pose_path = sorted(glob.glob(self.pose_dir + '/*.mat'))
        #import pdb; pdb.set_trace()
        # Load Image
        image = Image.open(image_path).resize((self.H, self.W))
        image = (np.array(image) / 255.0).astype(np.float32)

        # Load Pose Parameters (Expression, Rotation, Translation)
        #exp, angle, trans = read_mat(pose_path[idx])
        #exp = np.array(self.data_list[idx]['expression']).astype(np.float32)
        #import pdb; pdb.set_trace()
        #pose_ = np.array(self.data_list[idx]['transform_matrix'])[:3, :4].astype(np.float32)
        #trans = pose_[:,-1]

        #XXX this is from face2face
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        exp = np.array(meta_data['expression']).astype(np.float32)
        R_head_vec = np.array(meta_data['pose'])[0:3].astype(np.float32)
        trans = np.array(meta_data['world_mat'])[:,-1].astype(np.float32)

        pose = rotvec_to_rotmatrix(R_head_vec, trans).astype(np.float32)
        
        out = {}
        out['hwf'] = self.hwf
        out['image'], out['pose'], out['expression'], out['data_idx'], out['video_label'] = image, pose, exp, idx, video_label

        # Load Foreground Segmentation Mask (= Sampling Map)
        if self.mode == 'train':
            
            bbox = np.array([0.0, 1.0, 0.0, 1.0])
            bbox = np.array(meta_data['bbox'])/2+.5

            bbox[0:2], bbox[2:4] = self.H * bbox[0:2], self.W * bbox[2:4]
            bbox = torch.from_numpy(np.floor(bbox)).int()

            p = 0.9
            #import pdb;pdb.set_trace()
            sampling_map = np.zeros((self.H, self.W))
            sampling_map.fill(1 - p)
            sampling_map[bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
            sampling_map = (1 / sampling_map.sum()) * sampling_map
            sampling_map = sampling_map.reshape(-1)##(262144,)
            #XXX SUM == 1.0000000000000007

            

            out['sampling_map'] = sampling_map

        return out


if __name__ == "__main__":
    # Debug Dataset
    from torch.utils.data import DataLoader
    
    basedir = '/home/nas1_userB/sunghyun/Project/Sparse-Nerface/nerface_dataset/person_1'
    dataset = Vox_3dmm_Dataset_pervideo(basedir=basedir, mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        #import pdb; pdb.set_trace()

        if i == 3:
            break

