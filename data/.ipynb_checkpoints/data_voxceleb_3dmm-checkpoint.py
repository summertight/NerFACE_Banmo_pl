import os, glob
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
import json

#python train.py --config-name=nerface.yaml gpu=[0]  
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


class Vox_3dmm_Dataset(Dataset):
    def __init__(self, basedir, mode='train', img_size=[256, 256], trans_set_0=False):
        super().__init__()
        basedir = '/data/private/Projects/dataset/WWZRPTh-irU#004992#005154'##XXX This is fixed for beta-test#194
        #/home/nas4_user/jaeseonglee/sandbox/WWZRPTh-irU#004992#005154 200
        #/home/nas4_user/jaeseonglee/sandbox/DPDPVItsdg8#000092#000429 404

        self.trans_set_0 = trans_set_0
        self.basedir = basedir
        self.mode = mode
        self.H, self.W = img_size[0], img_size[1]
        #fov = 12
        with open(os.path.join(basedir, "aux_data/deca_res.json"), "r") as fp:
            self.meta = json.load(fp)
        # Dataset Path
        #self.image_dir = os.path.join(basedir)
        #with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
        #   self.meta = json.load(fp)
        self.data_list = self.meta['frames']

        #self.pose_dir = os.path.join(basedir.replace('images','3dmm'))
        #import pdb;pdb.set_trace()
        self.image_list = sorted(glob.glob(os.path.join(self.basedir,'*.jpg')))
        
        # Preprocess Intrinsic parameters
        #self.focal = fov_to_intrinsic(fov, self.W)
        self.focal = 256*4.26*2#XXX from kakao convention
        self.intrinsics = np.array([-self.focal, self.focal, .5, .5])
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.intrinsics]

        self.seg_masks = sorted(glob.glob(os.path.join(basedir,'masked_imgs','*.png')))

        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data_idx = idx

        image_path = self.image_list[idx]
        seg_path = self.seg_masks[idx]
        
        seg_img = Image.open(seg_path).resize((self.H, self.W))
        seg_img = (np.array(seg_img) / 255.0).astype(np.float32)

        image = Image.open(image_path).resize((self.H, self.W))
        image = (np.array(image) / 255.0).astype(np.float32)

        masked_image = seg_img[...,None] * image
        
        #import pdb; pdb.set_trace()
        
        digital_idx = str(idx).zfill(4)
        exp = np.array(self.data_list[digital_idx]['exp']).astype(np.float32)
        #R_head_vec = np.array(self.data_list[digital_idx]['pose'])[3:6].astype(np.float32)
        #trans = np.array(self.data_list[digital_idx]['world_mat'])[:,-1].astype(np.float32)

        #pose = rotvec_to_rotmatrix(R_head_vec, trans).astype(np.float32)
        
        pose = np.array(self.data_list[digital_idx]['transform_matrix'])[:3,:4].astype(np.float32)
        #import pdb;pdb.set_trace()
        pose = np.hstack((pose[:3,:3],pose[:3,-1:]/4))# 4scaled!
        out = {}
        out['hwf'] = self.hwf
        #XXX masked_image가 GT임
        out['image'], out['pose'], out['expression'], out['data_idx'] = masked_image, pose, exp, data_idx

        # Load Foreground Segmentation Mask (= Sampling Map)
        if self.mode == 'train':
            #import pdb;pdb.set_trace()
            
            p = 0.9
            sampling_map = np.zeros((self.H, self.W))
            sampling_map.fill(1 - p)
            sampling_map += seg_img
            sampling_map = (1 / sampling_map.sum()) * sampling_map
            sampling_map = sampling_map.reshape(-1)
            out['sampling_map'] = sampling_map

        '''
        if self.mode == 'train':
            
            bbox = np.array([0.0, 1.0, 0.0, 1.0])
            bbox = np.array(self.data_list[idx]['bbox'])/2+.5

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
        '''
        return out


if __name__ == "__main__":
    # Debug Dataset
    from torch.utils.data import DataLoader
    
    basedir = '/home/nas1_userB/sunghyun/Project/Sparse-Nerface/nerface_dataset/person_1'
    dataset = Vox_3dmm_Dataset(basedir=basedir, mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        #import pdb; pdb.set_trace()

        if i == 3:
            break

