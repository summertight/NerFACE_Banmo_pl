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


class NerFACE_IMavatar_Dataset(Dataset):
    def __init__(self, basedir, mode='train', img_size=[512, 512], trans_set_0=False):
        super().__init__()
        self.trans_set_0 = trans_set_0
        self.basedir = basedir
        self.mode = mode
        self.H, self.W = img_size[0], img_size[1]
        fov = 12
        aux_path = '/home/nas1_userA/jaeseonglee/nerface_data_preprocess/person1'
        # Dataset Path
        #self.image_dir = os.path.join(basedir)
        with open(os.path.join(aux_path, mode, "person1_flame_params.json"), "r") as fp:
            self.meta = json.load(fp)
        self.data_list = self.meta['frames']

        #self.pose_dir = os.path.join(basedir, mode, '3dmms')
        
        self.image_list = sorted(glob.glob(os.path.join(self.basedir,mode,'*.png')))
        #import pdb;pdb.set_trace()
        # Preprocess Intrinsic parameters
        #self.focal = fov_to_intrinsic(fov, self.W)
        
        
        self.intrinsics = np.array([-2223.21152, 2422.76352, 0.502588, 0.48830700000000005])
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.intrinsics]

        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data_idx = idx
        image_path = os.path.join(self.image_list[idx])
        #pose_path = sorted(glob.glob(self.pose_dir + '/*.mat'))
        #import pdb; pdb.set_trace()
        # Load Image
        image = Image.open(image_path).resize((self.H, self.W))
        image = (np.array(image) / 255.0).astype(np.float32)

        # Load Pose Parameters (Expression, Rotation, Translation)
        #exp, angle, trans = read_mat(pose_path[idx])
        exp = np.array(self.data_list[idx]['expression']).astype(np.float32)
        R_head_vec = np.array(self.data_list[idx]['pose'])[3:6].astype(np.float32)
        trans = np.array(self.data_list[idx]['world_mat'])[:,-1].astype(np.float32)
        
        #exp = np.array(self.data_list[idx]['expression']).astype(np.float32)

        #pose_ = np.array(self.data_list[idx]['transform_matrix'])[:3, :4].astype(np.float32)
        #trans = pose_[:,-1]
        '''
        (Pdb) pose.shape
        (3, 4)
        (Pdb) exp.shape
        (50,)
        (Pdb) trans.shape
        (3,)
        (Pdb) angle.shape
        (3,)
        '''
        #XXX this is from face2face
        #import pdb;pdb.set_trace()
        
        R_head_vec= -R_head_vec#XXX For IMAvatar Logic..?? IDK yet.6/19
        
        #if self.trans_set_0:
        #    trans = np.zeros_like(trans)

        pose = rotvec_to_rotmatrix(R_head_vec, trans).astype(np.float32)
        
        out = {}
        out['hwf'] = self.hwf
        out['image'], out['pose'], out['expression'], out['data_idx'] = image, pose, exp, data_idx

        # Load Foreground Segmentation Mask (= Sampling Map)
        if self.mode == 'train':
            if 'bbox' in self.data_list[idx].keys():
                bbox = np.array(self.data_list[idx]['bbox'])
            if True:
                bbox = np.array([0.0, 1.0, 0.0, 1.0])

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
    dataset = NerFACE_IMavatar_Dataset(basedir=basedir, mode='val')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        #import pdb; pdb.set_trace()

        if i == 3:
            break

