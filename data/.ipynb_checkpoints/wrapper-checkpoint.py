from data.data_frame import Frame_Dataset
#from data.data_frame_3dmm import Frame_3dmm_Dataset
from data.data_voxceleb_3dmm import Vox_3dmm_Dataset
from data.data_nerface_IMavatar import NerFACE_IMavatar_Dataset
from data.data_frame_test import Frame_Dataset_test
from data.data_voxceleb_3dmm_pervideo import Vox_3dmm_Dataset_pervideo
from data.data_frame_kakao import Frame_kakao_Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DatasetModule(pl.LightningDataModule):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.dataset
        self.dataset_name = cfg.dataset['name']
        try:
            self.trans_set_0 = cfg.train_params.trans_set_0 
        except:
            pass
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = None
        self.valid_set = None
        self.test_set = None

        if self.dataset_name == 'frame':
            if stage == 'test':
                self.test_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='test')
            else:
                self.train_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='train', data_size=self.cfg['data_size'])
                self.valid_set = Frame_Dataset(basedir=self.cfg['basedir'], mode='val')
        #if self.dataset_name == 'frame+3dmm':
        #    if stage == 'test':
        #        self.test_set = Frame_3dmm_Dataset(basedir=self.cfg['basedir'], mode='test', trans_set_0=self.trans_set_0)
        #    else:
        #        self.train_set = Frame_3dmm_Dataset(basedir=self.cfg['basedir'], mode='train', trans_set_0=self.trans_set_0)
        #        self.valid_set = Frame_3dmm_Dataset(basedir=self.cfg['basedir'], mode='val', trans_set_0=self.trans_set_0)
        if self.dataset_name == 'voxceleb':
            if stage == 'test':
                self.test_set = Vox_3dmm_Dataset(basedir=self.cfg['basedir'], mode='test', trans_set_0=self.trans_set_0)
            else:
                self.train_set = Vox_3dmm_Dataset(basedir=self.cfg['basedir'], mode='train', trans_set_0=self.trans_set_0)
                self.valid_set = Vox_3dmm_Dataset(basedir=self.cfg['basedir'], mode='val', trans_set_0=self.trans_set_0)
        if self.dataset_name == 'nerface+IM':
            if stage == 'test':
                self.test_set = NerFACE_IMavatar_Dataset(basedir=self.cfg['basedir'], mode='test')
            else:
                self.train_set = NerFACE_IMavatar_Dataset(basedir=self.cfg['basedir'], mode='train')
                self.valid_set = NerFACE_IMavatar_Dataset(basedir=self.cfg['basedir'], mode='val')
        if self.dataset_name == 'frame_test':
            if stage == 'test':
                self.test_set = Frame_Dataset_test(basedir=self.cfg['basedir'], mode='test')
            else:
                self.train_set = Frame_Dataset_test(basedir=self.cfg['basedir'], mode='train')
                self.valid_set = Frame_Dataset_test(basedir=self.cfg['basedir'], mode='val')
        if self.dataset_name == 'vox_pervideo':
            if stage == 'test':
                self.test_set = Vox_3dmm_Dataset_pervideo(basedir=self.cfg['basedir'], mode='test', trans_set_0=self.trans_set_0)
            else:
                self.train_set = Vox_3dmm_Dataset_pervideo(basedir=self.cfg['basedir'], mode='train', trans_set_0=self.trans_set_0)
                self.valid_set = Vox_3dmm_Dataset_pervideo(basedir=self.cfg['basedir'], mode='val', trans_set_0=self.trans_set_0)
        if self.dataset_name == 'frame_kakao':
            if stage == 'test':
                self.test_set = Frame_kakao_Dataset(basedir=self.cfg['basedir'], mode='test')
            else:
                self.train_set = Frame_kakao_Dataset(basedir=self.cfg['basedir'], mode='train')
                self.valid_set = Frame_kakao_Dataset(basedir=self.cfg['basedir'], mode='val')

    
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=True, 
                          drop_last=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          num_workers=self.cfg['num_workers'],
                          batch_size=self.cfg['batch_size'],
                          shuffle=False,
                          drop_last=True)