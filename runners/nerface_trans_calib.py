import pytorch_lightning as pl
#import wandb
#from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR


from models.nerface.base_function import canon_ray, meshgrid_xy, img2mse, mse2psnr, torch_normal_map, ray_sampling
from runners.run_utils import NerFACE
from utils.visualizer import cast_to_image, explicit_expr_control, explicit_pose_control
import utils.visualizer as vis

import numpy as np
from PIL import Image
import os
import scipy.io as sio

import json

class RunnerNerface_trans_calib(pl.LightningModule):

    def __init__(self, cfg, model_pack, start_epoch):
        super().__init__()
        self.cfg = cfg
        self.params = self.cfg.train_params
        self.start_epoch = start_epoch

        # Set-up model
        self.encode_position = model_pack['encode_position']
        self.encode_direction = model_pack['encode_direction']
        self.model_coarse = model_pack['model_coarse']
        self.model_fine = model_pack['model_fine']
        self.latent_codes = model_pack['latent_codes']
        self.delta_trans = model_pack['delta_trans']

        # Background
        if self.cfg.dataset in ['frame', 'frame_3dmm']:
            background = Image.open(os.path.join(cfg.dataset.basedir,'bg','00050.png'))
            background.thumbnail((cfg.dataset['H'], cfg.dataset['W']))
            self.background = torch.from_numpy(np.array(background).astype(np.float32))
        else:
            background = Image.new("RGB", (cfg.dataset['H'], cfg.dataset['W']), (255, 255, 255))
            background.thumbnail((cfg.dataset['H'], cfg.dataset['W']))
            self.background = torch.from_numpy(np.array(background).astype(np.float32)) / 255.

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        #torch.use_deterministic_algorithms(False)
    
    def training_step(self, batch, batch_idx):
        loss = {}

        # Fetch optimizer
        optimizer = self.optimizers()
        
        GT, pose, expression, data_idx = batch['image'].squeeze(), batch['pose'].squeeze(), batch['expression'].squeeze(), batch['data_idx']
        #TODO 이거 인덱스 2부터 불러옴 뭔가 이상함;;;
        #data 하나 단위로 뜯어옴
        ray_prob_map = batch['sampling_map'].cpu().numpy().squeeze()#TODO 512*512
        (H, W, focal) = batch['hwf']

        H, W, focal = int(H), int(W), focal.cpu().numpy().squeeze()
        device = GT.device

        latent_code = self.latent_codes[data_idx].to(device)
        delta_trans = self.delta_trans[data_idx].to(device)
        
        cam_o_full, r_d_full = canon_ray(H, W, focal, pose)#512x512x3
        

        grid = torch.stack(meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)), dim=-1)
        grid = grid.reshape((-1, 2))
        #512*512(262144)x2->flattening

        # Use importance sampling to sample mainly in the bbox with prob p
     
        #import pdb; pdb.set_trace()

        GT_color_sampled, bg_color_sampled, cam_o_sampled, r_d_sampled \
             = ray_sampling(grid, ray_prob_map, self.params.num_random_rays, 
                            GT, self.background, 
                            cam_o_full, r_d_full)
        #XXX calibrating cam_o
        cam_o_sampled = cam_o_sampled + delta_trans.expand_as(cam_o_sampled)
        #print(device, delata_)
        #print(ray_directions.shape,'ray directions input')
        # Model Forward
        rgb_coarse ,rgb_fine, _ = \
            NerFACE(self.model_coarse, self.model_fine,
                                 cam_o_sampled, r_d_sampled, self.cfg, mode="train",
                                 encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                 expressions = expression, background_prior=bg_color_sampled, latent_code = latent_code)
        #XXX torch.Size([2048, 3]) 이런식으로 뱉음
        #target_ray_values = GT_pixel

        loss['coarse'] = F.mse_loss(rgb_coarse[..., :3], GT_color_sampled[..., :3])
        loss['fine'] = F.mse_loss(rgb_fine[..., :3], GT_color_sampled[..., :3])
        loss['latent_code'] = 0.005 * torch.norm(latent_code)
        loss['delta_trans'] = 0.01 * torch.norm(delta_trans)
        
        loss['total'] = loss['coarse'] + loss['fine'] + loss['latent_code'] + loss['delta_trans']

        psnr = mse2psnr((loss['coarse'] + loss['fine']).item())

        # Backward
        optimizer.zero_grad()
        self.manual_backward(loss['total'])
        optimizer.step()
        
        # Log
        for k, v in loss.items():
            self.log(k, v.mean().detach().data.cpu())

        self.log('psnr_train', psnr)
        self.log('DELTA_trans',loss['delta_trans'])

        return {}


    def validation_step(self, batch, batch_idx):
        loss = {}
        outputs = {}

        with torch.no_grad():
            image, pose, expression, data_idx = batch['image'].squeeze(), batch['pose'].squeeze(), batch['expression'].squeeze(), batch['data_idx']
            (H, W, focal) = batch['hwf']
            H, W, focal = int(H), int(W), focal.cpu().numpy().squeeze()
            device = image.device

            latent_code = self.latent_codes[data_idx].to(device)
            delta_trans = self.delta_trans[data_idx].to(device)

            cam_o_full, r_d_full = canon_ray(H, W, focal, pose)
            cam_o_full = cam_o_full + delta_trans.expand_as(cam_o_full)

            # Model Forward
            rgb_fine, _,_, depth_fine, weight_fine = \
                NerFACE(self.model_coarse, self.model_fine,
                                    cam_o_full, r_d_full, self.cfg, mode="val",
                                    encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                    expressions = expression, background_prior=self.background.view(-1,3), latent_code = latent_code)
            target_ray_values = image
            
            loss['fine'] = 2 * img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
            
            

            # Visualization (GT Image / Pred Image / Depth Map / Weighted Sum)
            gt = cast_to_image(target_ray_values[..., :3])
            pred = cast_to_image(rgb_fine[..., :3])
            depth = np.stack((cast_to_image(depth_fine),) * 3, axis = -1)
            weight = np.stack((cast_to_image(weight_fine),) * 3, axis = -1)

            


            # Model Forward for Multiple Pose
            pose_list = explicit_pose_control(pose.cpu().numpy(), angle=5)

            output_control_list = []

            for pose in pose_list:
                rgb_fine, _,_, depth_fine, weight_fine = \
                    NerFACE(self.model_coarse, self.model_fine,
                                        cam_o_full, r_d_full, self.cfg, mode="val",
                                        encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                        expressions = expression, background_prior=self.background.view(-1,3), latent_code = latent_code)
                
                pred = cast_to_image(rgb_fine[..., :3])
                depth = np.stack((cast_to_image(depth_fine),) * 3, axis = -1)
                weight = np.stack((cast_to_image(weight_fine),) * 3, axis = -1)
                
                output_control = np.vstack((pred, depth, weight))
                output_control_list.append(output_control)

            outputs['psnr'] = mse2psnr(loss['fine'].item())
            outputs['gt_pred'] = np.hstack((gt, pred, depth, weight))
            outputs['pose_control'] = np.hstack(output_control_list)
        
        return outputs


    def training_step_end(self, outputs) -> None:
        # Scheduler update
        scheduler = self.lr_schedulers()
        scheduler.step()


    def validation_epoch_end(self, outputs) -> None:
        psnr, gt_pred, pose_control = [], [], []
        for output in outputs:
            if 'psnr' in output.keys():
                psnr.append(output['psnr'])
                gt_pred.append(output['gt_pred'])
                pose_control.append(output['pose_control'])

        # Log
        self.log('psnr_val', np.mean(psnr))
        self.logger.experiment.add_image('gt_pred1', np.array(gt_pred)[0].transpose((2,0,1)))
        self.logger.experiment.add_image('gt_pred2', np.array(gt_pred)[1].transpose((2,0,1)))
        self.logger.experiment.add_image('pose_control1', np.array(pose_control[0]).transpose((2,0,1)))
        self.logger.experiment.add_image('pose_control2', np.array(pose_control[1]).transpose((2,0,1)))


    def configure_optimizers(self):
        # Optimizer & Scheduler Check
        trainable_list = list(self.model_coarse.parameters()) + list(self.model_fine.parameters()) #+ list(self.latent_codes)
        trainable_list.append(self.latent_codes)
        trainable_list.append(self.delta_trans)
        
        optimizer = optim.Adam(trainable_list, lr=self.params['lr'], betas=(0.9, 0.999))
        scheduler = StepLR(optimizer, step_size=250000, gamma=0.1)
        return [optimizer], [scheduler]
    
    def export_optimized_translation(self):
        print(f'Saving optimized translation values!')
        t_dict=dict()
        t_dict['t']=[]
        for i in self.delta_trans:
            t = i.detach().numpy()
            t_dict['t'].append(t)
            sio.savemat('opted_trans.mat',t_dict)
    
    def custom_batch_test_pose_ctrl_opt(self, batch, normal = False):
        
        #assert option in ['pose','expr']
        
        #print('\n','***'*20,'\n',f'This is the {option} mode','\n','***'*20)
        
        image, pose, expression, data_idx = batch['image'].squeeze().cuda(), batch['pose'].squeeze().cuda(), batch['expression'].squeeze().cuda(), batch['data_idx']
        (H, W, focal) = batch['hwf']
        H, W, focal = int(H), int(W), focal.cpu().numpy().squeeze()
        device = image.device
        outputs = {}
        output_control_list=[]
        normal_map_imgs=[]
        with torch.no_grad():
            ray_origins, ray_directions = canon_ray(H, W, focal, pose)
            latent_code = self.latent_codes[data_idx].to(device)

            target_ray_values = image
            gt = cast_to_image(target_ray_values[..., :3])
            
            
            outputs['GT'] = gt
            
    
          
            
            #ctrl_val = 3 if option == "expr" else 2 #XXX 2 indicates the angle in Euler
            pose_list = explicit_pose_control(pose.cpu().numpy(),2)#XXX pose or expr
            
            for pose in pose_list:

                pose = torch.from_numpy(pose.astype(np.float32)).to(device)
                    
                ray_origins, ray_directions = canon_ray(H, W, focal, pose)
                latent_code = self.latent_codes[data_idx].to(device)
                
                rgb_fine, disp_fine, acc_fine, depth_fine, weight_fine = \
                NerFACE(self.model_coarse, self.model_fine,
                                    ray_origins, ray_directions, self.cfg, mode="val",
                                    encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                    expressions = expression, background_prior=self.background.view(-1,3), latent_code = latent_code)
            
                pred_temp = cast_to_image(rgb_fine[..., :3])
                depth_temp = np.stack((cast_to_image(depth_fine),) * 3, axis = -1)
                weight_temp = np.stack((cast_to_image(weight_fine),) * 3, axis = -1)
                
                output_control = np.vstack((pred_temp, depth_temp, weight_temp))
                output_control_list.append(output_control)
                
                #if normal == True:
                #    normal_map_imgs.append(torch_normal_map(disp_fine, focal, weight_fine).detach().cpu().numpy())
                    
            
            #normal_imgs = np.hstack(normal_map_imgs)
            #outputs['normal_imgs'] = normal_imgs
            
            ctrld_imgs = np.hstack(output_control_list)
            outputs['ctrld_imgs'] = ctrld_imgs
        
        return outputs
  
    def custom_batch_test(self, batch):
        tested_result = {}
        
        with torch.no_grad():
            GT, pose, expression = batch['image'].squeeze().cuda(), batch['pose'].squeeze().cuda(), batch['expression'].squeeze().cuda()
            (H, W, focal) = batch['hwf']
            H, W, focal = int(H), int(W), focal.cpu().numpy().squeeze()
            device = GT.device

            ray_origins, ray_directions = canon_ray(H, W, focal, pose)

            # Model Forward
            rgb_fine, _,_, depth_fine, weight_fine= \
                NerFACE(H, W, focal, self.model_coarse, self.model_fine,
                                    ray_origins, ray_directions, self.cfg, mode="val",
                                    encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                    expressions = expression, background_prior=self.background.view(-1,3).to(device), latent_code = torch.zeros(32).to(device))

        # Make output
        tested_result['depth_fine'] = depth_fine
        tested_result['weight_fine'] = weight_fine
        tested_result['gt'] = GT[..., :3]
        tested_result['pred'] = rgb_fine[..., :3]

        return tested_result
        
    def custom_batch_reenactment_test(self, batch, change_focal, driving_path):
        import random
        tested_result = {}
        drv_idx = random.choice(range(1000))
        with torch.no_grad():
            
            image, pose, expression = batch['image'].squeeze().cuda(), batch['pose'].squeeze().cuda(), batch['expression'].squeeze().cuda()
            (H, W, focal) = batch['hwf']
            H, W, focal = int(H), int(W), focal.cpu().numpy().squeeze()
            device = image.device
            
            with open(os.path.join(driving_path, "transforms_val.json"), "r") as fp:
                drv_json = json.load(fp)
            
            drv_data = drv_json['frames']
            
            drv_img_path = os.path.join(driving_path, drv_data[0]['file_path']+'.png')
            drv_img = Image.open(drv_img_path).resize((H, W))
            drv_img = (np.array(drv_img) / 255.0).astype(np.float32)
            drv_img = torch.from_numpy(drv_img).to(device)
            
            
            drv_focal = focal
            drv_pose = pose
            drv_expression = expression
                
            
            if change_focal == True:
                #drv_json['intrinsics']
                #drv_focal = np.array([-1481.96352, 1559.67488, 0.565694, 0.413902])
                drv_focal = np.array(drv_json['intrinsics'])
                drv_focal = torch.from_numpy(drv_focal)
                
            drv_pose = np.array(drv_data[drv_idx]['transform_matrix'])[:3, :4].astype(np.float32)
            drv_pose = torch.from_numpy(drv_pose).to(device)

            drv_expression = drv_data[drv_idx]['expression']
            drv_expression = np.array(drv_expression).astype(np.float32)
            drv_expression = torch.from_numpy(drv_expression).to(device)
            
            
            drv_ray_origins, drv_ray_directions = canon_ray(H, W, drv_focal, drv_pose)

            # Model Forward
            rgb_fine, _, _, _, _= \
                NerFACE(H, W, focal, self.model_coarse, self.model_fine,
                                    drv_ray_origins, drv_ray_directions, self.cfg, mode="val",
                                    encode_position_fn=self.encode_position, encode_direction_fn=self.encode_direction,
                                    expressions = drv_expression, background_prior=self.background.view(-1,3).to(device), latent_code = torch.zeros(32).to(device))
            

        # Make output
        #tested_result['depth_fine'] = depth_fine
        #tested_result['weight_fine'] = weight_fine
        tested_result['driving'] = drv_img[..., :3]
        tested_result['source'] = image[..., :3]
        tested_result['pred'] = rgb_fine[..., :3]

        return tested_result
