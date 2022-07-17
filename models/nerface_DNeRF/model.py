import torch
import torch.nn as nn



class Mapping_Net(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=10):
        super(Mapping_Net, self).__init__()

        include_input_xyz = 3
        
        #XXX This is for Canonicalization
        self.dim_xyz_D2C = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        #XXX 63d

        self.dim_lc_geo = 0
   
        #XXX Inject to CANONICAL Space
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.dim_xyz_D2C + self.dim_lc_geo, 128))
        for i in range(1, 4):
            if i == 3:
                self.layers.append(torch.nn.Linear(self.dim_xyz_D2C + self.dim_lc_geo + 128, 128))
            else:
                self.layers.append(torch.nn.Linear(128, 128))
        self.layers.append(torch.nn.Linear(128, 3))
        
        self.relu = torch.nn.functional.relu

        
    def forward(self, x, latent_code_geo=None, **kwargs):
        #import pdb;pdb.set_trace()
        xyz = x
        #latent_code_geo = latent_code_geo.repeat(xyz.shape[0], 1) ##torch.Size([65536, 32])
        #canon_skip = torch.cat((xyz, latent_code_geo), dim=1)
        canon_skip = xyz
        
        x = canon_skip

        for i in range(5):
            if i==3:
                x = self.layers[i](torch.cat((canon_skip, x), -1))
            else:
                x = self.layers[i](x)
            x = self.relu(x)
        
        return x
class Mapping_Net_condition(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=10):
        super(Mapping_Net_condition, self).__init__()

        include_input_xyz = 3
        
        #XXX This is for Canonicalization
        self.dim_xyz_D2C = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        #XXX 63d

        self.dim_lc_deform = 32
   
        #XXX Inject to CANONICAL Space
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.dim_xyz_D2C + self.dim_lc_deform, 128))
        for i in range(1, 4):
            if i == 3:
                self.layers.append(torch.nn.Linear(self.dim_xyz_D2C + self.dim_lc_deform + 128, 128))
            else:
                self.layers.append(torch.nn.Linear(128, 128))
        self.layers.append(torch.nn.Linear(128, 3))
        
        self.relu = torch.nn.functional.relu

        
    def forward(self, x, latent_code_deform=None, **kwargs):
        #import pdb;pdb.set_trace()
        xyz = x
        latent_code_deform = latent_code_deform.repeat(xyz.shape[0], 1) ##torch.Size([65536, 32])
        canon_skip = torch.cat((xyz, latent_code_deform), dim=1)
        #canon_skip = xyz
        
        x = canon_skip

        for i in range(5):
            if i==3:
                x = self.layers[i](torch.cat((canon_skip, x), -1))
            else:
                x = self.layers[i](x)
            x = self.relu(x)
        
        return x

class Inverse_Mapping_Net(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=10, inv_lc = False):
        super(Inverse_Mapping_Net, self).__init__()

        include_input_xyz = 3
        
        #XXX This is for Canonicalization
        self.dim_xyz_C2D = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        #XXX 63d

        
        self.layers = torch.nn.ModuleList()
        self.dim_lc_geo = 0
        if inv_lc:
            self.dim_lc_geo = 32
   
        #XXX Inject to CANONICAL Space
        
        self.layers.append(torch.nn.Linear(self.dim_xyz_C2D + self.dim_lc_geo, 128))
        for i in range(1, 4):
            if i == 3:
                self.layers.append(torch.nn.Linear(self.dim_xyz_C2D + self.dim_lc_geo + 128, 128))
            else:
                self.layers.append(torch.nn.Linear(128, 128))
        self.layers.append(torch.nn.Linear(128, 3))
        
        self.relu = torch.nn.functional.relu

        
    def forward(self, x, latent_code_geo_inv=None, **kwargs):
        import pdb;pdb.set_trace()
        xyz = x
        if latent_code_geo_inv is None:
            canon_skip = x
        else:
            latent_code_geo = latent_code_geo.repeat(xyz.shape[0], 1) ##torch.Size([65536, 32])
            canon_skip = torch.cat((xyz, latent_code_geo), dim=1)
            
        x = canon_skip

        for i in range(5):
            if i==3:
                x = self.layers[i](torch.cat((canon_skip, x), -1))
            else:
                x = self.layers[i](x)
            x = self.relu(x)
        
        return x



'''
class Nerface(torch.nn.Module):
    def __init__(self, num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False):
        super(Nerface, self).__init__()

        include_input_xyz = 3
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76#TODO 이거 64로 바꿔야함 #XXX FLAME이라서 50으로

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 24
        self.dim_expression = include_expression # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = 32

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        #print(x.shape, expr.shape, latent_code.shape)
        #import pdb; pdb.set_trace()
        x = xyz
        latent_code = latent_code.repeat(xyz.shape[0], 1) ##torch.Size([65536, 32])
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)
'''


if __name__=="__main__":
    mapper = Inverse_Mapping_Net()
    mapper(torch.zeros([2000,63]))
    pass
