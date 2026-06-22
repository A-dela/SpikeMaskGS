import torch
import tinycudann as tcnn
import torch.nn as nn

class SpaceTimeHashingField(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max, n_levels,n_features_per_level,base_resolution,n_neurons, 
                  hashmap_size, activation ):
        super(SpaceTimeHashingField, self).__init__()

        # tcnn : INGP implementation
        self.enc_model = tcnn.NetworkWithInputEncoding(
            n_input_dims = 4 ,
            n_output_dims= n_neurons ,    # as same as 4dgs
            encoding_config={
                "otype": "HashGrid" ,
                "n_levels": n_levels ,
                "n_features_per_level": n_features_per_level ,  # 2
                "log2_hashmap_size": hashmap_size ,  # 19
                "base_resolution": base_resolution ,
                "per_level_scale": 2.0 ,
            },

            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": activation,  
                "n_neurons": n_neurons,
                "n_hidden_layers": 1 ,
            },
        )
        
        self.xyz_deform = tcnn.Network(
            n_input_dims = n_neurons,
            n_output_dims = 3,
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 2 ,
            },
        )

        self.scales_deform = tcnn.Network(
            n_input_dims= n_neurons,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 2 ,
            },
        )

        self.rotations_deform = tcnn.Network(
            n_input_dims= n_neurons,
            n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 2 ,
            },
        )

        self.opacity_deform = tcnn.Network(
            n_input_dims= n_neurons,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 2 ,
            },
        )
        
        self.shs_deform = tcnn.Network(
            n_input_dims= n_neurons,
            n_output_dims= 16 * 3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": n_neurons,
                "n_hidden_layers": 2 ,
            },
        )


        # self.xyz_bound_min = xyz_bound_min
        # self.xyz_bound_max = xyz_bound_max
        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)
        

    def dump(self, path):
        torch.save(self.state_dict(),path)
        

    def get_contracted_xyz(self, xyz): 
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz

    def quaternion_multiply(a, b):
        """
        Multiply two sets of quaternions.
        
        Parameters:
        a (Tensor): A tensor containing N quaternions, shape = [N, 4]
        b (Tensor): A tensor containing N quaternions, shape = [N, 4]
        
        Returns:
        Tensor: A tensor containing the product of the input quaternions, shape = [N, 4]
        """
        a_norm=torch.nn.functional.normalize(a)
        b_norm=torch.nn.functional.normalize(b)
        w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
        w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

        return torch.stack([w, x, y, z], dim=1)

    def forward(self, xyz:torch.Tensor, scales, rotations, opacity, shs, time):

        contracted_xyz=self.get_contracted_xyz(xyz)                          # Shape: [N, 3]
        
        mask = ( contracted_xyz >= 0 ) & ( contracted_xyz <= 1 )
        mask = mask.all(dim=1)
        hash_inputs = torch.cat([contracted_xyz[mask], time[mask]],dim=-1)


        hidden =  self.enc_model(hash_inputs)  # torch.tensor( self.enc_model(hash_inputs) ,dtype= torch.float32) # [N, 128]
        masked_d_xyz = self.xyz_deform(hidden)
        masked_d_rot = self.rotations_deform(hidden)
        masked_d_scales = self.scales_deform(hidden)
        masked_d_opacity = self.opacity_deform(hidden)
        masked_d_shs = self.shs_deform(hidden).reshape([hidden.shape[0],16,3])


        new_xyz = torch.zeros_like(xyz,  device="cuda")
        new_rot = torch.zeros_like(rotations,  device="cuda")
        new_scales = torch.zeros_like(scales,  device="cuda")
        new_opacity = torch.zeros_like(opacity, device="cuda")
        new_shs = torch.zeros_like(shs,   device="cuda")



        new_xyz[mask], new_rot[mask], new_scales[mask], new_opacity[mask], new_shs[mask] = \
             masked_d_xyz + xyz[mask], masked_d_rot + rotations[mask], masked_d_scales + scales[mask],  masked_d_opacity + opacity[mask],  masked_d_shs + shs[mask]
        
        new_xyz[~mask], new_rot[~mask], new_scales[~mask], new_opacity[~mask], new_shs[~mask] = \
            xyz[~mask], rotations[~mask], scales[~mask], opacity[~mask], shs[~mask]
        
        return  new_xyz, new_scales,  new_rot, new_opacity, new_shs
    
    def get_params (self):

        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters(): # enc_model.para
            if  "enc_model" not in name:
                parameter_list.append(param)
        return parameter_list


    def get_hash_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "enc_model" in name:
                parameter_list.append(param)  
        return parameter_list

        
        
