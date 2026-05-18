import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
import re
import os
import re
import torch
from PIL import Image
import numpy as np
from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler, CT_transformer, CT_transformer_3D
import torch
import torch.nn.functional as F
import pydicom
import os
import itertools



class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.neighbor_slices = model_params.neighbor_slices 
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        elif self.condition_key == 'CT_transformer':
            self.cond_stage_model = CT_transformer(self.neighbor_slices)
        else:
            raise NotImplementedError

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weights_init):
        self.denoise_fn.apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def get_parameters(self):
        params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        return params

    def forward(self, x, x_name, y, context=None):

      
        neighbor = self.load_neighbor_slices(x_name, y, self.neighbor_slices)
        neighbor_dest = self.load_neighbor_slices_dest(x_name, y, self.neighbor_slices)
        attmap = self.get_attmap(x_name, y).squeeze(1)
        print(attmap.shape, end = "bug\n\n\n\n\n\n\n\n\n\n\n")

        # test_cross = self.get_cond_transformer(x_cond_latent, attmap)
        # print(test_cross.shape)
        context,  loss_to_learn_from_attmap = self.get_cond_stage_context(neighbor, attmap, neighbor_dest)

        loss_cos_sim = 0
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        losses, x0_recon = self.p_losses(x, y, context, t)
        return losses + loss_to_learn_from_attmap*0.01, x0_recon, loss_cos_sim
    def get_cond_stage_context(self, x_cond_latent, attmap, neighbor_dest):
        context, attn_weights = self.cond_stage_model(x_cond_latent, attmap, neighbor_dest)
        return context, attn_weights
    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        ### 
        x0_recon = self.denoise_fn(x_t, y.clone(), timesteps=t, context=context) #mạng đoán ra luôn x0_recon, 
        
        objective_recon = x_t - x0_recon
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        # log_dict = {
        #     "loss": recloss,
        #     "x0_recon": x0_recon
        # }
        return recloss, x0_recon

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, y.clone(), timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            print(x_t.shape, end = "bug\n\n\n\n\n\n")
            objective_recon = self.denoise_fn(x_t, y, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, x_name, context=None, clip_denoised=True, sample_mid_step=False):
        # if self.condition_key == "nocond":
        #     context = None
        # else:
        #     context = y if context is None else context
        attmap = self.get_attmap(x_name, y).squeeze(1)
        neighbor = self.load_neighbor_slices(x_name, y, self.neighbor_slices)
        neighbor_dest = self.load_neighbor_slices_dest(x_name, y, self.neighbor_slices)
        
        context, attn_weights = self.get_cond_stage_context(neighbor, attmap, neighbor_dest)
        #nc_r
        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, x_name, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, x_name, context, clip_denoised, sample_mid_step)
    




    def load_neighbor_slices(self, x_name_list, x_cond_latent, k=5):
        device = x_cond_latent.device
        all_conditions = []

        patient_dir = 'path to attmap data'

        for x_path in x_name_list:

            x_path = x_path.replace("AC", "NC")  
            x_name_only = os.path.basename(x_path)


            m = re.search(r"slice(\d+)", x_name_only)
            if not m:
                continue
            center = int(m.group(1))

            neighbor_tensors = []

            for j in range(center - k, center + k + 1):
                if j < 0:
                    dummy = torch.full((1,64,64), -1.0, device=device)
                    neighbor_tensors.append(dummy)
                    continue

                slice_str = f"slice{j:03d}"
                new_name = re.sub(r"slice\d+", slice_str, x_name_only)


                found = False
                for split in ['train/A', 'test/A', 'val/A']:
                    new_path = os.path.join(patient_dir, split, new_name) + ".png"
                    if os.path.exists(new_path):
                        img = Image.open(new_path).convert("L")
                        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
                        arr = arr * 2 - 1                                # [-1,1]
                        tensor_img = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,H,W)
                        tensor_img = F.interpolate(tensor_img.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                        neighbor_tensors.append(tensor_img)
                        found = True
                        break

                if not found:
                    dummy = torch.full((1,64,64), -1.0, device=device)
                    neighbor_tensors.append(dummy)

            neighbor_stack = torch.stack(neighbor_tensors, dim=0) 
            all_conditions.append(neighbor_stack.unsqueeze(0))    

        if not all_conditions:
            return torch.empty(0)  

        all_tensor = torch.cat(all_conditions, dim=0)  # (batch, num_slices, C, H, W)
        return all_tensor.squeeze(2)  

    def load_neighbor_slices_dest(self, x_name_list, x_cond_latent, k=5):
        device = x_cond_latent.device
        all_conditions = []

        patient_dir = '/home/kienpt/dataset_remove_background_3_channel_png_for_attmap'

        for x_path in x_name_list:

            x_name_only = os.path.basename(x_path)


            m = re.search(r"slice(\d+)", x_name_only)
            if not m:
                continue
            center = int(m.group(1))

            neighbor_tensors = []

            for j in range(center - k, center + k + 1):
                if j < 0:
                    dummy = torch.full((1,64,64), -1.0, device=device)
                    neighbor_tensors.append(dummy)
                    continue

                slice_str = f"slice{j:03d}"
                new_name = re.sub(r"slice\d+", slice_str, x_name_only)


                found = False
                for split in ['train/B', 'test/B', 'val/B']:
                    new_path = os.path.join(patient_dir, split, new_name) + ".png"
                    if os.path.exists(new_path):
                        # load grayscale
                        img = Image.open(new_path).convert("L")
                        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
                        arr = arr * 2 - 1                                # [-1,1]
                        tensor_img = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,H,W)
                        tensor_img = F.interpolate(tensor_img.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

                        neighbor_tensors.append(tensor_img)
                        found = True
                        break

                if not found:
                    dummy = torch.full((1,64,64), -1.0, device=device)
                    neighbor_tensors.append(dummy)

            neighbor_stack = torch.stack(neighbor_tensors, dim=0)  # (num_slices, C, H, W)
            all_conditions.append(neighbor_stack.unsqueeze(0))     # batch dim

        if not all_conditions:
            return torch.empty(0) 

        all_tensor = torch.cat(all_conditions, dim=0) 
        return all_tensor.squeeze(2) 


    def load_attended_slices(self, x_name_list, x_cond_latent):

        device = x_cond_latent.device
        all_conditions = []

        attended_dir = "Path to attended"

        for x_path in x_name_list:
            x_name_only = os.path.basename(x_path)
            x_name_only = x_name_only.replace("AC", "NC")


            attended_name = f"{x_name_only}_attended.npy"
            attended_path = os.path.join(attended_dir, attended_name)

            if not os.path.exists(attended_path):
                print(attended_path)
                # fallback: dummy stack (17 slice)
                dummy = torch.full((17,  64, 64), -1.0, device=device)
                print("bug\n")
                all_conditions.append(dummy.unsqueeze(0))
                continue

            # ===== LOAD =====
            vol = np.load(attended_path)              # [S,H,W], [-1,1]
            vol = torch.from_numpy(vol).float()       # [S,H,W]
            vol = torch.clamp(vol, -1.0, 1.0)

            vol = vol.to(device)                      # [S,1,64,64]

            all_conditions.append(vol.unsqueeze(0))   # [1,S,1,64,64]

        return torch.cat(all_conditions, dim=0)       # [B,S,1,64,64]


    def get_attmap(self, x_name, x_cond_latent):
        import numpy as np
        import torch
        import os

        def find_attmap_file(folder_path, name):
            for filename in os.listdir(folder_path):
                if name in filename and filename.endswith(".npy"):
                    return os.path.join(folder_path, filename)
            raise FileNotFoundError(f"Can't find {name} in {folder_path}")

        root_path = "Path to output of CT-ViT"
        conditions = []

        for i in range(x_cond_latent.shape[0]):
            patient_dir = os.path.join(root_path, self.reconstruct_path_before_rest_or_stress(x_name[i] + ".png"))

            if "REST" in x_name[i]:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPREST")
            else:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPSTRESS")

            pixel_array = np.load(attmap_path).astype(np.float32)


            tensor = torch.tensor(pixel_array, dtype=torch.float32)



            conditions.append(tensor.unsqueeze(0))
        return torch.cat(conditions, dim=0).to(x_cond_latent.device)



    def reconstruct_path_before_rest_or_stress(self, filename):
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        cut_index = None
        for i, part in enumerate(parts):
            if part in ('REST', 'STRESS'):
                cut_index = i
                break

        if cut_index is None:
            raise ValueError("Can't find REST or STRESS in filename")

        path_before = '/'.join(parts[:cut_index])
        return path_before