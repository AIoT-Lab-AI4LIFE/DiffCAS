import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import pydicom
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler, CT_transformer
from model.VQGAN.vqgan import VQModel
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.BrownianBridge.transformer.transformer import Transformer
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        elif self.condition_key == 'CT_transformer':
            self.cond_stage_model = CT_transformer()
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        elif self.condition_key == 'CT_transformer':
            print("get parameters to optimize: CT_transformer, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_name, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        ### Exp1 cross attenttion giữa CT và (NC ở latent space sau khi đi qua VQ GAN)
        print(x_latent.shape) # (8, 3, 64, 64)
        attmap = self.get_attmap(x_name, x_cond_latent).squeeze(1)

        # test_cross = self.get_cond_transformer(x_cond_latent, attmap)
        # print(test_cross.shape)
        context, attn_weight = self.get_cond_stage_context(x_cond, x_cond_latent, attmap)
        
        loss_cos_sim = 1 - F.cosine_similarity(x_cond_latent.reshape(x_cond_latent.shape[0], - 1), context.reshape(context.shape[0], -1), dim=1).mean()
        return super().forward(x_latent, x_cond_latent, context), loss_cos_sim

    def get_cond_stage_context(self, x_cond, x_cond_latent, attmap):
        if self.condition_key != 'CT_transformer':
            if self.cond_stage_model is not None:
                context = self.cond_stage_model(x_cond)
                if self.condition_key == 'first_stage':
                    context = context.detach()
            else:
                context = None
        else:
            context, attn_weights = self.cond_stage_model(x_cond_latent, attmap)
        return context, attn_weights
    
    def get_attmap(self, x_name, x_cond_latent):
        def find_attmap_file(folder_path, name):
            for filename in os.listdir(folder_path):
                if name in filename:
                    return os.path.join(folder_path, filename)
            raise FileNotFoundError("Không tìm thấy file chứa 'ATTMAPREST' trong folder.")

        # root_path = "/home/kienpt/dataset/unzip/Tim dicom"
        root_path = '/workdir/radish/kienpt/Tim dicom'
        conditions = []

        for i in range(x_cond_latent.shape[0]):
            patient_dir = os.path.join(root_path, self.reconstruct_path_before_rest_or_stress(x_name[i]+".png"))
            if 'REST' in x_name[i]:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPREST")
            else:
                attmap_path = find_attmap_file(patient_dir, "ATTMAPSTRESS")

            ds = pydicom.dcmread(attmap_path)

            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                pixel_array = ds.pixel_array.astype(np.float32)
            else:
                raise ValueError("File DICOM không chứa dữ liệu 3D (multi-frame).")

            tensor = torch.tensor(pixel_array, dtype=torch.float32)

            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)    # [0, 1]
                tensor = tensor * 2 - 1                              # [-1, 1]
            else:
                tensor = torch.zeros_like(tensor)
            tensor = torch.zeros_like(tensor)

            conditions.append(tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))

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
            raise ValueError("Không tìm thấy 'REST' hoặc 'STRESS' trong tên file.")

        path_before = '/'.join(parts[:cut_index])
        return path_before

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, x_name, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        attmap = self.get_attmap(x_name, x_cond_latent).squeeze(1)
        context, attn_weights = self.get_cond_stage_context(x_cond, x_cond_latent, attmap)

        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=context,
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=context,
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

