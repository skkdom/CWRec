import os
import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


import time as Time
import ipdb


torch.pi = math.pi

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)




class diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config['timesteps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.c = config['c']
        self.embedding_dim = config['embedding_size']
        self.negsample_step = config['negsample_step']

        if config['beta_sche'] == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif config['beta_sche'] == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif config['beta_sche'] == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        
        # 两层
        # 三层
        self.denoise_model_uncon = nn.Sequential(
            nn.Linear(self.embedding_dim*2, self.embedding_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim*2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        self.denoise_model_uncon = self.denoise_model_uncon.cuda()

        # Multimodal Fusion:
        # 主干仍采用 self-attention，WFE 作为小残差辅助分支
        self.w_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_v)

        # WFE 高频缩放参数
        self.wfe_scale = nn.Parameter(torch.ones(2, self.embedding_dim))
        # WFE 残差门控：gate = eps_max * sigmoid(logit)，默认接近 0，训练自行决定是否打开
        # self.wfe_eps_max = config.get('wfe_eps_max', 0.15)
        self.wfe_eps_max = config['wfe_eps_max']
        self.wfe_gate_logit = nn.Parameter(torch.tensor(-4.0))
        # 消融/实验开关
        # self.wfe_ablation = config.get('wfe_ablation', False)   # True: 完全关闭 WFE，等价 baseline
        # self.wfe_in_sample = config.get('wfe_in_sample', True)  # False: 采样阶段只用主干，不用 WFE
        self.wfe_ablation = config['wfe_ablation']  # True: 完全关闭 WFE，等价 baseline
        self.wfe_in_sample = config['wfe_in_sample']  # False: 采样阶段只用主干，不用 WFE
        # 共享的一层 LayerNorm，用于 attention 与 WFE
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)



    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
#         ipdb.set_trace()
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t_start, v_start, t, noise=None, loss_type="l2"):
        #
#         ipdb.set_trace()
        if noise is None:
            noise_x = torch.randn_like(x_start)
#             noise_t = torch.randn_like(t_start)
#             noise_v = torch.randn_like(v_start)
            # noise = torch.randn_like(x_start) / 100
# 
        #
#         ipdb.set_trace()
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_x)  ## id
#         t_noisy = self.q_sample(x_start=t_start, t=t, noise=noise_t)  ##text
#         v_noisy = self.q_sample(x_start=v_start, t=t, noise=noise_v)  ##vision

        
        t_emb = self.get_timestep_embedding(t, self.embedding_dim).cuda()

        # 主干: 保持原始顺序 [ID, Text, Vision, Time]，用 self-attention 融合
        diff_main = torch.cat(
            [x_noisy.unsqueeze(1), t_start.unsqueeze(1), v_start.unsqueeze(1), t_emb.unsqueeze(1)],
            dim=1
        )
        predicted_x_main = self.selfAttention(diff_main)

        if self.wfe_ablation:
            predicted_x = predicted_x_main  # 消融：完全关闭 WFE，等价 baseline
        else:
            # 辅助分支: 有序伪序列 [Vision, Text, Noisy_ID, Time_Emb] 以配合 Haar 小波
            diff_wfe = torch.cat(
                [v_start.unsqueeze(1), t_start.unsqueeze(1), x_noisy.unsqueeze(1), t_emb.unsqueeze(1)],
                dim=1
            )
            predicted_x_wfe = self.wfe(diff_wfe)
            # 思路A: WFE 作为小残差校正，不直接替代主干
            gate = self.wfe_eps_max * torch.sigmoid(self.wfe_gate_logit)
            predicted_x = predicted_x_main + gate * (predicted_x_wfe - predicted_x_main)
        
        
#         ipdb.set_trace()
        #
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss_x = F.mse_loss(x_start, predicted_x)
#             loss_t = F.mse_loss(t_start, predicted_t)
#             loss_v = F.mse_loss(v_start, predicted_v)
            loss = loss_x 
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, denoise_model, x_t, t_t, v_t, t, t_index):
#         ipdb.set_trace()
        t = t.cuda()

        
        t_emb = self.get_timestep_embedding(t, self.embedding_dim).cuda()
        # 主干: 原始顺序 [ID, Text, Vision, Time]
        diff_main = torch.cat(
            [x_t.unsqueeze(1), t_t.unsqueeze(1), v_t.unsqueeze(1), t_emb.unsqueeze(1)],
            dim=1
        )
        x_start_main = self.selfAttention(diff_main)

        if self.wfe_ablation or not self.wfe_in_sample:
            x_start = x_start_main  # 消融 / 采样阶段关闭 WFE：只用主干
        else:
            # 辅助分支: [Vision, Text, Noisy_ID, Time_Emb] 供 WFE 使用
            diff_wfe = torch.cat(
                [v_t.unsqueeze(1), t_t.unsqueeze(1), x_t.unsqueeze(1), t_emb.unsqueeze(1)],
                dim=1
            )
            x_start_wfe = self.wfe(diff_wfe)
            gate = self.wfe_eps_max * torch.sigmoid(self.wfe_gate_logit)
            x_start = x_start_main + gate * (x_start_wfe - x_start_main)


        model_mean_x = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )


        if t_index == 0:
            return model_mean_x
        else:
#             ipdb.set_trace()
            posterior_variance_t_x = extract(self.posterior_variance, t, x_t.shape)
            noise_x = torch.randn_like(x_t)

            return model_mean_x + torch.sqrt(posterior_variance_t_x) * noise_x

    @torch.no_grad()
    def sample(self, denoise_model, x_start, t_start, v_start):
#         ipdb.set_trace()
       
        noise_x = torch.randn_like(x_start)
#             noise_t = torch.randn_like(t_start)
#             noise_v = torch.randn_like(v_start)
            # noise = torch.randn_like(x_start) / 100
# 
        #
#         ipdb.set_trace()
        t = torch.tensor([self.timesteps-1] * x_start.shape[0]).to(x_start.device)
#         x_t = self.q_sample(x_start, t)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_x)
        x_t = x_noisy
        t_t = t_start
        v_t = v_start
        x_half = x_t
        x_quarter = x_t
        x_three_quarter = x_t

        for n in reversed(range(0, self.timesteps)):
#             ipdb.set_trace()
            x_t = self.p_sample(denoise_model, x_t, t_t, v_t,
                              torch.full((x_t.shape[0],), n, dtype=torch.long), n)
            if n== int((self.timesteps -1) * 0.75):
                x_quarter = x_t
            if n== int((self.timesteps -1) * 0.5):
                x_half = x_t                
            if n== int((self.timesteps -1) * 0.25):
                x_three_quarter = x_t            
                
            
#         ipdb.set_trace()
        return x_start, x_t, x_quarter, x_half, x_three_quarter


    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        This matches the implementation in Denoising Diffusion Probabilistic Models:
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert len(timesteps.shape) == 1
#         ipdb.set_trace()
        
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def wfe(self, X):
        """Wavelet Feature Enhancement: DWT -> learnable enhance on high-freq -> IDWT.
        X: [B, 4, d], order [Vision, Text, Noisy_ID, Time_Emb].
        返回重构后的 ID 通道 (index=2)，作为辅助校正方向。
        """
        X = self.ln(X)
        # Haar DWT: A = (x0+x1)/2, D = (x0-x1)/2 for pairs (0,1) and (2,3)
        A0 = (X[:, 0] + X[:, 1]) / 2   # [B, d]
        A1 = (X[:, 2] + X[:, 3]) / 2
        D0 = (X[:, 0] - X[:, 1]) / 2
        D1 = (X[:, 2] - X[:, 3]) / 2
        D = torch.stack([D0, D1], dim=1)   # [B, 2, d]
        D_tilde = D * self.wfe_scale       # [B, 2, d]
        # IDWT
        X_recon = torch.stack([
            A0 + D_tilde[:, 0], A0 - D_tilde[:, 0],
            A1 + D_tilde[:, 1], A1 - D_tilde[:, 1]
        ], dim=1)   # [B, 4, d]
        # 取 ID 通道（去噪目标）作为 WFE 融合结果
        y = X_recon[:, 2, :]   # [B, d]
        return y

    def selfAttention(self, features):
        """原始多模态 self-attention，用作主干融合模块。features: [B, #modality, d]."""
        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)

        # [B, #modality, #modality]
        attn = q.mul(self.embedding_dim ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [B, #modality, d]
        # average pooling over modalities
        y = features.mean(dim=-2)  # [B, d]

        return y