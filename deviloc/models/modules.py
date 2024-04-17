import torch
import torch.nn as nn
from timm.models.layers import DropPath
from kornia.geometry.calibration import undistort_points

from deviloc.utils.geometry import pts_cam2world, pts_world2cam


class Pts3DInfer(nn.Module):
    def __init__(self, config, dim_feat):
        super(Pts3DInfer, self).__init__()
        self.config = config

        self.d_point = config["dim_point"]
        self.nhead = config["nhead"]
        self.n_pts2d_layers = config["n_pts2d_layers"]
        self.n_pts3d_layers = config["n_pts3d_layers"]
        self.n_propa_layers = config["n_propa_layers"]
        self.d_model = config["dim_model"]
        self.n_infer_layers = config["n_infer_layers"]

        self.pts2d_scale = nn.Parameter(10 * torch.ones(1, 1, 2))
        self.pts2d_emb = MLP(2, self.d_point, self.d_point)

        self.depth_scale = nn.Parameter(10 * torch.ones(1, 1, 1))
        self.depth_emb = MLP(1, self.d_point, self.d_point)
        self.pts3d_emb = nn.Linear(self.d_point * 2, self.d_point, bias=False)
        self.depth_enc = AttentionLayer(self.d_point, self.nhead, attn_drop=0.0, layer_scale=1.0)

        self.propa_layers = nn.ModuleList([AttentionLayer(self.d_point, self.nhead, attn_drop=0.0, layer_scale=1.0)
                                           for _ in range(self.n_propa_layers)])
        
        self.merge2d_layers = MLP(dim_feat + self.d_point, self.d_model, self.d_model)
        self.merge3d_layers = MLP(dim_feat + self.d_point, self.d_model, self.d_model)

        self.infer_layers = nn.ModuleList([AttentionLayer(self.d_model, self.nhead, attn_drop=0.0, layer_scale=1.0)
                                           for _ in range(self.n_infer_layers * 2)])

        self.out_coords = nn.Sequential(MLP(self.d_model, self.d_model, self.d_model),
                                        nn.LayerNorm(self.d_model),
                                        MLP(self.d_model, self.d_model, 1)
                                        )
        self.out_conf = nn.Sequential(MLP(self.d_model + 1, self.d_model * 2, self.d_model),
                                      nn.LayerNorm(self.d_model),
                                      MLP(self.d_model, self.d_model, 1),
                                      nn.Sigmoid())

    def get_depth_scale(self, pts3d, mask=None):
        z = pts3d[..., -1].clone()
        if mask is None:
            mask = torch.ones_like(z).bool()
        max_z = torch.max(z * mask, dim=1, keepdim=True)[0]
        z[~mask] = 1e9
        min_z = torch.min(z, dim=1, keepdim=True)[0]
        std_z = (max_z - min_z) + 1e-8
        return min_z.unsqueeze(-1), std_z.unsqueeze(-1)
    
    def get_norm_kpts2d(self, kpts2d, K, dist_coeffs=None):
        if dist_coeffs is not None:
            kpts2d = undistort_points(kpts2d, K, dist_coeffs)
        
        kpts2d_homo = torch.cat((kpts2d, torch.ones_like(kpts2d[..., [0]])), dim=-1)
        kpts2d_homo = kpts2d_homo @ K.inverse().transpose(1, 2)
        
        return kpts2d_homo
        
    def forward(self, obs_pts3d, obs_feats, kpts2d, kpts2d_feat, cam_params):
        """
        feat: [B, C, H, W]
        obs_pts3d: [B, Npts3d, 3]
        kpts2d: [B, Nkpts2d, 2]
        mask_kpts2d: 
        """
        
        K, T, dist_coeffs = cam_params

        obs_pts3d_cam = pts_world2cam(obs_pts3d, T)
        mean_depth, std_depth = self.get_depth_scale(obs_pts3d_cam)

        kpts2d_homo = self.get_norm_kpts2d(kpts2d, K, dist_coeffs)
        kpts2d = kpts2d_homo[..., :2] * self.pts2d_scale

        kpts2d = self.pts2d_emb(kpts2d)

        obs_pts2d = obs_pts3d_cam[:, :, :2] / (obs_pts3d_cam[:, :, [2]] + 1e-8) * self.pts2d_scale
        obs_depth = (obs_pts3d_cam[:, :, [2]] - mean_depth) / std_depth 

        opts2d = self.pts2d_emb(obs_pts2d)
        odepth = self.depth_emb(obs_depth * self.depth_scale)
        odepth = self.depth_enc(odepth, odepth)

        opts3d = self.pts3d_emb(torch.cat((opts2d, odepth), dim=-1))

        for layer in self.propa_layers:
            kpts2d = layer(kpts2d, opts3d)
        
        kpts_feat = self.merge2d_layers(torch.cat((kpts2d, kpts2d_feat), dim=-1))
        opts_feat = self.merge3d_layers(torch.cat((opts3d, obs_feats), dim=-1))

        for idx in range(len(self.infer_layers) // 2):
            opts_feat = self.infer_layers[idx * 2](opts_feat, opts_feat)
            kpts_feat = self.infer_layers[idx * 2 + 1](kpts_feat, opts_feat)

        kpts_depth = self.out_coords(kpts_feat) * std_depth + mean_depth
        kpts3d_cam = kpts2d_homo * kpts_depth
        kpts3d = pts_cam2world(kpts3d_cam, T)

        kpts3d_conf = self.out_conf(torch.cat((kpts_feat, kpts_depth.detach() / std_depth), dim=-1))
        
        return kpts3d, kpts3d_conf


class MLP(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super(MLP, self).__init__()

        self.out = nn.Sequential(nn.Linear(in_dim, inter_dim),
                                 nn.GELU(),
                                 nn.Linear(inter_dim, out_dim),
                                 )
        
    def forward(self, x):
        return self.out(x)


class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., drop_path=0., layer_scale=None):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_layer = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_layer = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = MLP(dim, dim*2, dim)

    def forward(self, q, kv):
        B, N, C = q.shape
        q_vec = self.q_layer(self.norm1(q)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_vec = self.kv_layer(self.norm1(kv)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_vec, v_vec = kv_vec[0], kv_vec[1]
        attn = (q_vec @ k_vec.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v_vec).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = q + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))

        return x

