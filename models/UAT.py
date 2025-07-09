import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.pvtv2 import pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_cha, out_cha, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_cha),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class cross_layer_attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.num_heads = num_heads

        self.high_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.high_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.high_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.mask_q = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, high_fea, mask):
        B, N, C = high_fea.shape

        high_q = self.high_q(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        high_k = self.high_k(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        high_v = self.high_v(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if mask is None:
            high_attn = torch.matmul(high_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = (torch.matmul(high_attn, high_v)).transpose(2, 1).reshape(B, N, C)

        else:
            mask_q = self.mask_q(mask).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            high_attn = torch.matmul(mask_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = (torch.matmul(high_attn, high_v)).transpose(2, 1).reshape(B, N, C)

        return high_attn


class encoder_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = cross_layer_attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class cross_attention_encoder(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio):
        super(cross_attention_encoder, self).__init__()
        self.depth = depth
        self.block = encoder_block(dim, num_heads, mlp_ratio)

    def forward(self, x, mask):
        for _ in range(self.depth):
            x = self.block(x, mask)
        return x


class uncertainty_generation(nn.Module):
    def __init__(self, dim, imgsize):
        super(uncertainty_generation, self).__init__()
        self.imgsize = imgsize
        self.soft_split = nn.Unfold(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))                     #image-->tokens
        self.soft_fuse = nn.Fold(output_size=(self.imgsize // 4, self.imgsize // 4), kernel_size=(4, 4),
                                  stride=(4, 4), padding=(0, 0))                                           #tokens-->image
        self.mean_conv = nn.Sequential(
            nn.Conv2d(dim//16, dim//16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1, stride=1, padding=0),
        )

        self.std_conv = nn.Sequential(
            nn.Conv2d(dim//16, dim//16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1, stride=1, padding=0),
        )

        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def reparameterize(self, mu, logvar, k):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type:
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def forward(self, x):
        x = self.soft_fuse(x.transpose(-2, -1))

        mean = self.mean_conv(x)
        std = self.std_conv(x)

        prob = self.reparameterize(mean, std, 1)         

        prob_out = self.reparameterize(mean, std, 50)    
        prob_out = torch.sigmoid(prob_out)
        uncertainty = prob_out.var(dim=1, keepdim=True).detach()
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        uncertainty = (1 - uncertainty) * x
        uncertainty = self.soft_split(uncertainty).transpose(-2, -1)
        return prob, uncertainty


class probabilistic_attention(nn.Module):
    def __init__(self, dim, num_heads, imgsize, qkv_bias=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.tokens_to_uncertainty = uncertainty_generation(dim, imgsize)

        self.fea_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.fea_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.fea_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, fea, uncertainty_map=None, uncertainty=True):
        if uncertainty:
            B, N, C = fea.shape
            prob, uncertainty_q = self.tokens_to_uncertainty(fea)

            fea_q = self.fea_q(uncertainty_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            fea_k = self.fea_k(fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            fea_v = self.fea_v(fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            attn = torch.matmul(fea_q, fea_k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = (torch.matmul(attn, fea_v)).transpose(2, 1).reshape(B, N, C)
            return prob, uncertainty_q, attn

        else:
            B, N, C = fea.shape

            fea_q = self.fea_q(uncertainty_map).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            fea_k = self.fea_k(fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            fea_v = self.fea_v(fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            attn = torch.matmul(fea_q, fea_k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = (torch.matmul(attn, fea_v)).transpose(2, 1).reshape(B, N, C)
            return attn


class decoderblock(nn.Module):
    def __init__(self, dim, num_heads, imgsize, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = probabilistic_attention(dim=dim, num_heads=num_heads, imgsize=imgsize, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, fea, uncertainty_map=None, uncertainty=True):
        if uncertainty:
            prob, uncertainty_q, attn = self.attn(self.norm1(fea))
            fea = fea + self.drop_path(attn)
            fea = fea + self.drop_path(self.mlp(self.norm2(fea)))
            return prob, uncertainty_q, fea

        else:
            fea = fea + self.drop_path(self.attn(self.norm1(fea), self.norm1(uncertainty_map), uncertainty=False))
            fea = fea + self.drop_path(self.mlp(self.norm2(fea)))
            return fea


class Transformer_probabilistic_decoder(nn.Module):
    def __init__(self, dim, num_heads, depth, imgsize, mlp_ratio):
        super(Transformer_probabilistic_decoder, self).__init__()
        self.depth = depth
        self.block = decoderblock(dim, num_heads, imgsize, mlp_ratio)

    def forward(self, fea, uncertainty_map=None, uncertainty=True):

        if uncertainty:
            for _ in range(self.depth):
                prob, uncertainty_q, fea = self.block(fea)
                return prob, uncertainty_q, fea

        else:
            for _ in range(self.depth):
                fea = self.block(fea, uncertainty_map, uncertainty=False)
                return fea


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.imgsize = opt.imgsize
        self.backbone = pvt_v2_b2()
        path = './pvt_weights/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.dim_out = opt.dim
        self.sigmoid = nn.Sigmoid()
        self.soft_split = nn.Unfold(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        self.soft_fuse = nn.Fold(output_size=(self.imgsize // 4, self.imgsize // 4), kernel_size=(4, 4),
                                  stride=(4, 4), padding=(0, 0))

        self.conv_ref = BasicConv2d(2048, self.dim_out, kernel_size=1, stride=1, padding=0)
        self.conv0 = BasicConv2d(64, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv1 = BasicConv2d(128, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(320, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(512, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(self.dim_out, 1, kernel_size=1, stride=1, padding=0)

        self.TransformerEncoder3 = cross_attention_encoder(dim=self.dim_out*4*4, num_heads=16, depth=2, mlp_ratio=4.)
        self.TransformerEncoder2 = cross_attention_encoder(dim=self.dim_out*4*4, num_heads=16, depth=2, mlp_ratio=4.)
        self.TransformerEncoder1 = cross_attention_encoder(dim=self.dim_out*4*4, num_heads=16, depth=3, mlp_ratio=4.)
        self.TransformerEncoder0 = cross_attention_encoder(dim=self.dim_out*4*4, num_heads=16, depth=4, mlp_ratio=4.)

        self.TransDecoder_uncertainty = Transformer_probabilistic_decoder(dim=self.dim_out*4*4, num_heads=8,
                                                                          depth=4, imgsize=self.imgsize, mlp_ratio=4.0)
        self.TransDecoder = Transformer_probabilistic_decoder(dim=self.dim_out*4*4, num_heads=8,
                                                              depth=4, imgsize=self.imgsize, mlp_ratio=4.0)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, ref_x, y=None, training=True):

        B, _, _, _ = x.shape
        pvt = self.backbone(x)
        x0, x1, x2, x3 = pvt[0], pvt[1], pvt[2], pvt[3]

        ref_x = self.sigmoid(self.conv_ref(ref_x))
        x3 = torch.mul(self.conv3(x3), ref_x)
        x2 = torch.mul(self.conv2(x2), ref_x)
        x1 = torch.mul(self.conv1(x1), ref_x)
        x0 = torch.mul(self.conv0(x0), ref_x)

        x3 = self.soft_split(self.upsample8(x3)).transpose(-2, -1)
        x2 = self.soft_split(self.upsample4(x2)).transpose(-2, -1)
        x1 = self.soft_split(self.upsample2(x1)).transpose(-2, -1)
        x0 = self.soft_split(x0).transpose(-2, -1)

        s3 = self.TransformerEncoder3(x3, mask=None)
        s2 = self.TransformerEncoder2(x2, s3)
        s1 = self.TransformerEncoder1(x1, s2)
        s0 = self.TransformerEncoder0(x0, s1)

        prob, uncertainty_q, s3 = self.TransDecoder_uncertainty(s3, uncertainty_map=None, uncertainty=True)

        s2 = s2 + s3
        s2 = self.TransDecoder(s2, uncertainty_map=uncertainty_q, uncertainty=False)

        s1 = s1 + s2
        s1 = self.TransDecoder(s1, uncertainty_map=uncertainty_q, uncertainty=False)

        s0 = s0 + s1
        s0 = self.TransDecoder(s0, uncertainty_map=uncertainty_q, uncertainty=False)

        s3 = self.upsample4(self.conv_out(self.soft_fuse(s3.transpose(-2, -1))))
        s2 = self.upsample4(self.conv_out(self.soft_fuse(s2.transpose(-2, -1))))
        s1 = self.upsample4(self.conv_out(self.soft_fuse(s1.transpose(-2, -1))))
        s0 = self.upsample4(self.conv_out(self.soft_fuse(s0.transpose(-2, -1))))

        prob = self.upsample4(prob)

        if training:

            loss_prob = 0.1 * self.kl_loss(F.log_softmax(prob, dim=-1), F.softmax(y, dim=-1))

            return s3, s2, s1, s0, loss_prob

        else:

            return s3, s2, s1, s0
