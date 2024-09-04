import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def spike_function(v_scaled):
    z_ = (v_scaled > 0).float()

    def grad_fn(grad_output):
        dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
        grad_input = grad_output * dz_dv_scaled
        return grad_input, None

    return z_, grad_fn


class FSSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, d, T, K):
        ctx.save_for_backward(x, h, d, T)
        ctx.K = K

        out = torch.zeros_like(x)
        v = x
        for t in range(K):
            v_scaled = v - T[t]
            z = spike_function(v_scaled)[0]
            # z = (v_scaled > 0).float()
            out += z * d[t]
            v = v - z * h[t]
        return out

    # @staticmethod
    def backward(ctx, grad_output):
        x, h, d, T = ctx.saved_tensors
        K = ctx.K

        grad_x = torch.zeros_like(x)
        v = x
        for t in range(K):
            v_scaled = v - T[t]
            dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
            grad_input = grad_output * dz_dv_scaled
            grad_x += grad_input * d[t].to(grad_output.device)
            v = v - (v_scaled > 0).float() * h[t].to(grad_output.device)
        return grad_x, None, None, None, None


def fs_swish(x, h, d, T):
    K = len(h)
    return FSSwishFunction.apply(x, h, d, T, K)


# original
# class FSSwishLayer(nn.Module):
#     def __init__(self):
#         super(FSSwishLayer, self).__init__()
#         self.swish_h = torch.tensor([0.6667, 0.5870, 1.4833, 3.0379, 2.4488, 2.9923, 2.5679, 1.8541, 0.9022, 0.9178, 0.5000, 0.2846, 0.1595, 0.0839, 0.0417, 1.1349], dtype=torch.float32)
#         self.swish_d = torch.tensor([0.3649, 0.7144, -0.2097, 3.1201, 2.4468, 3.0275, 2.5682, 1.8909, 0.9216, 0.9120, 0.4913, 0.2730, 0.1540, 0.0853, 0.0470, 0.0264], dtype=torch.float32)
#         self.swish_T = torch.tensor([0.0834, 2.0955, -3.5100, 2.1758, 2.4932, 1.3676, 1.7017, 0.8178, -0.1175, -0.6438, -1.2931, -1.4991, -1.6387, -1.6972, -1.7321, -1.7469], dtype=torch.float32)
#
#     def forward(self, x):
#         out = fs_swish(x, self.swish_h, self.swish_d, self.swish_T)
#         return out

# k=24, 20,000 epoochs trained
class FSSwishLayer(nn.Module):
    def __init__(self):
        super(FSSwishLayer, self).__init__()
        self.swish_h = torch.tensor([ 0.4462,  0.9426,  0.5828,  0.2679,  0.1929,  1.1032,  0.0062,  1.7608,
         1.6892,  1.0465,  2.2203, -0.0518,  0.9965,  1.2357,  0.7535,  1.3039], dtype=torch.float32)
        self.swish_d = torch.tensor([ 0.1441,  1.0263,  0.5819,  0.2583,  0.0890,  0.8074,  0.1049,  1.2033,
         1.8082,  0.4312,  2.2586, -0.2693,  0.8391,  0.0463,  0.2339,  0.1115], dtype=torch.float32)
        self.swish_T = torch.tensor([-0.4326,  0.7987,  0.1965, -0.0293,  1.7898,  0.4043, -0.1738, -0.0356,
         2.1835, -0.0467,  2.3067, -1.7284,  1.2810,  0.9420, -0.2450, -0.5279], dtype=torch.float32)

    def forward(self, x):
        out = fs_swish(x, self.swish_h, self.swish_d, self.swish_T)
        return out


def fs(x):
    # replace swish with fs_swish
    fs_swish_layer = FSSwishLayer()
    return fs_swish_layer(x)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def relu(x):
    return torch.relu(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):

        # Print the shapes of x and h
        #print(f"x shape: {x.shape}, temb shape: {temb.shape}")

        h = x
        h = self.norm1(h)

        # Print input statistics before nonlinearity
        #print(f"Before nonlinearity: min={h.min().item()}, max={h.max().item()}, mean={h.mean().item()}, std={h.std().item()}")


        h = nonlinearity(h)
        #h = fs(h)

        # Print output statistics after nonlinearity
        #print(f"After nonlinearity: min={h.min().item()}, max={h.max().item()}, mean={h.mean().item()}, std={h.std().item()}")

        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        #h = h + self.temb_proj(fs(temb))[:, :, None, None]

        h = self.norm2(h)

        # Print input statistics before second nonlinearity
        #print(f"Before second nonlinearity: min={h.min().item()}, max={h.max().item()}, mean={h.mean().item()}, std={h.std().item()}")

        h = nonlinearity(h)
        #h = fs(h)

        # Print output statistics after second nonlinearity
        #print(f"After second nonlinearity: min={h.min().item()}, max={h.max().item()}, mean={h.mean().item()}, std={h.std().item()}")

        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ]).to(self.device)

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1).to(self.device)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout).to(self.device))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in).to(self.device))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv).to(self.device)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout).to(self.device)
        self.mid.attn_1 = AttnBlock(block_in).to(self.device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout).to(self.device)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout).to(self.device))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in).to(self.device))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv).to(self.device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in).to(self.device)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1).to(self.device)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch).to(self.device)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        #temb = fs(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        #h = fs(h)
        h = self.conv_out(h)
        return h