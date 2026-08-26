## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


from multiprocessing import context
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from torchvision.transforms.functional import resize, center_crop, normalize

########
# GFLOPs: 195.08, MParams: 8.00, Speed: 0.133s
# fusionNet = Network(
#     enc_blk_nums = [2, 2, 2],
#     middle_blk_num = 2,
#     dec_blk_nums = [2, 2, 2],
#     restormer_heads = [1, 2, 4],
#     restormer_middle_heads = 8,
# )

#### ----------------Original Cross Attention & Guidance & Refinement---------------  #####
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, inner_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.scale = (inner_dim // num_heads) ** -0.5
        
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, context):
        query_norm = self.norm_query(query)
        context_norm = self.norm_context(context)
        
        q = self.to_q(query_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        attention_update = self.to_out(output)

        return attention_update
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, context=None):
        x_out = x + self.attn(self.norm1(x))
        x_out = x_out + self.ffn(self.norm2(x_out))

        return x_out

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
        
# Baseline
class Network(nn.Module):
    def __init__(self, img_channel=3, width=48, middle_blk_num=1, 
                 enc_blk_nums=[], dec_blk_nums=[], 
                 restormer_heads=[1, 2, 4],
                 restormer_middle_heads=8,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        
        num_stages = len(enc_blk_nums)

        self.intro_vi = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.intro_ir = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        self.encoders_vi = nn.ModuleList()
        self.encoders_ir = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.reduces = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders_vi.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_heads[i], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            self.encoders_ir.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_heads[i], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan *= 2

        self.middle_reduce = nn.Conv2d(chan * 2, chan, kernel_size=1, stride=1)
        self.middle_blks = nn.Sequential(*[TransformerBlock(dim=chan, num_heads=restormer_middle_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(middle_blk_num)])

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            
            stage_idx = num_stages - 1 - i
            current_restormer_heads = restormer_heads[stage_idx]

            self.fusions.append(nn.Conv2d(chan *2, chan, kernel_size=1, stride=1))
            self.reduces.append(nn.Conv2d(chan *2, chan, kernel_size=1, stride=1))
            self.decoders.append(nn.Sequential(*[TransformerBlock(dim=chan, num_heads=current_restormer_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num)]))
            

    def forward(self, inp_vi, inp_ir):
        x_vi = self.intro_vi(inp_vi)
        x_ir = self.intro_ir(inp_ir)
        
        encs_vi = []
        encs_ir = []
        
        for encoder_vi, encoder_ir, down in zip(self.encoders_vi, self.encoders_ir, self.downs):
            x_vi = encoder_vi(x_vi)   
            x_ir = encoder_ir(x_ir)
            encs_vi.append(x_ir)
            encs_ir.append(x_ir)
            x_vi = down(x_vi)
            x_ir = down(x_ir)

        x = self.middle_reduce(torch.concat([x_vi, x_ir], dim=1))
        x = self.middle_blks(x)
        
        for decoder_blocks, up, fusion, reduce, enc_vi_skip, enc_ir_skip in zip(
            self.decoders, self.ups, self.fusions, self.reduces, encs_vi[::-1], encs_ir[::-1],
        ):
            x = up(x)
            enc_skip = fusion(torch.concat([enc_vi_skip, enc_ir_skip], dim=1))
            x = reduce(torch.concat([x, enc_skip],dim=1)) 
            x = decoder_blocks(x)
            
        x = self.ending(x)
        return x