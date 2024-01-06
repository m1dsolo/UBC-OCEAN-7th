from math import pi
from functools import wraps

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import torch
from torch import nn, einsum
import torch.nn.functional as F


def print_trainable_parameters(model: torch.nn) -> None:
    """Print number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param}"
        f" || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0., scale=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        if scale:
            self.scale = scale #**-1
        else:
            self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        A = sim.softmax(dim = -1)
        attn = self.dropout(A)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        if context.shape != x.shape:
            return self.to_out(out), A
        else:
            return self.to_out(out)
        

class DualQueryCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0., scale=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        if scale:
            self.scale = nn.Parameter(torch.tensor([scale])) #**-1
        else:
            self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

        # Attention ranking
        self.to_score_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_score_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, score_x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        score_q = self.to_score_q(score_x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        score_sim = einsum('b i d, b j d -> b i j', score_q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        A = sim.softmax(dim = -1)
        attn = self.dropout(A)

        score_attn = score_sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        score_out = einsum('b i j, b j d -> b i d', score_attn, v)
        score_out = rearrange(score_out, '(b h) n d -> b n (h d)', h = h)

        return self.to_out(out), A, self.to_score_out(score_out), score_attn


# Based on the merging approach from Truong et al. "How Transferable are Self-supervised Features in Medical Image Classification Tasks?"
class Merger(nn.Module):
    def __init__(self, proj_dim):
        super(Merger, self).__init__()
        
        self.vit_head = nn.Linear(384, proj_dim)
        self.swin_head = nn.Linear(768, proj_dim)
        self.swav_head = nn.Linear(2048, proj_dim)
        

    def forward(self, data):
        vit_out = self.vit_head(data['vit_feats'])
        swin_out = self.swin_head(data['swin_feats'])
        swav_out = self.swav_head(data['swav_feats'])
      
        joint = torch.cat([vit_out, swin_out, swav_out], dim=-1)
        return joint


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 1024,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        n_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        latent_bounds = 2,
        scale = None,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.n_classes = n_classes

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        self.proj_embeddings = nn.Identity()
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((num_latents, latent_dim)), 
                mean=0, 
                std=0.02, 
                a=-latent_bounds, 
                b=latent_bounds))
        

        self.score_latents = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((1, latent_dim)), 
                mean=0, 
                std=0.02, 
                a=-latent_bounds, 
                b=latent_bounds))
                    
        # Cross-Attention Layer
        get_cross_attn = lambda: PreNorm(latent_dim, DualQueryCrossAttention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout, scale=scale), context_dim = input_dim) #new
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_mil_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        
        
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        
        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_mil_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff, get_mil_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_mil_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, n_classes)
        )

        self.to_score_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, n_classes)
        )

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False,
    ):
        data = self.proj_embeddings(data)
                

        if len(data.shape)==2: # flops
            data= data.unsqueeze(0)  # flops
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

      
        score_x = repeat(self.score_latents, 'n d -> b n d', b = b)

        # layers
        for cross_attn, cross_ff, mil_ff, self_attns in self.layers: 
            x_attn, A_raw, score_x_attn, score_A  = cross_attn(x=x, score_x=score_x, context=data, mask=mask)
            x = x_attn + x
            x = cross_ff(x) + x
            score_x = score_x_attn + score_x
            score_x = mil_ff(score_x) + score_x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # to logits
        logits = self.to_logits(x)
        results_dict={'student_logits':self.to_score_logits(score_x), 'features_teacher':x, 'features_student':score_x}
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, score_A, results_dict       

