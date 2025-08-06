# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from .kv_cache import KVCache, RotatingKVCache
from .eval import eval_mx_arrays

import mlx.core as mx
import mlx.nn as nn
import time

from collections import defaultdict

@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    causal: bool
    norm_first: bool
    bias_ff: bool
    bias_attn: bool
    layer_scale: float | None
    positional_embedding: str
    use_conv_block: bool
    cross_attention: bool
    conv_kernel_size: int
    use_conv_bias: bool
    gating: bool
    norm: str
    context: int
    max_period: int
    max_seq_len: int
    kv_repeat: int
    dim_feedforward: int
    conv_layout: bool

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class Id(nn.Module):
    def __init__(self):
        super().__init__()

    @eval_mx_arrays
    def __call__(self, xs: mx.array) -> mx.array:
        return xs


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.scale = mx.ones(dim)

    @eval_mx_arrays
    def __call__(self, xs: mx.array) -> mx.array:
        return xs * self.scale


@dataclass
class LayerCache:
    self_attn: KVCache | RotatingKVCache
    cross_attn: tuple[mx.array, mx.array] | None = None

    def reset(self):
        self.self_attn.reset()
        self.cross_attn = None


@eval_mx_arrays
def modify_linear_layer(layer: nn.Linear, column_to_remove: list[int], input_side: bool = True, quantization = None):
    if isinstance(layer, nn.layers.quantized.QuantizedLinear):
        return layer
        #raise "Can't prune this layer"
    if not column_to_remove:
        return layer
    original_out_features, original_in_features = layer.weight.shape
    original_weights = layer.weight
    if input_side:
        rows_to_keep = mx.array([i  for i in range(original_in_features) if not i in column_to_remove])
    else:
        rows_to_keep = mx.array([i  for i in range(original_out_features) if not i in column_to_remove])

    print(rows_to_keep)
    if input_side:
        new_weights = original_weights[:, rows_to_keep]
    else:
        new_weights = original_weights[rows_to_keep]
    print("new_weights", new_weights.shape, "was", original_weights.shape)

    if input_side:
        scaling_factor = mx.sqrt(original_in_features / (len(column_to_remove)))
    else:
        scaling_factor = mx.sqrt(original_out_features / (len(column_to_remove)))
    if input_side:
        scaling_factor = (original_in_features / (len(column_to_remove)))
    else:
        scaling_factor = (original_out_features / (len(column_to_remove)))
    scaled_weights = new_weights * scaling_factor
    #scaled_weights = new_weights

    if input_side:
        new_layer = nn.Linear(input_dims=len(rows_to_keep), output_dims=original_out_features, bias = False)
    else:
        new_layer = nn.Linear(input_dims=original_in_features, output_dims=len(rows_to_keep), bias = False)
    new_params = {"weight": scaled_weights}
    new_layer.update(new_params)
    if quantization:
        new_layer = new_layer.to_quantized(group_size=64, bits = quantization)


    return new_layer

n_crossattention = 0
remove_layers_cross = {
        #4: [8, 9, 10, 11, 12, 13, 14, 15],
        #5: [8, 9, 10, 11, 12, 13, 14, 15],
        #7: [15],
        #6: [8, 9, 10, 11, 12, 13, 14, 15],
        #7: [8, 9, 10, 11, 12, 13, 14, 15],
        #8: [8, 9, 10, 11, 12, 13, 14, 15],
        #9: [8, 9, 10, 11, 12, 13, 14, 15],
        #10: [8, 9, 10, 11, 12, 13, 14, 15],
}
cross_times = defaultdict(int)

class CrossAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        global n_crossattention
        super().__init__()

        n_crossattention += 1
        self.n_self = n_crossattention
        print("Cross", n_crossattention, cfg.num_heads)

        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)
        self.patched_outproj = True
        self.in_proj_stride = cfg.d_model
        self.keep = list(range(0, num_kv))
        if self.n_self in remove_layers_cross:
            self.id_removed = remove_layers_cross[self.n_self]
            self.id_removed = list(set(range(0, num_kv)) & set(self.id_removed))
            print("num_kv", num_kv, "kv_repeat", cfg.kv_repeat, "d_model", (cfg.d_model // cfg.num_heads))
            proj_width = (cfg.d_model // cfg.num_heads)
            self.keep = list(set(range(0, num_kv)) - set(self.id_removed))
            self.drop_oproj = [x * proj_width + y for x in self.id_removed for y in range(0, proj_width)]
            print("drop_oproj", self.drop_oproj)
            self.drop_iproj = [x + self.cfg.d_model * i for x in self.drop_oproj for i in range(0, 3)]
            # Shouldn't we update the scale...?
            # Looks like it works better without it
            #self.scale = (len(self.keep)) ** (-0.5)
            self.patched_outproj = False
            #self.cfg.d_model = len(self.keep)
            print("blah", cfg.d_model, proj_width, len(self.id_removed), proj_width * len(self.id_removed))
            self.in_proj_stride = cfg.d_model - proj_width * len(self.id_removed)
            print("changed in_proj_stride from", cfg.d_model, "to", self.in_proj_stride)

    def quantize(self):
        cfg = self.cfg
        self.in_proj_q = nn.Linear(cfg.d_model, cfg.d_model, bias = cfg.bias_attn)
        self.in_proj_q.weight = self.in_proj.weight[:cfg.d_model]
        self.in_proj_q = self.in_proj_q.to_quantized(group_size = 64, bits = 4)
        self.out_proj = self.out_proj.to_quantized(group_size = 64, bits = 4)


    def __call__(
        self,
        xs: mx.array,
        cross_attention_src: mx.array,
        cache: LayerCache,
    ) -> mx.array:
        # TODO: Add some cross-attention kv caching.
        assert self.cfg.kv_repeat == 1, "only kv_repeat==1 is supported"
        startts = time.time()
        #print("ZA", time.time() - startts)

        b, t, hd = xs.shape
        qkv_w = self.in_proj.weight
        #print("BABA", xs.shape, qkv_w[:self.in_proj_stride].T.shape)
        #q = xs @ qkv_w[:self.in_proj_stride].T
        q = self.in_proj_q(xs)
        q = q.reshape(b, t, len(self.keep), self.cfg.head_dim).swapaxes(1, 2)
        #print("ZB", time.time() - startts)

        if cache.cross_attn is None:
            b_kv, t_kv, hd_kv = cross_attention_src.shape
            assert b == b_kv
            assert hd == hd_kv
            assert "bias" not in self.in_proj
            k = cross_attention_src @ qkv_w[self.in_proj_stride:2 * self.in_proj_stride].T
            k = k.reshape(b, t_kv, len(self.keep), self.cfg.head_dim).swapaxes(1, 2)
            v = cross_attention_src @ qkv_w[2 * self.in_proj_stride:].T
            v = v.reshape(b, t_kv, len(self.keep), self.cfg.head_dim).swapaxes(1, 2)
            cache.cross_attn = k, v
        else:
            k, v = cache.cross_attn

        if self.n_self in remove_layers_cross:
            ##print("Bias attention", self.cfg.bias_attn)
            ##print("qkv pre", k.shape, k.shape, v.shape)
            ##print("Keep columns", self.keep)
            #k = k[:, self.keep] #mx.concatenate((k[:, :id_removed], k[:, (id_removed+1):]), axis = 1)
            #v = v[:, self.keep] #mx.concatenate((v[:, :id_removed], v[:, (id_removed+1):]), axis = 1)
            #q = q[:, self.keep] #mx.concatenate((q[:, :id_removed], q[:, (id_removed+1):]), axis = 1)
            ##print("qkv post", k.shape, k.shape, v.shape)
            hd = k.shape[1] * k.shape[3]

        #print("ZC", time.time() - startts)
        #print("ZD", time.time() - startts)
        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        #print("ZE", time.time() - startts)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        #print("res", xs.shape)
        #print("ZF", time.time() - startts)

        xs = self.out_proj(xs)
        #print("ZG", time.time() - startts)
        return xs


n_attention = 0
remove_layers = {
}
# default quant is 8
quantizations = {
}
for i in range(0, 160):
    #remove_layers[i] = list(range(4, 16))
    quantizations[i] = 4
# Those transformers aren't used
#for i in range(144, 150):
#    remove_layers[i] = list(range(4, 8))
# 
#for i in range(19, 144, 4):
#    remove_layers[i] = list(range(15, 16))
#for i in range(18, 144, 4):
#    remove_layers[i] = list(range(15, 16))
remove_layers_keep_iproj = {
}
# 16 - 144 kv can't be pruned because they share their cache (unless you remove all their heads (step = 4)
#for i in range(44, 144, 4):
#    remove_layers_keep_iproj[i] = list(range(4, 16))
attn_times = defaultdict(int)
class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        global n_attention
        super().__init__()
        self.n_self = n_attention
        n_attention += 1

        print("Attention", n_attention, cfg.num_heads)
        num_kv = cfg.num_heads // cfg.kv_repeat
        out_dim = cfg.d_model + 2 * num_kv * cfg.d_model // cfg.num_heads
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, out_dim, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope":
            self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.max_period)

        self.patched_outproj = False
        self.in_proj_stride = cfg.d_model
        self.keep = list(range(0, num_kv))
        self.in_proj_heads = len(self.keep)
        self.patched = False
        self.id_removed = []
        self.keep_iproj = True
        if self.n_self in remove_layers_keep_iproj:
            self.id_removed = remove_layers_keep_iproj[self.n_self]
        if self.n_self in remove_layers:
            self.id_removed = remove_layers[self.n_self]
            self.keep_iproj = False
        if self.id_removed:
            self.patched = True
            self.id_removed = list(set(range(0, num_kv)) & set(self.id_removed))
            print("num_kv", num_kv, "kv_repeat", cfg.kv_repeat, "d_model", (cfg.d_model // cfg.num_heads))
            proj_width = (cfg.d_model // cfg.num_heads)
            self.keep = list(set(range(0, num_kv)) - set(self.id_removed))
            self.drop_oproj = [x * proj_width + y for x in self.id_removed for y in range(0, proj_width)]
            print("drop_oproj", self.drop_oproj)
            self.drop_iproj = [x + self.cfg.d_model * i for x in self.drop_oproj for i in range(0, 3)]
            # Shouldn't we update the scale...?
            # Looks like it works better without it
            #self.scale = (len(self.keep)) ** (-0.5)
            self.patched_outproj = False
            #self.cfg.d_model = len(self.keep)
            print("blah", cfg.d_model, proj_width, len(self.id_removed), proj_width * len(self.id_removed))
            self.in_proj_stride = cfg.d_model - proj_width * len(self.id_removed)
            print("changed in_proj_stride from", cfg.d_model, "to", self.in_proj_stride)
            print("out_dim is", out_dim)
        self.in_proj_heads = len(self.keep)

    def __call__(
        self,
        xs: mx.array,
        cache: KVCache | RotatingKVCache,
        mask: mx.array | None = None,
    ) -> mx.array:
        assert self.cfg.kv_repeat == 1, "only kv_repeat==1 is supported"
        #if not self.patched_outproj:
        #    self.patched_outproj = True
        #    quant = 8
        #    if self.n_self in quantizations:
        #        quant = quantizations[self.n_self]
        #    if self.patched:
        #        if isinstance(self.in_proj, nn.layers.quantized.QuantizedLinear):
        #            num_kv = self.cfg.num_heads // self.cfg.kv_repeat
        #            self.keep = list(range(0, num_kv))
        #            raise
        #        else:
        #            self.out_proj = modify_linear_layer(self.out_proj, self.drop_oproj, quantization = quant)
        #            if self.keep_iproj:
        #                self.in_proj = self.in_proj.to_quantized(group_size=64, bits=quant)
        #            else:
        #                self.in_proj = modify_linear_layer(self.in_proj, self.drop_iproj, input_side = False, quantization = quant)
        #    else:
        #        if not isinstance(self.in_proj, nn.layers.quantized.QuantizedLinear):
        #            self.out_proj = self.out_proj.to_quantized(group_size=64, bits=quant)
        #            self.in_proj = self.in_proj.to_quantized(group_size=64, bits=quant)
        #        else:
        #        pass
        #    if self.keep_iproj:
        #        self.in_proj_heads = self.cfg.num_heads
        #    else:
        #        self.in_proj_heads = len(self.keep)

        b, t, hd = xs.shape
        qkv = self.in_proj(xs).reshape(b, t, 3, self.in_proj_heads, self.cfg.head_dim)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.rope is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)

        #if False and not self.patched:
        k, v = cache.update_and_fetch(k, v)
        k_len = k.shape[2]
        k_target_len = t + min(self.cfg.context, k_len - t)
        if k_target_len < k_len:
            k = k[:, :, k_len - k_target_len :]
            v = v[:, :, k_len - k_target_len :]

        #if self.keep_iproj:
        k = k[:, self.keep]
        v = v[:, self.keep]
        q = q[:, self.keep]
        hd = k.shape[1] * k.shape[3]
        #print(f"QKV dims, value_head_dim {v.shape[-1]} query_head_dim {q.shape[-1]}, query_sequence_length {q.shape[2]}, key_sequence_length {k.shape[2]}")
        xs = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        xs = xs.transpose(0, 2, 1, 3).reshape(b, t, hd)
        xs = self.out_proj(xs)
        return xs


class MlpGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        if cfg.dim_feedforward == 4 * cfg.d_model:
            hidden = 11 * cfg.d_model // 4

        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias_ff)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias_ff)

    @eval_mx_arrays
    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.linear_in(xs)
        b, t, _ = xs.shape
        xs = xs.reshape(b, t, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class MlpNoGating(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias_ff)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias_ff)

    @eval_mx_arrays
    def __call__(self, xs: mx.array) -> mx.array:
        return self.linear2(nn.gelu_approx(self.linear1(xs)))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        assert not cfg.use_conv_block, "conv-block is not supported"
        if cfg.gating:
            self.gating = MlpGating(cfg)
        else:
            # TODO: Use a better name?
            self.gating = MlpNoGating(cfg)

        if cfg.norm == "layer_norm":
            self.norm1 = nn.LayerNorm(cfg.d_model, 1e-5)
            self.norm2 = nn.LayerNorm(cfg.d_model, 1e-5)
        elif cfg.norm == "rms_norm":
            self.norm1 = nn.RMSNorm(cfg.d_model, 1e-8)
            self.norm2 = nn.RMSNorm(cfg.d_model, 1e-8)
        else:
            raise ValueError(f"unsupported norm type {cfg.norm}")

        if cfg.layer_scale is not None:
            self.layer_scale_1 = LayerScale(cfg.d_model)
            self.layer_scale_2 = LayerScale(cfg.d_model)
        else:
            self.layer_scale_1 = Id()
            self.layer_scale_2 = Id()
        self.self_attn = Attention(cfg)
        self.cfg = cfg

        if cfg.cross_attention:
            # Always use layer-norm for the cross-attention.
            self.norm_cross = nn.LayerNorm(cfg.d_model, 1e-5)
            self.cross_attention = CrossAttention(cfg)
        else:
            self.cross_attention = None

    @eval_mx_arrays
    def _cross_attention_block(
        self,
        x: mx.array,
        cache: LayerCache,
        cross_attention_src: mx.array,
    ) -> mx.array:
        assert self.cross_attention is not None
        x_orig = x
        x = self.norm_cross(x)
        update = self.cross_attention(x, cross_attention_src, cache)
        return x_orig + update

    @eval_mx_arrays
    def norm1_(self, x):
        x = self.norm1(x)
        return x

    @eval_mx_arrays
    def layer_scale_1_(self, x):
        x = self.layer_scale_1(x)
        return x

    @eval_mx_arrays
    def layer_scale_2_(self, x):
        x = self.layer_scale_2(x)
        return x

    @eval_mx_arrays
    def __call__(
        self,
        xs: mx.array,
        cache: LayerCache,
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        n1 = self.norm1_(xs)
        n1 = self.self_attn(n1, cache=cache.self_attn)
        xs = xs + self.layer_scale_1_(n1)
        if self.cross_attention is not None:
            assert cross_attention_src is not None
            xs = self._cross_attention_block(xs, cache, cross_attention_src)
        else:
            assert cross_attention_src is None
        xs = xs + self.layer_scale_2_(self.gating(self.norm2(xs)))
        return xs


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg=cfg) for _ in range(cfg.num_layers)]

    @eval_mx_arrays
    def __call__(
        self,
        xs: mx.array,
        cache: list[LayerCache],
        cross_attention_src: None | mx.array = None,
    ) -> mx.array:
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c, cross_attention_src=cross_attention_src)
        return xs

    def make_cache(self) -> list[LayerCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat

        a = []
        for l in self.layers:
            a.append(
                LayerCache(
                    KVCache(head_dim=self.cfg.head_dim, n_kv_heads=len(l.self_attn.keep))
                )
            )
        return a

    def make_rot_cache(self) -> list[LayerCache]:
        num_kv_heads = self.cfg.num_heads // self.cfg.kv_repeat
        a = []
        for l in self.layers:
            a.append(
                LayerCache(
                    RotatingKVCache(
                        head_dim=self.cfg.head_dim,
                        n_kv_heads=len(l.self_attn.keep),
                        max_size=self.cfg.max_seq_len,
                    )
                )
            )
        return a


class ProjectedTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, input_dim: int, output_dims: list[int]):
        super().__init__()

        self.conv_layout = cfg.conv_layout
        self.transformer = Transformer(cfg)
        if input_dim == cfg.d_model:
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(input_dim, cfg.d_model, bias=False)

        output_projs = []
        for output_dim in output_dims:
            if output_dim == cfg.d_model:
                p = None
            else:
                p = nn.Linear(cfg.d_model, output_dim, bias=False)
            output_projs.append(p)
        self.output_projs = output_projs

    @eval_mx_arrays
    def __call__(
        self,
        xs: mx.array,
        cache: list[LayerCache],
        cross_attention_src: None | mx.array = None,
    ) -> list[mx.array]:
        if self.conv_layout:
            xs = xs.swapaxes(1, 2)
        if self.input_proj is not None:
            xs = self.input_proj(xs)
        xs = self.transformer(xs, cache=cache, cross_attention_src=cross_attention_src)
        outs = []
        for output_proj in self.output_projs:
            if output_proj is None:
                out = xs
            else:
                out = output_proj(xs)
            if self.conv_layout:
                out = out.swapaxes(1, 2)
            outs.append(out)
        return outs

    def make_cache(self) -> list[LayerCache]:
        return self.transformer.make_cache()

    def make_rot_cache(self) -> list[LayerCache]:
        return self.transformer.make_rot_cache()
