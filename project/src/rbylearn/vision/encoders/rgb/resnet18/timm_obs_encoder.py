'''
DISCLAIMER:

This script contains copy-pasted utility functions from the pointW/sym_in_dp repository.
It is copied here for convenient reproducibility of their encoder architecture because we also made some minor adjumstments (see NOTE/TODO comments below). 

Specifically, the contents come from:

sym_in_dp/common/pytorch_util.py
sym_in_dp/common/module_attr_mixin.py
sym_in_dp/model/vision/timm_obs_encoder.py

'''
import copy
from typing import Dict, Callable, List
import collections
import logging
import timm

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

logger = logging.getLogger(__name__)

RGB_KEY = "image"  # the single camera key we feed into TimmObsEncoder (this doesn't need to match their dataset's key, because we don't use their dataset here, we just want to match the encoder architecture)

def make_shape_meta(image_size: int = 84) -> dict:
    return {"obs": {
        RGB_KEY: {"shape": (3, image_size, image_size), "type": "rgb", "horizon": 1},
        "robot0_eef_pos": {"shape": (3,), "type": "low_dim", "horizon": 1, "ignore_by_policy": True},
    }}


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class TrainRandomTestCenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.random_crop = torchvision.transforms.RandomCrop(size=size)
        self.center_crop = torchvision.transforms.CenterCrop(size=size)
    
    def forward(self, x):
        if self.training:
            return self.random_crop(x)
        else:
            return self.center_crop(x)

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            frozen_attention_embedding: bool=False
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        self.low_dim_horizon = shape_meta['obs']['robot0_eef_pos']['horizon']

        assert global_pool == ''

        model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                num_classes=0            # remove classification layer
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None


        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        
        if use_group_norm:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        example_obs = torch.zeros(
            (1, 3) + image_shape, 
            dtype=self.dtype,
            device=self.device)
        
        example_obs = transform(example_obs)
        example_output = model(example_obs)

        feature_dim = example_output.shape[1]
        feature_map_shape = example_output.shape[2:]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        logger.info('rgb keys: %s', rgb_keys)
        logger.info('low_dim keys: %s', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        
        if self.feature_aggregation == 'attention_pool_2d':
            # NOTE: I added this if statement because the attention pool 2d has a random initialization and I want to be able to freeze it for reproducibility.
            if frozen_attention_embedding:
                torch.manual_seed(0)
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        elif self.feature_aggregation == 'adaptive_avg_pool2d':
            self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        elif self.feature_aggregation == 'spatial_softmax':
            ### NOTE: I removed the spatial_softmax option because we don't use it and it requires robomimic which which raised a bunch of dependency issues on a macbook.
            raise NotImplementedError(f": I removed the spatial_softmax option because we don't use it and it requires robomimic which which raised a bunch of dependency issues on a macbook.")
            #     num_kp = feature_dim//2
            #     self.spatial_softmax = SpatialSoftmax(
            #         input_shape=(feature_dim, *feature_map_shape),
            #         num_kp=num_kp
            #     )
        else:
            raise NotImplementedError(f"Unsupported feature aggregation: {self.feature_aggregation}")
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)
        elif self.feature_aggregation == 'adaptive_avg_pool2d':
            return self.adaptive_avg_pool2d(feature).squeeze(-1).squeeze(-1)
        elif self.feature_aggregation == 'spatial_softmax':
            return self.spatial_softmax(feature).reshape(feature.shape[0], -1)
        else:
            raise NotImplementedError(f"Unsupported feature aggregation: {self.feature_aggregation}")
        
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.shape[1] > self.shape_meta['obs'][key]['horizon']:
                img = img[:, -self.shape_meta['obs'][key]['horizon']:, ...]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if data.shape[1] > self.shape_meta['obs'][key]['horizon']:
                data = data[:, -self.shape_meta['obs'][key]['horizon']:, ...]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape[1:]




