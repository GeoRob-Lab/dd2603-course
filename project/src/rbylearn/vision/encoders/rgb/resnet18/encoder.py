"""
ResNet18 image encoder (as in pointW/sym_in_dp), as a Backbone (rbylearn/vision/encoders/backbone.py). Registered in
rbylearn/vision/encoders/rgb/__init__.py as "resnet18", "resnet18-frozen" and "resnet18-untrained". The network itself is
sym_in_dp's TimmObsEncoder, copied into timm_obs_encoder.py (see the DISCLAIMER there).
"""
import contextlib
from pathlib import Path
import torch
import torchvision.transforms as T

from rbylearn.vision.encoders.backbone import Backbone
from rbylearn.vision.encoders.rgb.resnet18.timm_obs_encoder import TimmObsEncoder, TrainRandomTestCenterCrop, make_shape_meta, RGB_KEY

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FREEZE_ATTENTION_POOL = False  # whether to freeze the attention pool weights (for reproducibility)

def resnet18(pretrained: bool, frozen: bool, image_size: int = 84, crop_size: int = 76, downsample_ratio: int = 32, use_group_norm: bool = True, seed: int | None = 0) -> Backbone:    
    """
    Description
    -----------
    This is supposed to be a replacmenet for the ResNet18 backbone used in pointW/sym_in_dp's for frame averaging (train_diffusion_unet_pretrained_res18_fa_rel_traj .yaml) as closely as this module's Backbone shape allows: 
    timm's 'resnet18.tv_in1k' tag (NOT torchvision's resnet18 -- timm tags different training recipes separately, so this is a different checkpoint even for "the same architecture"
    BatchNorm replaced with GroupNorm,
    ImageNet normalization, and AttentionPool2d for the spatial-to-vector step.

    Parameters
    ----------
    pretrained : bool
        Whether to load the timm checkpoint for 'resnet18.tv_in1k' (pretrained=True) or to use the architecture with random weights (pretrained=False).
    frozen : bool
        Whether to freeze the weights of the backbone and AttentionPool2d (frozen=True) or to allow them to be trained (frozen=False). 
        In the original pointW/sym_in_dp pipeline, the backbone is NOT frozen and so the ResNet18 and AttentionPool2d weights are fine-tuned while training the policy.
        However, here we also allow for frozen=True, which is useful for reproducibility and for using the backbone as a fixed feature extractor. 
    image_size : int
        The size of the square input image that this backbone expects. The default is 76, which matches the crop size used in pointW/sym_in_dp's yaml. 
        The AttentionPool2d's positional embedding is fixed at construction time based on this size. 
        Note that this is not an actual image crop -- the input image should be resized/cropped upstream to this size before being passed to the backbone. 
    """

    ### Step 1: This creates exactly the same ResNet18 encoder as in pointW/sym_in_dp, with the same pretrained weights, GroupNorm, and AttentionPool2d. 
    # NOTE: that this backbone does not include anything about frame averaging.
    
    # Seed the random initialization (attention pool, and the ResNet itself when untrained) without changing the global random number generator. seed=None gives a fresh random init.
    rng = torch.random.fork_rng(devices=[]) if seed is not None else contextlib.nullcontext()

    with rng:
        shape_meta = make_shape_meta()
        if seed is not None:
            torch.manual_seed(seed)        
        # TimmObsEncoder expects diffusion-policy shape_meta. It always reads robot0_eef_pos's horizon,
        # so we declare a dummy low-dim key and mark it ignore_by_policy so it doesn't end up in the features.
        encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="resnet18.tv_in1k",
            pretrained=pretrained,
            frozen=frozen,
            global_pool="",
            transforms=[
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                TrainRandomTestCenterCrop(crop_size),  # center crop in eval mode, random crop in train mode
            ],
            imagenet_norm=True,  # unused by TimmObsEncoder (also in sym_in_dp); normalization happens in transforms        
            use_group_norm=use_group_norm,
            share_rgb_model=False,
            feature_aggregation="attention_pool_2d",
            downsample_ratio=downsample_ratio,
            frozen_attention_embedding=FREEZE_ATTENTION_POOL,  # fixed attention-pool init, so dataset and inference encoders match
        ).eval()

    ### Step 2: Define the preprocessing and embedding functions that will be used by the Backbone object. This is not part of the original pointW/sym_in_dp pipeline, but are needed for this codebase's Backbone interface.
    # One image (3, H, W) in -> one output out; a batch (B, 3, H, W) in -> a batch out (as the point-cloud encoders).
    def preprocess(image):
        x = T.functional.to_tensor(image)  # PIL -> (3, H, W) in [0, 1], same range as sym_in_dp's dataset (/255)
        return T.functional.resize(x, [image_size, image_size], antialias=True)  # (3, image_size, image_size); the encoder normalizes and center-crops

    def embed_spatial(x):
        batch = x if x.dim() == 4 else x.unsqueeze(0)
        out = encoder.key_model_map[RGB_KEY](encoder.key_transform_map[RGB_KEY](batch))  # (B, D, H', W')
        return out if x.dim() == 4 else out[0]

    def embed_pooled(x):
        batch = x if x.dim() == 4 else x.unsqueeze(0)
        out = encoder({RGB_KEY: batch.unsqueeze(1)})  # (B, T=1, C, H, W) -> (B, D)
        return out if x.dim() == 4 else out[0]

    return Backbone("resnet18", preprocess, embed_pooled, embed_spatial, IMAGENET_MEAN, IMAGENET_STD, module=encoder)
