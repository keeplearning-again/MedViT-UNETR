# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import pdb
from collections.abc import Sequence
import torch
import torch.nn as nn
from functools import partial
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock, UnetResBlock
# from monai.networks.nets.vit import ViT
from vit import ViT
from adapter import SpatialPriorModule, InteractionBlock, deform_inputs
from monai.utils import deprecated_arg, ensure_tuple_rep
from torch.nn.init import normal_
from timm.models.layers import trunc_normal_
from ops.modules import MSDeformAttn
import math
class UNETR_adapter(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        conv_inplane: int = 64,
        deform_num_heads: int = 6,
        n_points: int = 4,
        init_values: float = 0.,
        drop_path_rate: float = 0.4,
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cp: bool = False,
        with_cffn: bool = True,
        use_extra_extractor: bool = True,
        add_vit_feature: bool = True,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: patch embedding layer type. Defaults to "conv".
            norm_name: feature normalization type and arguments. Defaults to "instance".
            conv_block: if convolutional block is used. Defaults to True.
            res_block: if residual block is used. Defaults to True.
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dims. Defaults to 3.
            qkv_bias: apply the bias term for the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False.
            conv_inplane: for adapter, the input plane dim for the spatial prior module. Defaults to 64.
            deform_num_heads: number of attention heads for the deformable attention. Defaults to 6.
            n_points: number of sampling points for the deformable attention. Defaults to 4.
            init_values: initial value for the injector. Defaults to 0.
            drop_path_rate: drop path rate. Defaults to 0.4.
            cffn_ratio: ratio of the cffn layer. Defaults to 0.25.
            deform_ratio: ratio of the deformable attention. Defaults to 0.5.
            interaction_indexes: indexes for the interaction block. Defaults to [[0, 2], [3, 5], [6, 8], [9, 11]].
            with_cp: save memory. Defaults to False.
            with_cffn: if the cffn layer is used. Defaults to True.
            use_extra_extractor: if the extra extractor is used. Defaults to True.
            add_vit_feature: if the vit feature is added. Defaults to True.


        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_layers = 12 # vit-base 12 layers
        self.drop_path_rate = drop_path_rate 
        self.img_size = ensure_tuple_rep(img_size, spatial_dims) # (224, 224, 224)
        # self.img_size = (224, 224, 64)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims) # (16, 16, 16)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.img_size, self.patch_size)) # (14, 14, 14)
        self.hidden_size = hidden_size # sequence dim == 768
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.add_vit_feature = add_vit_feature
        if self.patch_size[0] == 8:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 2,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        else:
            self.encoder0 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder1 = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 2,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 4,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 16,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )

            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size,
                out_channels=feature_size * 16,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 16,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size * 1,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        self.up = nn.ConvTranspose3d(hidden_size, hidden_size, 2, 2)
        # adapter for the ViT model
        self.spm = SpatialPriorModule(input_dim=in_channels, inplanes=conv_inplane, embed_dim=hidden_size, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=hidden_size, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.interaction_indexes = interaction_indexes
        self.level_embed = nn.Parameter(torch.zeros(3, hidden_size))

        self.norm1 = nn.SyncBatchNorm(hidden_size)
        self.norm2 = nn.SyncBatchNorm(hidden_size)
        self.norm3 = nn.SyncBatchNorm(hidden_size)
        self.norm4 = nn.SyncBatchNorm(hidden_size)
        # self.norm5 = nn.SyncBatchNorm(hidden_size)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()  

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        deform_inputs1, deform_inputs2 = deform_inputs(x_in)
        _, _, H, W, D = x_in.shape
        H, W, D = H // 16, W // 16, D // 16
        c1, c2, c3, c4 = self.spm(x_in)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        x = self.vit.patch_embedding(x_in)
        bs, n, dim = x.shape
        if hasattr(self.vit, "cls_token"):
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.vit.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W, D)
            outs.append(x.transpose(1, 2).view(bs, dim, D, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        # add vit feature !!!

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2, D * 2).contiguous() # 8s
        c3 = c3.transpose(1, 2).view(bs, dim, H, W, D).contiguous() # 16s
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2, D // 2).contiguous() # 32s
        c2 = c2.permute(0, 1, 4, 2, 3)
        c1 = self.up(c2) + c1 # 4s
        c1 = c1.permute(0, 1, 3, 4, 2)
        c2 = c2.permute(0, 1, 3, 4, 2)
        c4 = self.norm4(c4)
        c3 = self.norm3(c3)
        c2 = self.norm2(c2)
        c1 = self.norm1(c1)
        enc2 = self.encoder1(c1) # 2s
        enc1 = self.encoder0(x_in) # 1
        enc3 = self.encoder2(c1) # 4s
        # import pdb; pdb.set_trace()
        enc4 = self.encoder3(c2) # 8s
        enc5 = self.encoder4(c3) # 16s
        # enc6 = self.encoder5(c4) # 32s
        dec5 = self.decoder5(c4, enc5) # 16s
        dec3 = self.decoder4(dec5, enc4) # 8s
        dec2 = self.decoder3(dec3, enc3) # 4s
        dec1 = self.decoder2(dec2, enc2) # 2s
        out = self.decoder1(dec1, enc1) # 1
        return self.out(out)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.utils.checkpoint as cp

    from ops.modules import MSDeformAttn
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    import torch.distributed as dist
    import torch.multiprocessing as mp
    dist.init_process_group(backend='nccl', init_method='env://')
    x = torch.randn(1, 1, 96, 96, 64).cuda()
    model = UNETR_adapter(in_channels=1, out_channels=1, img_size=(96, 96, 64), patch_size=(16, 16, 16)).cuda()
    out = model(x)
    print(out.shape)