import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _get_in_channels(params: dict) -> int:
    """Resolve input-channel count. Priority: encoder.in_channels,
    then len(data.field_keys), else 1."""
    enc = params['density']['encoder']
    if 'in_channels' in enc:
        return enc['in_channels']
    data = params['density'].get('data', {})
    if 'field_keys' in data:
        return len(data['field_keys'])
    return 1


def build_encoder(params: dict) -> nn.Module:
    """Factory: dispatches on params['density']['encoder']['type']."""
    cfg = params['density']['encoder']
    in_channels = _get_in_channels(params)
    encoder_type = cfg['type']
    if encoder_type == 'cnn3d':
        return ConvNet3DDensity(cfg, in_channels=in_channels)
    elif encoder_type == 'swin3d':
        return Swin3DEncoder(cfg, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class ConvNet3DDensity(nn.Module):
    """3D CNN encoder for density fields (any cubic resolution).

    4 conv blocks (1->32->64->128->256, kernel=3, stride=2) -> global avg pool
    -> Linear projection to summary_dim.
    """

    def __init__(self, cfg: dict, in_channels: int = 1):
        super().__init__()
        summary_dim = cfg['summary_dim']
        use_checkpointing = cfg.get('use_checkpointing', False)
        self.use_checkpointing = use_checkpointing

        self.blocks = nn.ModuleList([
            self._conv_block(in_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        ])
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(256, summary_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"ConvNet3DDensity: in_channels={in_channels}, summary_dim={summary_dim}, params={n_params:,}")

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.global_pool(x).flatten(1)
        x = self.projection(x)
        return x


class _CheckpointBlock(nn.Module):
    """Wraps a block for gradient checkpointing."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)


class Swin3DEncoder(nn.Module):
    """Swin3D encoder for density fields (any cubic resolution).

    Wraps torchvision swin3d_t, replaces stem conv for 1-channel input,
    replaces head with projection to summary_dim. Includes data augmentation
    (random 90-degree rotations + flips) during training.
    """

    def __init__(self, cfg: dict, in_channels: int = 1):
        super().__init__()
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights

        summary_dim = cfg['summary_dim']
        use_checkpointing = cfg.get('use_checkpointing', True)
        proj_dropout = cfg.get('proj_dropout', 0.0)

        # Build base swin3d_t model
        swin = swin3d_t(weights=None)

        # Replace stem conv to match input channel count
        old_conv = swin.patch_embed.proj
        swin.patch_embed.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        logging.info(f"Swin3D: modified patch_embed.proj for {in_channels}-channel input")

        # Replace head with Identity to get features
        original_head = swin.head
        swin.head = nn.Identity()
        if hasattr(original_head, 'in_features'):
            swin_output_dim = original_head.in_features
        else:
            swin_output_dim = swin.patch_embed.proj.out_channels * 8
        logging.info(f"Swin3D: output dim = {swin_output_dim}")

        # Apply gradient checkpointing
        if use_checkpointing:
            for stage in swin.features:
                if isinstance(stage, nn.Sequential):
                    for i in range(len(stage)):
                        stage[i] = _CheckpointBlock(stage[i])
            logging.info("Swin3D: gradient checkpointing enabled")

        self.swin = swin
        self.projection = nn.Linear(swin_output_dim, summary_dim)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Swin3DEncoder: summary_dim={summary_dim}, params={n_params:,}")

    def _augment(self, field):
        """Random augmentations preserving periodicity (training only)."""
        axes_pairs = [(2, 3), (2, 4), (3, 4)]
        for axes in axes_pairs:
            if torch.rand(1) < 0.5:
                k = torch.randint(1, 4, (1,)).item()
                field = torch.rot90(field, k=k, dims=axes)
        for dim in [2, 3, 4]:
            if torch.rand(1) < 0.5:
                field = torch.flip(field, dims=[dim])
        return field

    def forward(self, x):
        if self.training:
            x = self._augment(x)
        features = self.swin(x)
        output = self.proj_dropout(self.projection(features))
        return output
