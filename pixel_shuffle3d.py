import torch.nn as nn
import torch

class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)

class PixelUnshuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_unshuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-3] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects depth to be divisible by downscale_factor, but input.size(-3)={x.shape[-3]} is not divisible by {self.upscale_factor}'
            )
        elif x.shape[-2] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)={x.shape[-2]} is not divisible by {self.upscale_factor}'
            )
        elif x.shape[-1] % self.upscale_factor != 0:
            raise RuntimeError(
                f'pixel_unshuffle expects width to be divisible by downscale_factor, but input.size(-1)={x.shape[-1]} is not divisible by {self.upscale_factor}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]

        out_depth = in_depth // self.upscale_factor
        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor
        nOut = channels * self.upscale_factor**3

        input_view = x.contiguous().view(
            *x.shape[:-4],
            channels,
            out_depth,
            self.upscale_factor,
            out_height,
            self.upscale_factor,
            out_width,
            self.upscale_factor
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-5, -3, -1, -6, -4, -2]
        output = input_view.permute(axes).contiguous()
        
        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)