
import pixel_shuffle


obj1 = pixel_shuffle.PixelShuffle3d()

print('END')

# import torch.nn as nn

# class PixelShuffle3d(nn.Module):
#     def __init__(self, upscale_factor=None):
#         super().__init__()

#         print(upscale_factor)

#         if upscale_factor is None:
#             raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')
#         else:
#             print('OK')