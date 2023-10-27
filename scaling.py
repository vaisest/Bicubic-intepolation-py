from itertools import repeat
import math
import cv2 as cv
import numpy as np

import pathlib
import argparse
from typing import cast, Callable
from subprocess import Pool


# bicubic convolution kernel aka catmull-rom spline
# the value of a here is -0.5 as that was used in Keys' version
def u(s: float, a: float = -0.5):
    s = abs(s)

    if 0 <= s < 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 <= s < 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a

    return 0


def scale_channel(
    image: np.ndarray, ratio: float, H: int, W: int, u: Callable[[float], float]
):
    # create new image
    new_H = math.floor(H * ratio)
    new_W = math.floor(W * ratio)
    big_image = np.zeros((new_H, new_W))

    for j in range(new_H):
        for i in range(new_W):
            x = i * (1 / ratio) + 2
            y = j * (1 / ratio) + 2

            # we separate x and y to integer and fractional parts
            ix = int(x)
            decx = x - ix
            # ix and iy are essentially the closest original pixels
            # as all the old pixels are in integer positions
            iy = int(y)
            # decx and decy as the fractional parts are then the distances
            # to the original pixels on the left and above
            decy = y - iy

            # original two-dimensional method from paper
            # where we go through e.g. horizontally
            #                                               (ix-1), ix, (ix+1), (ix+2)
            # this means the distances used in the kernel will be
            #                                           (decx-1), decx, (decx + 1), (decx + 2)
            big_image[j, i] = sum(
                sum(
                    image[iy + M, ix + L] * u(decx + L) * u(decy + M)
                    for M in range(-1, 2 + 1)
                )
                for L in range(-1, 2 + 1)
            )

    return big_image


def main(in_file: pathlib.Path, out_file: pathlib.Path, ratio: float):
    im_data = cv.imread(in_file)

    H, W, C = im_data.shape

    # pad by 2 px
    im_data = cv.copyMakeBorder(im_data, 2, 2, 2, 2, cv.BORDER_REFLECT)

    # docs say this is slow, but it takes 4 ms for a 3840x2160 image
    # which doesnt sound slow in this context
    channels = cv.split(im_data)

    # scaling images with big sizes can take a long time
    # and with how slow this implementation is
    # it makes sense to scale each channel separately at the same time
    with Pool(C) as p:
        return cv.merge(
            p.starmap(
                scale_channel,
                zip(channels, repeat(ratio), repeat(H), repeat(W), repeat(u)),
            )
        )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Python bicubic interpolation",
        description="This program implements the bicubic convolution interpolation image scaling algorithm. Additionally other convolution kernels for e.g. Lanczos and bilinear are implemented.",
    )
    arg_parser.add_argument("in_file", type=pathlib.Path)
    arg_parser.add_argument("out_file", type=pathlib.Path)
    arg_parser.add_argument("scaling_ratio", type=float, nargs="?", default=2.0)

    args = arg_parser.parse_args()
    # argparse is not type hinted?
    main(args.in_file, args.out_file, args.scaling_ratio)
