import argparse
import concurrent.futures
import math
import pathlib
import time
from functools import cache
from itertools import repeat
from typing import Callable

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage


# cache seems to speed this up enough to be worth
@cache
def u(s: float):
    # bicubic convolution kernel aka catmull-rom spline
    # the value of a here is -0.5 as that was used in Keys' version
    a: float = -0.5
    s = abs(s)
    if 0 <= s < 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 <= s < 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    return 0


def scale_channel(
    image: np.ndarray, ratio: float, H: int, W: int, u: Callable[[float], float]
) -> np.ndarray:
    image = image.astype(np.float64) / 255

    # create new image
    new_H = math.floor(H * ratio)
    new_W = math.floor(W * ratio)
    big_image = np.zeros((new_H, new_W))

    for j in range(new_H):
        for i in range(new_W):
            x = i * (1 / ratio) + 2
            y = j * (1 / ratio) + 2

            # print(x, y, x - int(x), y - int(y))
            # we separate x and y to integer and fractional parts
            ix = int(x)
            decx = ix - x
            # ix and iy are essentially the closest original pixels
            # as all the old pixels are in integer positions
            iy = int(y)
            # decx and decy as the fractional parts are then the distances
            # to the original pixels on the left and above
            decy = iy - y

            # pix = sum(
            #     sum(
            #         image[iy + M, ix + L] * u(decx + L) * u(decy + M)
            #         for L in range(-1, 2 + 1)
            #     )
            #     for M in range(-1, 2 + 1)
            # )

            pix = sum(
                sum(
                    image[iy + M, ix + L] * u(decx + L) * u(decy + M)
                    for L in range(-1, 2 + 1)
                )
                for M in range(-1, 2 + 1)
            )

            # we limit results to [0, 1] because bicubic interpolation
            # can produce pixel values outside the original range
            big_image[j, i] = max(min(1, pix), 0)
            # big_image[j, i] = pix

    return (big_image * 255).astype(np.uint8)
    # return big_image


def main(in_file: pathlib.Path, out_file: pathlib.Path, ratio: float):
    im_data = cv.imread(str(in_file))

    # im_data = cv.cvtColor(im_data, cv.COLOR_RGB2BGR)

    start = time.perf_counter()
    print("Scaling image...")

    H, W, C = im_data.shape

    # pad by 2 px
    im_data_p = cv.copyMakeBorder(im_data, 2, 2, 2, 2, cv.BORDER_REFLECT)

    channels = cv.split(im_data_p)

    out_im_data: np.ndarray = np.zeros(1)

    # scaling images with big sizes can take a long time
    # and with how slow this implementation is
    # it makes sense to scale each channel separately at the same time

    # https://github.com/python/cpython/issues/66587
    with concurrent.futures.ProcessPoolExecutor(max_workers=C) as executor:
        out_im_data = cv.merge(
            list(
                executor.map(
                    scale_channel,
                    channels,
                    repeat(ratio),
                    repeat(H),
                    repeat(W),
                    repeat(u),
                )
            )
        )

    print(f"Finished scaling in {time.perf_counter() - start} seconds")

    # print(im_data.min(), im_data.max(), im_data.dtype, im_data.shape)
    # print(out_im_data.min(), out_im_data.max(), out_im_data.dtype, out_im_data.shape)
    # proper = cv.resize(im_data, None, None, ratio, ratio, cv.INTER_CUBIC)
    # # proper = skimage.util.img_as_ubyte(
    # #     skimage.transform.rescale(im_data, ratio, channel_axis=-1, order=3)
    # # )
    # # print(proper.min(), proper.max(), proper.dtype, proper.shape)

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0, 0].imshow(im_data)
    # ax[0, 0].set_title("Original")
    # ax[0, 1].imshow(out_im_data)
    # ax[0, 1].set_title("My scale")
    # ax[1, 0].set_title("Proper")
    # ax[1, 0].imshow(proper)
    # print(cv.PSNR(out_im_data, proper))
    # print(cv.PSNR(out_im_data, out_im_data))
    # ax[1, 1].set_title("Diffy")
    # # ldiffy = out_im_data.astype(np.float32) / 255 - proper.astype(np.float32) / 255
    # diffy = cv.absdiff(out_im_data, proper)

    # # print(diffy.min(), diffy.max(), diffy.dtype, diffy.shape)
    # ax[1, 1].imshow(diffy)
    # plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Python bicubic interpolation",
        description="This program implements the bicubic convolution interpolation image scaling algorithm.",
    )
    arg_parser.add_argument("in_file", type=pathlib.Path)
    arg_parser.add_argument("out_file", type=pathlib.Path)
    arg_parser.add_argument("scaling_ratio", type=float, nargs="?", default=2.0)

    args = arg_parser.parse_args()

    main(args.in_file, args.out_file, args.scaling_ratio)
