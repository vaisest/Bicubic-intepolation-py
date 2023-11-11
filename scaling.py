import argparse
import concurrent.futures
import math
import pathlib
import time
from functools import cache
from itertools import repeat
from typing import Callable, TypeVar
from functools import partial

import cv2 as cv
import numpy as np
import skimage
import matplotlib.pyplot as plt


def bilinear(s: float) -> float:
    # linear kernel for bilinear
    s = abs(s)
    if 0 <= s < 1:
        return 1 - s
    return 0.0


def nn(s: float) -> float:
    # nearest neighbor
    if -0.5 <= s < 0.5:
        return 1.0
    return 0.0


# cache seems to speed this up enough to be worth
@cache
def bicubic(s: float) -> float:
    # bicubic convolution kernel aka catmull-rom spline
    # the value of a here is -0.5 as that was used in Keys' version
    a: float = -0.75
    s = abs(s)
    if 0 <= s < 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 <= s < 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    return 0.0


@cache
def mn(B: float, C: float, x: float) -> float:
    x = abs(x)

    if x < 1:
        return (1 / 6) * (
            (12 - 9 * B - 6 * C) * x**3
            + (-18 + 12 * B + 6 * C) * x**2
            + (6 - 2 * B)
        )
    elif 1 <= x < 2:
        return (1 / 6) * (
            (-B - 6 * C) * x**3
            + (6 * B + 30 * C) * x**2
            + (-12 * B - 48 * C) * x
            + (8 * B + 24 * C)
        )

    return 0.0


def mitchell_netravali(B: float, C: float) -> Callable[[float], float]:
    # mitchell netravali filter which can reprsent
    # many different cubic splines based on B and C
    # https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters

    # It should be noted that the catmull-rom spline too
    # can be represented with this with values B=0 and C=0.5.
    # More specifically when B=0, C is just the -a value in Keys' kernel u

    return partial(mn, B, C)


def lanczos(x: float) -> float:
    # lancozs kernel, which is more specifically
    # a sinc filter windowed to a smaller size (here 2)
    a = 2
    if -a < x < a:
        return float(np.sinc(x) * np.sinc(x / a))
    return 0.0


def plot_kernels(*kernels: Callable[[float], float]):
    import matplotlib.pyplot as plt

    xs = np.linspace(-4, 4, 500)
    for kernel in kernels:
        plt.plot(xs, [kernel(x) for x in xs], label=kernel.__name__)
    plt.xlim(-4, 4)
    plt.legend()
    plt.grid(True)
    plt.show()


T = TypeVar("T", int, float)


def clamp(n: T, a: T, b: T) -> T:
    # clamps n to [a, b]
    return min(b, max(n, a))


def scale_channel(
    image: np.ndarray, ratio: float, H: int, W: int, u: Callable[[float], float]
) -> np.ndarray:
    image = image.astype(np.float64) / 255

    # create new image
    new_H = math.floor(H * ratio)
    new_W = math.floor(W * ratio)
    big_image = np.zeros((new_H, new_W))

    for j in range(new_H):
        # scale new image's coordinate to be in old image based on its midpoint
        y = ((j + 0.5) / ratio) - 0.5
        # we separate x and y to integer and fractional parts
        iy = int(y)
        # ix and iy are essentially the closest original pixels
        # as all the old pixels are in integer positions
        # decx and decy as the fractional parts are then the (negative) distances
        # to the original pixels on the left and above
        decy = iy - y
        for i in range(new_W):
            x = ((i + 0.5) / ratio) - 0.5
            ix = int(x)
            decx = ix - x
            # print(i, x, ix, decx)
            # if i > 4:
            #     exit()

            # # It should be noted that bicubic is just cubic, but in two dimensions.
            # # So this can be calculated by interpolating four intermediate points in the x direction
            # # and then interpolating from those in the y direction:
            # p_1 = (
            #     image[iy - 1, ix + (-1)] * u(decx - 1)
            #     + image[iy - 1, ix] * u(decx)
            #     + image[iy - 1, ix + 1] * u(decx + 1)
            #     + image[iy - 1, ix + 2] * u(decx + 2)
            # )

            # p_2 = (
            #     image[iy, ix + (-1)] * u(decx - 1)
            #     + image[iy, ix] * u(decx)
            #     + image[iy, ix + 1] * u(decx + 1)
            #     + image[iy, ix + 2] * u(decx + 2)
            # )

            # p_3 = (
            #     image[iy + 1, ix + (-1)] * u(decx - 1)
            #     + image[iy + 1, ix] * u(decx)
            #     + image[iy + 1, ix + 1] * u(decx + 1)
            #     + image[iy + 1, ix + 2] * u(decx + 2)
            # )

            # p_4 = (
            #     image[iy + 2, ix + (-1)] * u(decx - 1)
            #     + image[iy + 2, ix] * u(decx)
            #     + image[iy + 2, ix + 1] * u(decx + 1)
            #     + image[iy + 2, ix + 2] * u(decx + 2)
            # )

            # pix = (
            #     p_1 * u(decy - 1)
            #     + p_2 * u(decy)
            #     + p_3 * u(decy + 1)
            #     + p_4 * u(decy + 2)
            # )

            pix = sum(
                sum(  # clamp indexes to source image range, but kernel u distances are not clamped
                    image[clamp(iy + M, 0, H - 1), clamp(ix + L, 0, W - 1)]
                    * u(decx + L)
                    * u(decy + M)
                    for L in range(-1, 2 + 1)
                )
                for M in range(-1, 2 + 1)
            )

            # we limit results to [0, 1] because bicubic interpolation
            # can produce pixel values outside the original range
            big_image[j, i] = clamp(pix, 0.0, 1.0)

    # without rounding there are various 1 pixel differences
    return (big_image * 255).round().astype(np.uint8)


def main(in_file: pathlib.Path, out_file: pathlib.Path, ratio: float):
    im_data = cv.imread(str(in_file))

    # # because plt uses rgb
    im_data = cv.cvtColor(im_data, cv.COLOR_RGB2BGR)

    start = time.perf_counter()
    print("Scaling image...")

    # plot_kernels(bicubic, l, nn, lanczos, mitchell_netravali(B=0, C=0.75))

    H, W, C = im_data.shape

    # pad by 2 px
    # im_data_p = cv.copyMakeBorder(im_data, 2, 2, 2, 2, cv.BORDER_REFLECT)

    # channels = cv.split(im_data_p)
    channels = cv.split(im_data)
    # channels = cv.split(im_data[:, :, 0])

    out_im_data: np.ndarray = np.zeros(1)

    # change kernel here
    kernel_to_use: Callable[[float], float] = bicubic

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
                    repeat(kernel_to_use),
                )
            )
        )

    print(f"Finished scaling in {time.perf_counter() - start} seconds")

    cv.imwrite(str(out_file), out_im_data)

    # plt.imshow(out_im_data)
    # plt.show()

    print(im_data.min(), im_data.max(), im_data.dtype, im_data.shape)
    print(out_im_data.min(), out_im_data.max(), out_im_data.dtype, out_im_data.shape)
    proper_cv = cv.resize(im_data, None, None, ratio, ratio, cv.INTER_CUBIC)
    proper_skimage = skimage.util.img_as_ubyte(
        skimage.transform.rescale(im_data, ratio, channel_axis=-1, order=3)
    )
    # # print(proper.min(), proper.max(), proper.dtype, proper.shape)

    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax[0, 0].imshow(im_data)
    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(out_im_data)
    ax[0, 1].set_title("My scale")

    ax[1, 0].set_title("Proper OpenCV")
    ax[1, 0].imshow(proper_cv)
    ax[1, 1].set_title("Proper Skimage")
    ax[1, 1].imshow(proper_cv)

    print("my scale vs proper_cv psnr:", cv.PSNR(out_im_data, proper_cv))

    ax[2, 0].set_title("Absdiff OpenCV vs My")
    diffy_cv = cv.absdiff(out_im_data, proper_cv)
    ax[2, 0].imshow(diffy_cv)
    ax[2, 1].set_title("Absdiff Skimage vs My")
    diffy_skimage = cv.absdiff(out_im_data, proper_skimage)
    ax[2, 1].imshow(diffy_skimage)

    ax[3, 0].set_title("diffy-cv nonzero locations")
    nz = np.zeros(diffy_cv.shape, np.uint8)
    for a, b, c in zip(*diffy_cv.nonzero()):
        # print(a, b, c)
        nz[a, b, c] = 255
    print(nz)
    ax[3, 0].imshow(nz)
    ax[3, 1].set_title("Absdiff CV vs Skimage")
    ax[3, 1].imshow(cv.absdiff(proper_cv, proper_skimage))

    print(
        "diffy_cv",
        diffy_cv.min(),
        diffy_cv.max(),
        diffy_cv.dtype,
        diffy_cv.shape,
        diffy_cv.nonzero(),
    )
    print(
        "diffy_skimage",
        diffy_skimage.min(),
        diffy_skimage.max(),
        diffy_skimage.dtype,
        diffy_skimage.shape,
    )
    print(
        "proper_skimage vs proper_opencv psnr:",
        cv.PSNR(out_im_data, proper_cv),
        cv.absdiff(proper_cv, proper_skimage).max(),
    )
    plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Python bicubic interpolation",
        description="This program implements the bicubic convolution interpolation image scaling algorithm.",
    )
    arg_parser.add_argument("in_file", type=pathlib.Path)
    arg_parser.add_argument("out_file", type=pathlib.Path, nargs="?")
    arg_parser.add_argument("scaling_ratio", type=float, nargs="?", default=2.0)

    args = arg_parser.parse_args()

    main(args.in_file, args.out_file, args.scaling_ratio)
