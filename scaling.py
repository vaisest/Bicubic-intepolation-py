import argparse
import concurrent.futures
import math
import pathlib
import time
from functools import cache
from itertools import repeat
from typing import Callable
from functools import partial

import cv2 as cv
import numpy as np


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
    a: float = -0.5
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
    # More specifically when B=0, C is just the -a value in the catmull-rom spline

    return partial(mn, B, C)


def lanczos(x: float) -> float:
    # lancozs kernel, which is more specifically
    # a sinc filter windowed to a smaller size (here 2)
    # Causes box artefacts for some reason. I am unable to find the reason.
    a = 2
    if -a < x < a:
        return float(np.sinc(x) * np.sinc(x / a))
    return 0.0


def plot_kernels(*kernels: Callable[[float], float]):
    import matplotlib.pyplot as plt

    xs = np.linspace(-4, 4, 500)
    for kernel in kernels:
        func_name = kernel.__name__ if hasattr(kernel, "__name__") else "unknown"
        plt.plot(xs, [kernel(x) for x in xs], label=func_name)
    plt.xlim(-4, 4)
    plt.legend()
    plt.grid(True)
    plt.show()


def scale_channel(
    image: np.ndarray, ratio: float, u: Callable[[float], float]
) -> np.ndarray:
    image = image.astype(np.float64) / 255

    H, W = image.shape
    # create new image
    new_H = math.floor((H - 4) * ratio)
    new_W = math.floor((W - 4) * ratio)
    big_image = np.zeros((new_H, new_W))

    for j in range(new_H):
        # scale new image's coordinate to be in old image based on its midpoint
        y = ((j + 0.5) / ratio) - 0.5 + 2
        # we separate x and y to integer and fractional parts
        iy = int(y)
        # ix and iy are essentially the closest original pixels
        # as all the old pixels are in integer positions
        # decx and decy as the fractional parts are then the (negative) distances
        # to the original pixels on the left and above
        decy = iy - y
        for i in range(new_W):
            x = ((i + 0.5) / ratio) - 0.5 + 2
            ix = int(x)
            decx = ix - x

            pix = sum(
                image[iy + M, ix + L] * u(decx + L) * u(decy + M)
                for L in range(-1, 2 + 1)
                for M in range(-1, 2 + 1)
            )

            # # It should be noted that bicubic is just cubic, but in two dimensions.
            # # So this can be calculated by interpolating four intermediate points in the x direction
            # ps = [
            #     sum(
            #         image[clamp(iy + M, 0, H - 1), clamp(ix + L, 0, W - 1)]
            #         * u(decx + L)
            #         for L in range(-1, 2 + 1)
            #     )
            #     for M in range(-1, 2 + 1)
            # ]

            # # and then interpolating from those in the y direction:
            # pix = (
            #     ps[0] * u(decy - 1)
            #     + ps[1] * u(decy)
            #     + ps[2] * u(decy + 1)
            #     + ps[3] * u(decy + 2)
            # )

            big_image[j, i] = pix

    # we limit results to [0, 1] because bicubic interpolation
    # can produce pixel values outside the original range
    # and without rounding there are various 1 pixel differences
    return (np.clip(big_image, 0.0, 1.0) * 255).round().astype(np.uint8)
    # return (big_image * 255).round().astype(np.uint8)


def main(in_file: pathlib.Path, out_file: pathlib.Path, ratio: float):
    im_data = cv.imread(str(in_file))

    # # because plt uses rgb
    # im_data = cv.cvtColor(im_data, cv.COLOR_RGB2BGR)

    start = time.perf_counter()

    # plot_kernels(bicubic, bilinear, nn, lanczos, mitchell_netravali(B=0, C=0.75))

    H, W, C = im_data.shape

    print(f"Scaling image from {W}x{H} to {int(W*ratio)}x{int(H*ratio)}...")

    padded = cv.copyMakeBorder(im_data, 2, 2, 2, 2, cv.BORDER_REPLICATE)

    channels = cv.split(padded)

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
                    repeat(kernel_to_use),
                )
            )
        )

    # # single thread
    # out_im_data = cv.merge(
    #     list(scale_channel(channels[c], ratio, kernel_to_use) for c in range(C))
    # )

    print(f"Finished scaling in {time.perf_counter() - start} seconds")

    cv.imwrite(str(out_file), out_im_data)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Python bicubic interpolation",
        description="This program implements the bicubic convolution interpolation image scaling algorithm.",
    )
    arg_parser.add_argument("in_file", type=pathlib.Path)
    arg_parser.add_argument("out_file", type=pathlib.Path, nargs="?")
    arg_parser.add_argument("-s", "--scale", type=float, nargs="?", default=2.0)

    args = arg_parser.parse_args()

    main(args.in_file, args.out_file, args.scale)
