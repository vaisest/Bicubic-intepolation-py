from itertools import repeat
import math
import signal
import cv2 as cv
import numpy as np

import pathlib
import argparse
from typing import cast, Callable
from multiprocessing import Pool
from functools import cache
import time
import concurrent.futures


# bicubic convolution kernel aka catmull-rom spline
# the value of a here is -0.5 as that was used in Keys' version
def u(s: float, a: float = -0.5):
    # print(s)
    s = abs(s)
    if 0 <= s < 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 <= s < 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    return 0


def scale_channel(
    image: np.ndarray, ratio: float, H: int, W: int, u: Callable[[float], float]
) -> np.ndarray:
    # image = image / 255
    orig_min = image.min()
    orig_max = image.max()

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

            # original two-dimensional method from paper
            # where we go through e.g. horizontally
            #                                               (ix-1), ix, (ix+1), (ix+2)
            # this means the distances used in the kernel will be
            #                                           (decx-1), decx, (decx + 1), (decx + 2)
            # [
            #     print([(L - decx, M - decy) for L in range(-1, 2 + 1)])
            #     for M in range(-1, 2 + 1)
            # ]
            big_image[j, i] = sum(
                sum(
                    image[iy + L, ix + L] * u(M - decx) * u(L - decy)
                    for M in range(-1, 2 + 1)
                )
                for L in range(-1, 2 + 1)
            )
    current_min = big_image.min()
    current_max = big_image.max()

    big_image = (
        ((big_image - current_min + orig_min) / current_max) * orig_max
    ).astype(np.uint8)

    # x, y = i * (1 / ratio) + 2, j * (1 / ratio) + 2

    # x1 = 1 + x - math.floor(x)
    # x2 = x - math.floor(x)
    # x3 = math.floor(x) + 1 - x
    # x4 = math.floor(x) + 2 - x

    # y1 = 1 + y - math.floor(y)
    # y2 = y - math.floor(y)
    # y3 = math.floor(y) + 1 - y
    # y4 = math.floor(y) + 2 - y

    # mat_l = np.matrix([[u(x1), u(x2), u(x3), u(x4)]])
    # mat_m = np.matrix(
    #     [
    #         [
    #             image[int(y - y1), int(x - x1)],
    #             image[int(y - y2), int(x - x1)],
    #             image[int(y + y3), int(x - x1)],
    #             image[int(y + y4), int(x - x1)],
    #         ],
    #         [
    #             image[int(y - y1), int(x - x2)],
    #             image[int(y - y2), int(x - x2)],
    #             image[int(y + y3), int(x - x2)],
    #             image[int(y + y4), int(x - x2)],
    #         ],
    #         [
    #             image[int(y - y1), int(x + x3)],
    #             image[int(y - y2), int(x + x3)],
    #             image[int(y + y3), int(x + x3)],
    #             image[int(y + y4), int(x + x3)],
    #         ],
    #         [
    #             image[int(y - y1), int(x + x4)],
    #             image[int(y - y2), int(x + x4)],
    #             image[int(y + y3), int(x + x4)],
    #             image[int(y + y4), int(x + x4)],
    #         ],
    #     ]
    # )
    # mat_r = np.matrix([[u(y1)], [u(y2)], [u(y3)], [u(y4)]])
    # big_image[j, i] = np.dot(np.dot(mat_l, mat_m), mat_r)

    # print()
    # big_image = cv.normalize(big_image, None, 255, 0, cv.NORM_MINMAX, cv.CV)
    return big_image


def main(in_file: pathlib.Path, out_file: pathlib.Path, ratio: float):
    im_data = cv.imread(str(in_file))

    H, W, C = im_data.shape

    # pad by 2 px
    im_data_p = cv.copyMakeBorder(im_data, 2, 2, 2, 2, cv.BORDER_REFLECT)

    # docs say this is slow, but it takes 4 ms for a 3840x2160 image
    # which doesnt sound so slow
    channels = cv.split(im_data_p)

    out_im_data: np.ndarray | None = None

    start = time.perf_counter()
    print("Scaling image...")

    # scaling images with big sizes can take a long time
    # and with how slow this implementation is
    # it makes sense to scale each channel separately at the same time

    # https://github.com/python/cpython/issues/66587
    with concurrent.futures.ProcessPoolExecutor(max_workers=C) as executor:
        out_im_data = cv.merge(
            list(
                executor.map(
                    scale_channel,
                    # [channels[0]],
                    channels,
                    repeat(ratio),
                    repeat(H),
                    repeat(W),
                    repeat(u),
                )
            )
        )

    # with Pool(C) as p:
    #     result = p.starmap_async(
    #         scale_channel,
    #         zip(
    #             [channels[0]],
    #             repeat(ratio),
    #             repeat(H),
    #             repeat(W),
    #             repeat(u),
    #         ),
    #     )
    #     # https://www.reddit.com/r/learnpython/comments/152sfp8/how_to_stop_multiprocessingpool_with_ctrlc_python/
    #     while not result.ready():
    #         time.sleep(0.2)
    #     out_im_data = cv.merge(result.get(10))
    #     p.terminate()

    print(im_data.min(), im_data.max(), im_data.dtype, im_data.shape)
    print(out_im_data.min(), out_im_data.max(), out_im_data.dtype, out_im_data.shape)
    proper = cv.resize(im_data, None, None, 2, 2, cv.INTER_CUBIC)
    print(proper.min(), proper.max(), proper.dtype, proper.shape)

    print(f"Finished scaling in {time.perf_counter() - start} seconds")
    cv.imshow("Original image", im_data)
    cv.imshow("Scaled image", out_im_data)
    cv.imshow("Proper image", proper)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
