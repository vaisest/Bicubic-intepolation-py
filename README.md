# Bicubic interpolation in Python and Rust

This repository contains an image scaling implementation using the bicubic convolution interpolation algorithm written in Python for learning purposes. As the Python version is unusably slow, there is also a Rust implementation included that is quite fast and which might be useful for quickly scaling a large amount of images. Using `RUSTFLAGS='-C target-cpu=native` might be useful as there is no manual SIMD usage, but the Rust code works well with auto vectorization, resulting in doubled performance in my tests.

The method was originally introduced in [Cubic convolution interpolation for digital image processing](https://ieeexplore.ieee.org/document/1163711) by R. Keys in 1981. The method works using the function
$$g(x, y) = \sum_{l=-1}^2 \sum_{m=-1}^2 c_{i + l,j + m} u(dx + l) u(dy + m),$$
which produces a new value for a position $(x, y)$ in the new image by scaling these coordinates back to the dimensions of the source image as $(i, j)$, and then by summing the nearest 16 pixels. The weights are calculated using each pixel's distance using the function $u$ which is known as the interpolation kernel. There are multiple kernels available, but for example the one used in Keys' research was 
```math
u(s) = \begin{cases}
        \frac{3}{2} \left| s \right|^3 - \frac{5}{2} \left| s \right|^2 + 1                       & 0 \leq \left| s \right| < 1 \\
        -\frac{1}{2} \left| s \right|^3 + \frac{5}{2} \left| s \right|^2 - 4 \left| s \right| + 2 & 1 \leq \left| s \right| < 2 \\
        0                                                                                         & \text{else}.
    \end{cases}
```
Some other kernels like the Mitchell-Netravali filter are also implemented. Generally different kernels affect the sharpness of the produced image. For more information on reconstruction filters / kernels see [ImageMagick's documentation](https://imagemagick.org/Usage/filter/).
<!-- ```txt
$ echo "1280x720 -> 2560x1440"; hyperfine --warmup 1 'python ../scaling.py -s 2 ../test_720p_wp.png ../out_py.png'
1280x720 -> 2560x1440
Benchmark 1: python ../scaling.py -s 2 ../test_720p_wp.png ../out_py.png
  Time (mean ± σ):     36.361 s ±  0.467 s    [User: 83.620 s, System: 0.728 s]
  Range (min … max):   35.881 s … 37.533 s    10 runs
```

```txt
$ echo "1280x720 -> 2560x1440"; hyperfine --warmup 1 '.\target\release\bicubic_rs.exe -s 2 ../test_720p_wp.png ../out_rs.png'              
1280x720 -> 2560x1440
Benchmark 1: .\target\release\bicubic_rs.exe -s 2 ../test_720p_wp.png ../out_rs.png
  Time (mean ± σ):     625.0 ms ±   4.3 ms    [User: 493.8 ms, System: 9.4 ms]
  Range (min … max):   619.3 ms … 632.9 ms    10 runs
```
60 times faster
Make sure to use `RUSTFLAGS='-C target-cpu=native'` as at least on my system (5800X3D) this more than doubles execution speed.
-->
