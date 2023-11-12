# Bicubic interpolation in Python and Rust

This repository contains a image scaling using the bicubic convolution interpolation algorithm written in Python for learning purposes.

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
