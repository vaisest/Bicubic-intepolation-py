use image::io::Reader as ImgReader;
use image::{DynamicImage, Pixel, Rgb, Rgb32FImage};
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "Bicubic")]
struct Opt {
    #[structopt(parse(from_os_str))]
    input_path: PathBuf,

    #[structopt(parse(from_os_str))]
    output_path: PathBuf,

    #[structopt(short = "s", long = "scale", default_value = "2")]
    ratio: f32,
}

fn pad_img(img: &Rgb32FImage, n: u32) -> Rgb32FImage {
    // Reflect border

    let dest = Rgb32FImage::from_fn(img.width() + 2 * n, img.height() + 2 * n, |x, y| {
        let xr: u32;
        if x < n {
            xr = (n - 1) - x;
        } else if x >= img.width() + n {
            // reflect back from original image based on how much over the edge of original image
            xr = img.width() - (1 + (x - n) - img.width());
        } else {
            xr = x - n;
        }

        let yr: u32;
        if y < n {
            yr = (n - 1) - y;
        } else if y >= img.height() + n {
            yr = img.height() - (1 + (y - n) - img.height());
        } else {
            yr = y - n;
        }

        return *img.get_pixel(xr, yr);
    });
    return dest;
}

fn bicubic(_s: f32) -> f32 {
    let a = -0.5f32;
    let s = _s.abs();
    if 0.0 <= s && s < 1.0 {
        return (a + 2.0) * s.powf(3.0) - (a + 3.0) * s.powf(2.0) + 1.0;
    } else if 1.0 <= s && s < 2.0 {
        return a * s.powf(3.0) - 5.0 * a * s.powf(2.0) + 8.0 * a * s - 4.0 * a;
    }
    return 0.0;
}

fn clamp(n: f32) -> f32 {
    // clamps n to [0, 1]
    return 0.0f32.max(1.0f32.min(n));
}

fn scale<F>(img: &Rgb32FImage, ratio: f32, w: u32, h: u32, u: F) -> Rgb32FImage
where
    F: Fn(f32) -> f32,
{
    let new_w = ((w as f32) * ratio).floor() as u32;
    let new_h = ((h as f32) * ratio).floor() as u32;

    let dest = Rgb32FImage::from_fn(new_w, new_h, |i, j| {
        let y = (j as f32 - 0.5) * (1.0 / ratio) + 2.0;
        let iy = y.trunc() as i32;
        println!("{:?}, {:?}", y, iy);
        let decy = y.trunc() - y;

        let x = (i as f32 - 0.5) * (1.0 / ratio) + 2.0;
        let ix = x.trunc() as i32;
        let decx = x.trunc() - x;

        // essentially the following
        // pix = sum(
        //         sum(
        //             image[iy + M, ix + L] * u(decx - L) * u(decy - M)
        //             for L in range(-1, 2 + 1)
        //         )
        //         for M in range(-1, 2 + 1)
        //     )

        let pix: Rgb<f32> = (-1i32..=2)
            .map(|m: i32| {
                return (-1i32..=2)
                    .map(|l| {
                        let p = img
                            .get_pixel((ix + l) as u32, (iy + m) as u32)
                            .map(|v| v * u(decx + l as f32) * u(decy + m as f32));
                        return p;
                    })
                    .fold(Rgb([0.0, 0.0, 0.0]), |a, b| {
                        return Rgb([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
                    });
            })
            .fold(Rgb([0.0, 0.0, 0.0]), |a, b| {
                return Rgb([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
            })
            .map(clamp);

        return pix;
    });
    return dest;
}

fn main() {
    println!("Hello, world!");
    let opt = Opt::from_args();

    println!("Scaling image...");
    let timer = Instant::now();

    let input_img = ImgReader::open(&opt.input_path)
        .expect("Could not load input image")
        .decode()
        .expect("Could not decode image")
        .into_rgb32f();

    let w = input_img.width();
    let h = input_img.height();

    let padded = pad_img(&input_img, 2);
    let scaled = scale(&padded, opt.ratio, w, h, bicubic);

    let proper = ImgReader::open(&opt.input_path)
        .expect("asd")
        .decode()
        .expect("osd")
        .resize(
            (w as f32 * opt.ratio).floor() as u32,
            (h as f32 * opt.ratio).floor() as u32,
            image::imageops::FilterType::CatmullRom,
        );

    println!(
        "Finished scaling in {:?} seconds",
        timer.elapsed().as_secs_f32()
    );

    (DynamicImage::ImageRgb32F(scaled).into_rgb8())
        .save(opt.output_path)
        .expect("Failed to write output scaled image");
    (proper.into_rgb8())
        .save("../out_proper.png")
        .expect("Failed to write output scaled image");
}
