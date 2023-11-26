use image::io::Reader as ImgReader;
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba, Rgba32FImage};
use std::f32::consts::PI;
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

#[allow(dead_code)]
// nearest neighbor kernel
fn nn(s: f32) -> f32 {
    if -0.5 <= s && s < 0.5 {
        return 1.0;
    }
    return 0.0;
}

#[allow(dead_code)]
// linear kernel
fn bilinear(mut s: f32) -> f32 {
    s = s.abs();
    if 0.0 <= s && s < 1.0 {
        return 1.0 - s;
    }
    return 0.0;
}

#[allow(dead_code)]
// bicubic kernel, aka catmull-rom spline
fn bicubic(mut s: f32) -> f32 {
    let a = -0.5f32;
    s = s.abs();
    if 0.0 <= s && s < 1.0 {
        return (a + 2.0) * s.powf(3.0) - (a + 3.0) * s.powf(2.0) + 1.0;
    } else if 1.0 <= s && s < 2.0 {
        return a * s.powf(3.0) - 5.0 * a * s.powf(2.0) + 8.0 * a * s - 4.0 * a;
    }
    return 0.0;
}

#[allow(dead_code)]
fn mn(b: f32, c: f32, mut x: f32) -> f32 {
    x = x.abs();

    if x < 1.0 {
        return (1.0 / 6.0)
            * ((12.0 - 9.0 * b - 6.0 * c) * x.powi(3)
                + (-18.0 + 12.0 * b + 6.0 * c) * x.powi(2)
                + (6.0 - 2.0 * b));
    } else if x < 2.0 {
        return (1.0 / 6.0)
            * ((-b - 6.0 * c) * x.powi(3)
                + (6.0 * b + 30.0 * c) * x.powi(2)
                + (-12.0 * b - 48.0 * c) * x
                + (8.0 * b + 24.0 * c));
    } else {
        return 0.0;
    }
}

#[allow(dead_code)]
// mitchell-netravali filter
fn mitchell_netravali(b: f32, c: f32) -> impl Fn(f32) -> f32 {
    move |x| mn(b, c, x)
}

fn sinc(mut x: f32) -> f32 {
    // normalized sinc to be exact
    if x == 0.0 {
        return 1.0;
    }
    x *= PI;
    return x.sin() / x;
}
#[allow(dead_code)]
// lanczos kernel aka sinc filter, with window size 2
fn lanczos2(x: f32) -> f32 {
    let a = 2.0f32;
    if -a < x && x < a {
        return sinc(x) * sinc(x / a);
    } else {
        return 0.0;
    }
}

fn pad(img: &Rgba32FImage) -> Rgba32FImage {
    // pad by copying border
    let mut dest: Rgba32FImage = Rgba32FImage::new(img.width() + 4, img.height() + 4);

    for y in 0..2 {
        // top
        for x in 0..dest.width() {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel((x.saturating_sub(2)).clamp(0, img.width() - 1), 0),
            );
        }
    }
    for y in 2..img.height() {
        // left
        for x in 0..2 {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel(0, (y.saturating_sub(2)).clamp(0, img.height() - 1)),
            );
        }
        // right
        for x in img.width()..dest.width() {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel(
                    img.width() - 1,
                    (y.saturating_sub(2)).clamp(0, img.height() - 1),
                ),
            );
        }
    }
    for y in img.height()..dest.height() {
        // bottom
        for x in 0..dest.width() {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel(
                    (x.saturating_sub(2)).clamp(0, img.width() - 1),
                    img.height() - 1,
                ),
            );
        }
    }

    dest.copy_from(img, 2, 2).expect("pad panic");

    return dest;
}

// unsafe because input should be padded where oob is not possible
pub fn scale_padded<F>(img: &Rgba32FImage, ratio: f32, kernel: F) -> Rgba32FImage
where
    F: Fn(f32) -> f32,
{
    let new_w = (((img.width() - 4) as f32) * ratio) as u32;
    let new_h = (((img.height() - 4) as f32) * ratio) as u32;

    let mut dest = ImageBuffer::new(new_w, new_h);

    for j in 0..dest.height() {
        let y = (j as f32 + 0.5) * (1.0 / ratio) - 0.5 + 2.0;
        let iy = y as i32;
        let decy = y.trunc() - y;

        for i in 0..dest.width() {
            let x = (i as f32 + 0.5) * (1.0 / ratio) - 0.5 + 2.0;
            let ix = x as i32;
            let decx = x.trunc() - x;

            let mut pix = [0.0f32; 4];
            for m in -1i32..=2 {
                for l in -1i32..=2 {
                    let p: [f32; 4];
                    // Safe, source image is treated as a padded image
                    unsafe {
                        p = img.unsafe_get_pixel((ix + l) as u32, (iy + m) as u32).0;
                    }

                    pix[0] += p[0] * kernel(decx + l as f32) * kernel(decy + m as f32);
                    pix[1] += p[1] * kernel(decx + l as f32) * kernel(decy + m as f32);
                    pix[2] += p[2] * kernel(decx + l as f32) * kernel(decy + m as f32);
                    pix[3] += p[3] * kernel(decx + l as f32) * kernel(decy + m as f32);
                }
            }

            pix[0] = pix[0].clamp(0.0, 1.0);
            pix[1] = pix[1].clamp(0.0, 1.0);
            pix[2] = pix[2].clamp(0.0, 1.0);
            pix[3] = pix[3].clamp(0.0, 1.0);

            dest.put_pixel(i, j, Rgba(pix));
        }
    }
    return dest;
}

fn main() {
    let opt = Opt::from_args();

    let input_img = ImgReader::open(&opt.input_path)
        .expect("Could not load input image")
        .with_guessed_format()
        .expect("Could not determine input image format")
        .decode()
        .expect("Could not decode image")
        .into_rgba32f();

    println!(
        "Scaling image from {:?}x{:?} to {:?}x{:?}...",
        input_img.width(),
        input_img.height(),
        (input_img.width() as f32 * opt.ratio) as u32,
        (input_img.height() as f32 * opt.ratio) as u32
    );

    let timer = Instant::now();

    let padded = pad(&input_img);
    drop(input_img);
    let scaled: Rgba32FImage;
    scaled = scale_padded(&padded, opt.ratio, bicubic);

    println!(
        "Finished scaling in {:?} seconds",
        timer.elapsed().as_secs_f32()
    );

    (DynamicImage::ImageRgba32F(scaled).into_rgba8())
        .save(opt.output_path)
        .expect("Failed to write output scaled image");
}
