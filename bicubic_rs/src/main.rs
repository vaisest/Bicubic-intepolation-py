use image::io::Reader as ImgReader;
use image::{DynamicImage, GenericImage, GenericImageView, Pixel, Rgb, Rgb32FImage};
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

// #[inline(never)]
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

fn pad(img: &Rgb32FImage) -> Rgb32FImage {
    let mut dest: Rgb32FImage = Rgb32FImage::new(img.width() + 4, img.height() + 4);

    for y in 0..2 {
        for x in 0..2 {
            dest.put_pixel(x, y, *img.get_pixel(0, 0));
        }
        for x in img.width()..img.width() + 2 {
            dest.put_pixel(x, y, *img.get_pixel(0, 0));
        }
    }
    for y in img.height()..img.height() + 2 {
        for x in 0..2 {
            dest.put_pixel(x, y, *img.get_pixel(0, 0));
        }
        for x in img.width()..img.width() + 2 {
            dest.put_pixel(x, y, *img.get_pixel(0, 0));
        }
    }

    dest.copy_from(img, 2, 2).expect("pad panic");

    return dest;
}

// #[inline(never)] TODO: TEST PADDING SPEED
pub fn scale<F>(img: &Rgb32FImage, ratio: f32, u: F) -> Rgb32FImage
where
    F: Fn(f32) -> f32,
{
    let new_w = ((img.width() as f32) * ratio) as u32;
    let new_h = ((img.height() as f32) * ratio) as u32;

    let dest = Rgb32FImage::from_fn(new_w, new_h, |i, j| {
        let y = (j as f32 + 0.5) * (1.0 / ratio) - 0.5;
        let iy = y as i32;
        let decy = y.trunc() - y;

        let x = (i as f32 + 0.5) * (1.0 / ratio) - 0.5;
        let ix = x as i32;
        let decx = x.trunc() - x;

        let pix: Rgb<f32> = (-1i32..=2)
            .map(|m: i32| {
                return (-1i32..=2)
                    .map(|l| {
                        let p = img
                            .get_pixel(
                                (ix + l).clamp(0, img.width() as i32 - 1) as u32,
                                (iy + m).clamp(0, img.height() as i32 - 1) as u32,
                            )
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
            .map(|n| n.clamp(0.0, 1.0));

        return pix;
    });
    return dest;
}

// unsafe because input should be padded where oob is not possible
pub unsafe fn scaledpad<F>(img: &Rgb32FImage, ratio: f32, u: F) -> Rgb32FImage
where
    F: Fn(f32) -> f32,
{
    let new_w = (((img.width() - 4) as f32) * ratio) as u32;
    let new_h = (((img.height() - 4) as f32) * ratio) as u32;

    let mut dest = Rgb32FImage::new(new_w, new_h);
    for j in 0..new_h {
        let y = (j as f32 + 0.5) * (1.0 / ratio) - 0.5 + 2.0;
        let iy = y as i32;
        let decy = y.trunc() - y;

        for i in 0..new_w {
            let x = (i as f32 + 0.5) * (1.0 / ratio) - 0.5 + 2.0;
            let ix = x as i32;
            let decx = x.trunc() - x;

            let mut pix = [0.0, 0.0, 0.0];
            for m in -1i32..=2 {
                for l in -1i32..=2 {
                    let v = img
                        .unsafe_get_pixel((ix + l) as u32, (iy + m) as u32)
                        .map(|v| v * u(decx + l as f32) * u(decy + m as f32));
                    pix[0] += v[0];
                    pix[1] += v[1];
                    pix[2] += v[2];
                }

                pix[0] = pix[0].clamp(0.0, 1.0);
                pix[1] = pix[1].clamp(0.0, 1.0);
                pix[2] = pix[2].clamp(0.0, 1.0);
                dest.unsafe_put_pixel(i, j, Rgb(pix));
            }
        }
    }
    return dest;
}

fn main() {
    let opt = Opt::from_args();

    let input_img = ImgReader::open(&opt.input_path)
        .expect("Could not load input image")
        .decode()
        .expect("Could not decode image")
        .into_rgb32f();

    println!(
        "Scaling image from {:?}x{:?} to {:?}x{:?}...",
        input_img.width(),
        input_img.height(),
        (input_img.width() as f32 * opt.ratio) as u32,
        (input_img.height() as f32 * opt.ratio) as u32
    );

    let timer = Instant::now();

    let padded = pad(&input_img);
    let scaled: Rgb32FImage;
    unsafe {
        scaled = scaledpad(&padded, opt.ratio, bicubic);
    }
    println!(
        "Finished scaling in {:?} seconds",
        timer.elapsed().as_secs_f32()
    );
    // let timer = Instant::now();
    // let scaled = scale(&input_img, opt.ratio, bicubic);

    // println!(
    //     "Finished scaling in {:?} seconds",
    //     timer.elapsed().as_secs_f32()
    // );

    (DynamicImage::ImageRgb32F(scaled).into_rgb8())
        .save(opt.output_path)
        .expect("Failed to write output scaled image");
}
