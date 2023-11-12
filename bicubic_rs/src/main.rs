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

//TODO: rest of the filters
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
    // pad by copying border
    let mut dest: Rgb32FImage = Rgb32FImage::new(img.width() + 4, img.height() + 4);

    for y in 0..2 {
        // top
        for x in 0..dest.width() {
            dest.put_pixel(x, y, *img.get_pixel((x - 2).clamp(0, img.width() - 1), 0));
        }
    }
    for y in 2..img.height() {
        // left
        for x in 0..2 {
            dest.put_pixel(x, y, *img.get_pixel(0, (y - 2).clamp(0, img.height() - 1)));
        }
        // right
        for x in img.width()..dest.width() {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel(img.width() - 1, (y - 2).clamp(0, img.height() - 1)),
            );
        }
    }
    for y in img.height()..dest.height() {
        // bottom
        for x in 0..dest.width() {
            dest.put_pixel(
                x,
                y,
                *img.get_pixel((x - 2).clamp(0, img.width() - 1), img.height() - 1),
            );
        }
    }

    dest.copy_from(img, 2, 2).expect("pad panic");

    return dest;
}

// unsafe because input should be padded where oob is not possible
pub unsafe fn scale_padded<F>(img: &Rgb32FImage, ratio: f32, u: F) -> Rgb32FImage
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
                    let p = img.unsafe_get_pixel((ix + l) as u32, (iy + m) as u32);
                    let v = p.channels();

                    pix[0] += v[0] * u(decx + l as f32) * u(decy + m as f32);
                    pix[1] += v[1] * u(decx + l as f32) * u(decy + m as f32);
                    pix[2] += v[2] * u(decx + l as f32) * u(decy + m as f32);
                }
            }
            pix[0] = pix[0].clamp(0.0, 1.0);
            pix[1] = pix[1].clamp(0.0, 1.0);
            pix[2] = pix[2].clamp(0.0, 1.0);
            dest.unsafe_put_pixel(i, j, Rgb(pix));
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
        scaled = scale_padded(&padded, opt.ratio, bicubic);
    }

    println!(
        "Finished scaling in {:?} seconds",
        timer.elapsed().as_secs_f32()
    );

    (DynamicImage::ImageRgb32F(scaled).into_rgb8())
        .save(opt.output_path)
        .expect("Failed to write output scaled image");
}
