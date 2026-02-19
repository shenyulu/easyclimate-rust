use ndarray::{Array2, ArrayView2};

#[derive(Clone, Copy)]
pub struct LambertProj {
    pub center_lon: f64,
    pub n: f64,
    pub f: f64,
    pub rho0: f64,
}

const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;
const HALF_RAD_PER_DEG: f64 = RAD_PER_DEG / 2.0;

pub fn create_proj(center_lon: f64, center_lat: f64, lat1: f64, lat2: f64) -> LambertProj {
    let lat1r = lat1 * RAD_PER_DEG;
    let lat2r = lat2 * RAD_PER_DEG;

    let n = if (lat1 - lat2).abs() > 0.0 {
        (lat1r.cos() / lat2r.cos()).ln()
            / (((90.0 + lat2) * HALF_RAD_PER_DEG).tan() / ((90.0 + lat1) * HALF_RAD_PER_DEG).tan()).ln()
    } else {
        lat1r.sin()
    };

    let f = lat1r.cos() * (( (90.0 + lat1) * HALF_RAD_PER_DEG).tan()).powf(n) / n;
    let rho0 = f / (((90.0 + center_lat) * HALF_RAD_PER_DEG).tan()).powf(n);

    LambertProj { center_lon, n, f, rho0 }
}

/// lonlat(N,2) -> lambert(N,2)
pub fn to_map_points(pts: &ArrayView2<f64>, proj: &LambertProj) -> Array2<f64> {
    let n = pts.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let lon = pts[(i, 0)];
        let lat = pts[(i, 1)];
        let rho = proj.f / (((90.0 + lat) * HALF_RAD_PER_DEG).tan()).powf(proj.n);
        let arg = proj.n * (lon - proj.center_lon) * RAD_PER_DEG;
        out[(i, 0)] = rho * arg.sin() / RAD_PER_DEG;
        out[(i, 1)] = (proj.rho0 - rho * arg.cos()) / RAD_PER_DEG;
    }
    out
}

/// Corresponding to the projection inference of Python S2: lon-span < 180 and not crossing the equator: contentReference[oaicite:15]{index=15}
pub fn infer_lambert_proj(
    x0: (f64, f64),
    step: (f64, f64),
    size: (usize, usize),
) -> Result<LambertProj, String> {
    let lon0 = x0.0;
    let lon1 = x0.0 + (size.0 as f64 - 1.0) * step.0;
    let lat0 = x0.1;
    let lat1 = x0.1 + (size.1 as f64 - 1.0) * step.1;

    let lon_min = lon0.min(lon1);
    let lon_max = lon0.max(lon1);
    let lat_min = lat0.min(lat1);
    let lat_max = lat0.max(lat1);

    let lon_span = lon_max - lon_min;
    if lon_span >= 180.0 {
        return Err("optimized_convolution_S2 requires longitude span < 180 degrees".into());
    }

    let center_lon = 0.5 * (lon_min + lon_max);
    let center_lat = 0.5 * (lat_min + lat_max);

    let (std1, std2) = if lat_min >= 0.0 {
        (30.0, 60.0)
    } else if lat_max <= 0.0 {
        (-30.0, -60.0)
    } else {
        return Err("optimized_convolution_S2 does not support domains crossing the equator".into());
    };

    Ok(create_proj(center_lon, center_lat, std1, std2))
}

/// half kernel size opt (scalar) for margin estimation :contentReference[oaicite:16]{index=16}
pub fn half_kernel_size_opt_scalar(sigma: f64, step: f64, num_iter: usize) -> Result<usize, String> {
    if sigma <= 0.0 || step <= 0.0 || num_iter == 0 {
        return Err("invalid sigma/step/num_iter".into());
    }
    let s = sigma / step;
    let half = (((1.0 + 12.0 * s * s / (num_iter as f64)).sqrt() - 1.0) / 2.0).floor() as i64;
    Ok(half.max(0) as usize)
}

/// 对应 python _infer_lambert_grid: Project the four corners and expand by min/max + margin_steps: contentReference[oaicite:17]{index=17}
pub fn infer_lambert_grid(
    x0: (f64, f64),
    step: (f64, f64),
    size: (usize, usize),
    proj: &LambertProj,
    margin_steps: usize,
) -> Result<((f64, f64), (usize, usize)), String> {
    let lon0 = x0.0;
    let lon1 = x0.0 + (size.0 as f64 - 1.0) * step.0;
    let lat0 = x0.1;
    let lat1 = x0.1 + (size.1 as f64 - 1.0) * step.1;

    let corners = Array2::from_shape_vec(
        (4, 2),
        vec![
            lon0, lat0,
            lon0, lat1,
            lon1, lat0,
            lon1, lat1,
        ],
    ).unwrap();
    let lam_corners = to_map_points(&corners.view(), proj);

    let min_x = lam_corners.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_x = lam_corners.column(0).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_y = lam_corners.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_y = lam_corners.column(1).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let margin_x = margin_steps as f64 * step.0;
    let margin_y = margin_steps as f64 * step.1;

    let lam_x0 = (min_x - margin_x, min_y - margin_y);
    let lam_size_x = (((max_x - min_x + 2.0 * margin_x) / step.0).ceil() as isize + 2).max(2) as usize;
    let lam_size_y = (((max_y - min_y + 2.0 * margin_y) / step.1).ceil() as isize + 2).max(2) as usize;

    Ok((lam_x0, (lam_size_x, lam_size_y)))
}

/// Lambert field -> lonlat field resample (Bilinear)
pub fn resample_lambert_to_lonlat(
    lam_field: &ndarray::ArrayView2<f32>, // (lam_ny, lam_nx)
    lam_x0: (f64, f64),
    x0: (f64, f64),
    step: (f64, f64),
    size: (usize, usize),
    proj: &LambertProj,
) -> ndarray::Array2<f32> {
    let (nx, ny) = size;
    let (lam_ny, lam_nx) = lam_field.dim();

    let mut out = ndarray::Array2::<f32>::zeros((ny, nx));

    for j in 0..ny {
        let lat = x0.1 + j as f64 * step.1;
        for i in 0..nx {
            let lon = x0.0 + i as f64 * step.0;

            // forward project point
            let rho = proj.f / (((90.0 + lat) * HALF_RAD_PER_DEG).tan()).powf(proj.n);
            let arg = proj.n * (lon - proj.center_lon) * RAD_PER_DEG;
            let mx = rho * arg.sin() / RAD_PER_DEG;
            let my = (proj.rho0 - rho * arg.cos()) / RAD_PER_DEG;

            // map to lambert grid index
            let fx = (mx - lam_x0.0) / step.0;
            let fy = (my - lam_x0.1) / step.1;

            if fx < 0.0 || fy < 0.0 || fx >= (lam_nx as f64 - 1.0) || fy >= (lam_ny as f64 - 1.0) {
                out[(j, i)] = f32::NAN;
                continue;
            }

            let x0i = fx.floor() as usize;
            let y0i = fy.floor() as usize;
            let xw = (fx - x0i as f64) as f32;
            let yw = (fy - y0i as f64) as f32;

            let v00 = lam_field[(y0i, x0i)];
            let v10 = lam_field[(y0i, x0i + 1)];
            let v11 = lam_field[(y0i + 1, x0i + 1)];
            let v01 = lam_field[(y0i + 1, x0i)];

            out[(j, i)] =
                (1.0 - xw) * (1.0 - yw) * v00 +
                xw * (1.0 - yw) * v10 +
                xw * yw * v11 +
                (1.0 - xw) * yw * v01;
        }
    }

    out
}
