use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::lambert;
use crate::lambert::LambertProj;

// -----------------------------
// small helpers
// -----------------------------
fn normalize_values_inplace(val: &mut [f64]) -> f64 {
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for &v in val.iter() {
        if v < vmin {
            vmin = v;
        }
        if v > vmax {
            vmax = v;
        }
    }
    let offset = 0.5 * (vmin + vmax);
    for v in val.iter_mut() {
        *v -= offset;
    }
    offset
}

fn max_dist_weight(max_dist: f64) -> f64 {
    (-max_dist * max_dist / 2.0).exp()
}

const RAD_PER_DEG: f64 = std::f64::consts::PI / 180.0;

#[inline]
fn dist_s2_deg(lon0: f64, lat0: f64, lon1: f64, lat1: f64) -> f64 {
    let lat0r = lat0 * RAD_PER_DEG;
    let lat1r = lat1 * RAD_PER_DEG;
    let dlon = (lon1 - lon0) * RAD_PER_DEG;
    let mut arg = lat0r.sin() * lat1r.sin() + lat0r.cos() * lat1r.cos() * dlon.cos();
    if arg > 1.0 {
        arg = 1.0;
    } else if arg < -1.0 {
        arg = -1.0;
    }
    arg.acos() / RAD_PER_DEG
}

fn to_vec_len2_f64(name: &str, v: Vec<f64>) -> PyResult<[f64; 2]> {
    if v.len() != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "{name} must have length 2, got {}",
            v.len()
        )));
    }
    Ok([v[0], v[1]])
}

fn to_vec_len2_usize(name: &str, v: Vec<usize>) -> PyResult<[usize; 2]> {
    if v.len() != 2 {
        return Err(PyRuntimeError::new_err(format!(
            "{name} must have length 2, got {}",
            v.len()
        )));
    }
    Ok([v[0], v[1]])
}

// -----------------------------
// naive S2
// -----------------------------
fn interpolate_naive_s2(
    pts: &ArrayView2<f64>, // (N,2) lon,lat
    mut val: Vec<f64>,
    sigma: f64,            // scalar sigma (deg, in your call it's "grid-units" but still degrees metric on S2 wrapper)
    x0: (f64, f64),
    step: (f64, f64),
    size: (usize, usize), // (nx, ny)
) -> Array2<f32> {
    let offset = normalize_values_inplace(&mut val);

    let nx = size.0;
    let ny = size.1;

    let scale = 2.0 * sigma * sigma;
    let nobs = pts.nrows();

    let mut out = vec![f32::NAN; nx * ny];

    // parallelize over rows
    out.par_chunks_mut(nx)
        .enumerate()
        .for_each(|(j, row)| {
            let yc = x0.1 + j as f64 * step.1;
            for i in 0..nx {
                let xc = x0.0 + i as f64 * step.0;

                let mut wsum = 0.0f64;
                let mut vsum = 0.0f64;

                for k in 0..nobs {
                    let lon = pts[(k, 0)];
                    let lat = pts[(k, 1)];
                    let d = dist_s2_deg(xc, yc, lon, lat);
                    let w = (-(d * d) / scale).exp();
                    wsum += w;
                    vsum += w * val[k];
                }

                if wsum > 0.0 {
                    row[i] = (vsum / wsum + offset) as f32;
                } else {
                    row[i] = f32::NAN;
                }
            }
        });

    Array2::from_shape_vec((ny, nx), out).expect("shape")
}

// -----------------------------
// optimized_convolution_S2
// -----------------------------
fn interpolate_opt_convol_s2(
    pts: &ArrayView2<f64>, // (N,2) lon,lat
    val: Vec<f64>,
    sigma_xy: (f64, f64),
    x0: (f64, f64),
    step: (f64, f64),
    size: (usize, usize), // (nx, ny)
    num_iter: i32,
    max_dist: f64, // in sigma units
    resample: bool,
    lambert_proj: Option<(f64, f64, f64, f64, f64)>,
    lambert_grid: Option<((f64, f64), (usize, usize))>,
    auto_proj: bool,
) -> PyResult<Array2<f32>> {
    // projection
    let proj: LambertProj = if let Some(p) = lambert_proj {
        LambertProj {
            center_lon: p.0,
            n: p.1,
            f: p.3,
            rho0: p.4,
        }
    } else {
        if !auto_proj {
            return Err(PyRuntimeError::new_err(
                "lambert_proj must be provided when auto_proj=False",
            ));
        }
        lambert::infer_lambert_proj(x0, step, size).map_err(PyRuntimeError::new_err)?
    };

    // lambert grid
    let (lam_x0, lam_size) = if let Some(g) = lambert_grid {
        (g.0, g.1)
    } else {
        if !auto_proj {
            return Err(PyRuntimeError::new_err(
                "lambert_grid must be provided when auto_proj=False",
            ));
        }
        let max_sigma = sigma_xy.0.max(sigma_xy.1);
        let min_step = step.0.min(step.1);

        let num_iter_u: usize = usize::try_from(num_iter)
            .map_err(|_| PyRuntimeError::new_err("num_iter must be >= 0 and fit in usize"))?;

        let margin_steps = lambert::half_kernel_size_opt_scalar(max_sigma, min_step, num_iter_u)
            .map_err(PyRuntimeError::new_err)?
            + 2;

        lambert::infer_lambert_grid(x0, step, size, &proj, margin_steps)
            .map_err(PyRuntimeError::new_err)?
    };

    // lonlat -> lambert pts
    let lam_pts_arr = lambert::to_map_points(pts, &proj); // (N,2)
    let mut lam_pts: Vec<(f64, f64, f64)> = Vec::with_capacity(lam_pts_arr.nrows());
    for i in 0..lam_pts_arr.nrows() {
        lam_pts.push((lam_pts_arr[(i, 0)], lam_pts_arr[(i, 1)], 0.0));
    }

    let mw = max_dist_weight(max_dist);

    // call existing planar optimized convolution (reused from barnes.rs)
    let lam_field_dyn = crate::barnes::interpolate_opt_convolution(
        &lam_pts,
        val,
        &[sigma_xy.0, sigma_xy.1],
        &[lam_x0.0, lam_x0.1],
        &[step.0, step.1],
        &[lam_size.0, lam_size.1],
        num_iter,
        mw,
    );

    // lam_field_dyn is (lam_ny, lam_nx) because barnes.rs reverses size
    let lam_nx = lam_size.0;
    let lam_ny = lam_size.1;
    let lam_field = lam_field_dyn
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| PyRuntimeError::new_err("internal: lam_field is not 2D"))?;
    if lam_field.shape() != [lam_ny, lam_nx] {
        return Err(PyRuntimeError::new_err(format!(
            "internal: lam_field shape mismatch, got {:?}, expect ({},{})",
            lam_field.shape(),
            lam_ny,
            lam_nx
        )));
    }

    if !resample {
        return Ok(lam_field.to_owned());
    }

    // resample lambert -> lonlat
    let out = lambert::resample_lambert_to_lonlat(
        &lam_field.view(),
        lam_x0,
        x0,
        step,
        size,
        &proj,
    );
    Ok(out)
}

// -----------------------------
// Python binding
// -----------------------------
#[pyfunction]
#[pyo3(signature = (
    pts, val, sigma, x0, step, size,
    method="optimized_convolution_S2",
    num_iter=4,
    max_dist=3.5,
    resample=true,
    lambert_proj=None,
    lambert_grid=None,
    auto_proj=true
))]
pub fn barnes_s2<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<f64>,
    val: PyReadonlyArray1<f64>,
    sigma: &Bound<'py, PyAny>,
    x0: Vec<f64>,
    step: &Bound<'py, PyAny>,
    size: Vec<usize>,
    method: &str,
    num_iter: i32,
    max_dist: f64,
    resample: bool,
    lambert_proj: Option<(f64, f64, f64, f64, f64)>,
    lambert_grid: Option<((f64, f64), (usize, usize))>,
    auto_proj: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    // dim must be 2
    if pts.shape().len() != 2 || pts.shape()[1] != 2 {
        return Err(PyRuntimeError::new_err(
            "pts must be a 2D array of shape (N,2)",
        ));
    }
    let ptsv = pts.as_array();
    let vv = val.as_slice()?.to_vec();

    let x0v = to_vec_len2_f64("x0", x0)?;
    let sizev = to_vec_len2_usize("size", size)?;

    // step: accept scalar or len-2
    let step_vec: Vec<f64> = if let Ok(s) = step.extract::<f64>() {
        vec![s, s]
    } else if let Ok(v) = step.extract::<Vec<f64>>() {
        v
    } else {
        return Err(PyRuntimeError::new_err(
            "step must be a float or length-2 array-like",
        ));
    };
    let stepv = to_vec_len2_f64("step", step_vec)?;

    // sigma: accept scalar or len-2
    let sigma_vec: Vec<f64> = if let Ok(s) = sigma.extract::<f64>() {
        vec![s, s]
    } else if let Ok(v) = sigma.extract::<Vec<f64>>() {
        v
    } else {
        return Err(PyRuntimeError::new_err(
            "sigma must be a float or length-2 array-like",
        ));
    };
    let sigv = to_vec_len2_f64("sigma", sigma_vec)?;

    let x0t = (x0v[0], x0v[1]);
    let stept = (stepv[0], stepv[1]);
    let sizet = (sizev[0], sizev[1]); // (nx, ny)

    let out2: Array2<f32> = match method {
        "optimized_convolution_S2" => {
            interpolate_opt_convol_s2(
                &ptsv,
                vv,
                (sigv[0], sigv[1]),
                x0t,
                stept,
                sizet,
                num_iter,
                max_dist,
                resample,
                lambert_proj,
                lambert_grid,
                auto_proj,
            )?
        }
        "naive_S2" => {
            // naive assumes isotropic sigma
            interpolate_naive_s2(&ptsv, vv, sigv[0], x0t, stept, sizet)
        }
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "invalid method: {method}"
            )))
        }
    };

    Ok(out2.into_dyn().into_pyarray(py))
}
