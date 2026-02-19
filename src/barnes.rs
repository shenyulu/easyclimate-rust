use std::f64::consts::PI;

use ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

// -----------------------------
// helpers: parse python inputs
// -----------------------------

fn py_to_vec_f64<'py>(obj: &Bound<'py, PyAny>, dim: usize, name: &str) -> PyResult<Vec<f64>> {
    // Accept scalar or array-like of length dim.
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(vec![v; dim]);
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() != dim {
            return Err(PyRuntimeError::new_err(format!(
                "{name} length mismatch: expected {dim}, got {}",
                v.len()
            )));
        }
        return Ok(v);
    }
    // Try numpy array
    if let Ok(arr) = obj.extract::<PyReadonlyArrayDyn<f64>>() {
        let a = arr.as_array();
        if a.len() == 1 {
            return Ok(vec![a[[]]; dim]);
        }
        if a.len() != dim {
            return Err(PyRuntimeError::new_err(format!(
                "{name} length mismatch: expected {dim}, got {}",
                a.len()
            )));
        }
        return Ok(a.iter().cloned().collect());
    }
    Err(PyRuntimeError::new_err(format!(
        "cannot parse {name} as scalar or length-{dim} array"
    )))
}

fn py_to_size<'py>(obj: &Bound<'py, PyAny>, dim: usize) -> PyResult<Vec<usize>> {
    // Python: if dim==1, can be scalar; else must be array-like
    if dim == 1 {
        if let Ok(n) = obj.extract::<usize>() {
            return Ok(vec![n]);
        }
    }
    let v = obj.extract::<Vec<usize>>().map_err(|_| {
        PyRuntimeError::new_err(format!(
            "size must be an array-like of length {dim} (or scalar if dim==1)"
        ))
    })?;
    if v.len() != dim {
        return Err(PyRuntimeError::new_err(format!(
            "size length mismatch: expected {dim}, got {}",
            v.len()
        )));
    }
    Ok(v)
}

fn max_dist_weight(max_dist: f64) -> f64 {
    (-max_dist * max_dist / 2.0).exp()
}

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
    let offset = (vmin + vmax) / 2.0;
    for v in val.iter_mut() {
        *v -= offset;
    }
    offset
}


// --------------------------------------
// kernel params
// --------------------------------------

fn get_half_kernel_size_opt(sigma: &[f64], step: &[f64], num_iter: i32) -> Vec<i32> {
    // ((sqrt(1+12*s*s/num_iter) - 1)/2).astype(int)
    let ni = num_iter as f64;
    sigma
        .iter()
        .zip(step.iter())
        .map(|(&sg, &st)| {
            let s = sg / st;
            (((1.0 + 12.0 * s * s / ni).sqrt() - 1.0) / 2.0).floor() as i32
        })
        .collect()
}

fn get_tail_value(sigma: &[f64], step: &[f64], num_iter: i32) -> Vec<f64> {
    // matches _get_tail_value in interpolation.py :contentReference[oaicite:2]{index=2}
    let half = get_half_kernel_size_opt(sigma, step, num_iter);
    let ni = num_iter as f64;

    half.iter()
        .zip(sigma.iter())
        .zip(step.iter())
        .map(|((&t, &sg), &st)| {
            let t_f = t as f64;
            let kernel_size = 2.0 * t_f + 1.0;
            let sigma_rect_sqr = (t_f + 1.0) * t_f / 3.0 * st * st;
            0.5 * kernel_size * (sg * sg / ni - sigma_rect_sqr) / (((t_f + 1.0) * st).powi(2) - sg * sg / ni)
        })
        .collect()
}

fn get_half_kernel_size(sigma: &[f64], step: &[f64], num_iter: i32) -> Vec<i32> {
    // (sqrt(3/num_iter)*sigma/step + 0.5).astype(int) :contentReference[oaicite:3]{index=3}
    let ni = num_iter as f64;
    sigma
        .iter()
        .zip(step.iter())
        .map(|(&sg, &st)| ((3.0 / ni).sqrt() * sg / st + 0.5).floor() as i32)
        .collect()
}

// --------------------------------------
// 1D accumulate kernels (exact phases)
// --------------------------------------

fn accumulate_array(mut in_arr: Vec<f64>, mut h_arr: Vec<f64>, arr_len: usize, rect_len: usize, num_iter: usize) -> Vec<f64> {
    // matches _accumulate_array :contentReference[oaicite:4]{index=4}
    let h0 = (rect_len - 1) / 2;
    let h1 = rect_len - h0;

    for _ in 0..num_iter {
        let mut accu = 0.0;

        // phase a: accumulate first h0 elements (k=-h0..-1)
        for k in (-(h0 as isize))..0 {
            accu += in_arr[(k + h0 as isize) as usize];
        }

        // phase b: k=0..h1-1
        for k in 0..h1 {
            accu += in_arr[k + h0];
            h_arr[k] = accu;
        }

        // phase c: k=h1..arr_len-h0-1
        for k in h1..(arr_len - h0) {
            accu += in_arr[k + h0] - in_arr[k - h1];
            h_arr[k] = accu;
        }

        // phase d: k=arr_len-h0..arr_len-1
        for k in (arr_len - h0)..arr_len {
            accu -= in_arr[k - h1];
            h_arr[k] = accu;
        }

        std::mem::swap(&mut in_arr, &mut h_arr);
    }
    in_arr
}

fn accumulate_tail_array(
    mut in_arr: Vec<f64>,
    mut h_arr: Vec<f64>,
    arr_len: usize,
    rect_len: usize,
    num_iter: usize,
    alpha: f64,
) -> Vec<f64> {
    // matches _accumulate_tail_array :contentReference[oaicite:5]{index=5}
    let h0 = (rect_len - 1) / 2;
    let h0_1 = h0 + 1;
    let h1 = rect_len - h0;

    for _ in 0..num_iter {
        let mut accu = 0.0;

        // phase a: accumulate first h0 elements
        for k in (-(h0 as isize))..0 {
            accu += in_arr[(k + h0 as isize) as usize];
        }

        // phase b: k=0..h1-1
        for k in 0..h1 {
            accu += in_arr[k + h0];
            h_arr[k] = accu + alpha * in_arr[k + h0_1];
        }

        // phase c: k=h1..arr_len-h0_1-1
        for k in h1..(arr_len - h0_1) {
            accu += in_arr[k + h0] - in_arr[k - h1];
            h_arr[k] = accu + alpha * (in_arr[k - h1] + in_arr[k + h0_1]);
        }

        // phase c': k=arr_len-h0_1
        {
            let k = arr_len - h0_1;
            accu += in_arr[k + h0] - in_arr[k - h1];
            h_arr[k] = accu + alpha * in_arr[k - h1];
        }

        // phase d: k=arr_len-h0..arr_len-1
        for k in (arr_len - h0)..arr_len {
            accu -= in_arr[k - h1];
            h_arr[k] = accu + alpha * in_arr[k - h1];
        }

        std::mem::swap(&mut in_arr, &mut h_arr);
    }
    in_arr
}

// --------------------------------------
// indexing helpers (reverse dims like py)
// --------------------------------------

#[inline]
fn idx2(y: usize, x: usize, nx: usize) -> usize {
    y * nx + x
}

#[inline]
fn idx3(z: usize, y: usize, x: usize, ny: usize, nx: usize) -> usize {
    (z * ny + y) * nx + x
}

// --------------------------------------
// inject data (algorithm B.1)
// --------------------------------------

fn inject_data_1d(vg: &mut [f64], wg: &mut [f64], pts: &[(f64, f64, f64)], val: &[f64], x0: &[f64], step: &[f64], size: &[usize]) {
    let nx = size[0] as f64;
    for (k, &(px, _, _)) in pts.iter().enumerate() {
        let xc = (px - x0[0]) / step[0];
        if xc < 0.0 || xc >= nx - 1.0 {
            continue;
        }
        let xi = xc.floor() as usize;
        let xw = xc - xi as f64;

        let w0 = 1.0 - xw;
        vg[xi] += w0 * val[k];
        wg[xi] += w0;

        let w1 = xw;
        vg[xi + 1] += w1 * val[k];
        wg[xi + 1] += w1;
    }
}

fn inject_data_2d(vg: &mut [f64], wg: &mut [f64], pts: &[(f64, f64, f64)], val: &[f64], x0: &[f64], step: &[f64], size: &[usize]) {
    let nx = size[0] as f64;
    let ny = size[1] as f64;
    let nxu = size[0];

    for (k, &(px, py, _)) in pts.iter().enumerate() {
        let xc = (px - x0[0]) / step[0];
        let yc = (py - x0[1]) / step[1];
        if xc < 0.0 || yc < 0.0 || xc >= nx - 1.0 || yc >= ny - 1.0 {
            continue;
        }
        let xi = xc.floor() as usize;
        let yi = yc.floor() as usize;
        let xw = xc - xi as f64;
        let yw = yc - yi as f64;

        let w00 = (1.0 - xw) * (1.0 - yw);
        let w10 = xw * (1.0 - yw);
        let w11 = xw * yw;
        let w01 = (1.0 - xw) * yw;

        let i00 = idx2(yi, xi, nxu);
        let i10 = idx2(yi, xi + 1, nxu);
        let i11 = idx2(yi + 1, xi + 1, nxu);
        let i01 = idx2(yi + 1, xi, nxu);

        vg[i00] += w00 * val[k];
        wg[i00] += w00;

        vg[i10] += w10 * val[k];
        wg[i10] += w10;

        vg[i11] += w11 * val[k];
        wg[i11] += w11;

        vg[i01] += w01 * val[k];
        wg[i01] += w01;
    }
}

fn inject_data_3d(vg: &mut [f64], wg: &mut [f64], pts: &[(f64, f64, f64)], val: &[f64], x0: &[f64], step: &[f64], size: &[usize]) {
    let nx = size[0] as f64;
    let ny = size[1] as f64;
    let nz = size[2] as f64;

    let nxu = size[0];
    let nyu = size[1];

    for (k, &(px, py, pz)) in pts.iter().enumerate() {
        let xc = (px - x0[0]) / step[0];
        let yc = (py - x0[1]) / step[1];
        let zc = (pz - x0[2]) / step[2];

        if xc < 0.0 || yc < 0.0 || zc < 0.0 || xc >= nx - 1.0 || yc >= ny - 1.0 || zc >= nz - 1.0 {
            continue;
        }

        let xi = xc.floor() as usize;
        let yi = yc.floor() as usize;
        let zi = zc.floor() as usize;
        let xw = xc - xi as f64;
        let yw = yc - yi as f64;
        let zw = zc - zi as f64;

        let w000 = (1.0 - xw) * (1.0 - yw) * (1.0 - zw);
        let w100 = xw * (1.0 - yw) * (1.0 - zw);
        let w110 = xw * yw * (1.0 - zw);
        let w010 = (1.0 - xw) * yw * (1.0 - zw);

        let w001 = (1.0 - xw) * (1.0 - yw) * zw;
        let w101 = xw * (1.0 - yw) * zw;
        let w111 = xw * yw * zw;
        let w011 = (1.0 - xw) * yw * zw;

        let i000 = idx3(zi, yi, xi, nyu, nxu);
        let i100 = idx3(zi, yi, xi + 1, nyu, nxu);
        let i110 = idx3(zi, yi + 1, xi + 1, nyu, nxu);
        let i010 = idx3(zi, yi + 1, xi, nyu, nxu);

        let i001 = idx3(zi + 1, yi, xi, nyu, nxu);
        let i101 = idx3(zi + 1, yi, xi + 1, nyu, nxu);
        let i111 = idx3(zi + 1, yi + 1, xi + 1, nyu, nxu);
        let i011 = idx3(zi + 1, yi + 1, xi, nyu, nxu);

        let vk = val[k];

        vg[i000] += w000 * vk; wg[i000] += w000;
        vg[i100] += w100 * vk; wg[i100] += w100;
        vg[i110] += w110 * vk; wg[i110] += w110;
        vg[i010] += w010 * vk; wg[i010] += w010;

        vg[i001] += w001 * vk; wg[i001] += w001;
        vg[i101] += w101 * vk; wg[i101] += w101;
        vg[i111] += w111 * vk; wg[i111] += w111;
        vg[i011] += w011 * vk; wg[i011] += w011;
    }
}

// --------------------------------------
// convolution passes + thresholding
// --------------------------------------

fn threshold_weights(wg: &mut [f64], conv_scale_factor: f64) {
    for w in wg.iter_mut() {
        if *w < conv_scale_factor {
            *w = f64::NAN;
        }
    }
}

fn product_conv_scale_factor_convolution(kernel_size: &[i32], sigma: &[f64], step: &[f64], num_iter: i32, max_dist_weight: f64) -> f64 {
    // kernel_size**num_iter / sqrt(2*pi) / (sigma/step) then product, then * max_dist_weight
    // matches :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
    let ni = num_iter as i32;
    let mut p = 1.0;
    for m in 0..sigma.len() {
        let ks = kernel_size[m] as f64;
        let term = ks.powi(ni) / ( (2.0 * PI).sqrt() ) / (sigma[m] / step[m]);
        p *= term;
    }
    p * max_dist_weight
}

fn product_conv_scale_factor_opt(kernel_size: &[i32], tail_value: &[f64], sigma: &[f64], step: &[f64], num_iter: i32, max_dist_weight: f64) -> f64 {
    // (kernel_size + 2*tail_value)**num_iter / sqrt(2*pi) / (sigma/step) product * max_dist_weight
    // matches :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    let ni = num_iter as i32;
    let mut p = 1.0;
    for m in 0..sigma.len() {
        let base = (kernel_size[m] as f64) + 2.0 * tail_value[m];
        let term = base.powi(ni) / ( (2.0 * PI).sqrt() ) / (sigma[m] / step[m]);
        p *= term;
    }
    p * max_dist_weight
}

// --------------------------------------
// algorithm B: convolution (no tail)
// --------------------------------------

fn interpolate_convolution(
    pts: &[(f64, f64, f64)],
    mut val: Vec<f64>,
    sigma: &[f64],
    x0: &[f64],
    step: &[f64],
    size: &[usize],
    num_iter: i32,
    max_dist_weight: f64,
) -> ArrayD<f32> {
    let dim = size.len();
    let offset = normalize_values_inplace(&mut val);

    // reverse dims for output: rsize = size[::-1]
    let rsize: Vec<usize> = size.iter().cloned().rev().collect();
    let ngrid: usize = rsize.iter().product();

    let mut vg = vec![0.0_f64; ngrid];
    let mut wg = vec![0.0_f64; ngrid];

    // inject
    match dim {
        1 => inject_data_1d(&mut vg, &mut wg, pts, &val, x0, step, size),
        2 => inject_data_2d(&mut vg, &mut wg, pts, &val, x0, step, size),
        3 => inject_data_3d(&mut vg, &mut wg, pts, &val, x0, step, size),
        _ => unreachable!(),
    }

    let half = get_half_kernel_size(sigma, step, num_iter);
    let kernel_size: Vec<i32> = half.iter().map(|&t| 2 * t + 1).collect();

    // convolve along each axis (in the reversed layout!)
    // Layout:
    // 1D: [x]
    // 2D: [y, x]
    // 3D: [z, y, x]
    let iters = num_iter as usize;

    if dim == 1 {
        let arr_len = size[0];
        let rect_len = kernel_size[0] as usize;
        let h = vec![0.0; arr_len];

        let v_new = accumulate_array(vg.clone(), h.clone(), arr_len, rect_len, iters);
        let w_new = accumulate_array(wg.clone(), h.clone(), arr_len, rect_len, iters);
        vg = v_new;
        wg = w_new;

        let csf = product_conv_scale_factor_convolution(&kernel_size, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    } else if dim == 2 {
        let nx = size[0];
        let ny = size[1];

        // x-direction: each row y
        {
            let rect_len = kernel_size[0] as usize;
            let h = vec![0.0; nx];

            for j in 0..ny {
                let mut row_v = Vec::with_capacity(nx);
                let mut row_w = Vec::with_capacity(nx);
                for i in 0..nx {
                    let id = idx2(j, i, nx);
                    row_v.push(vg[id]);
                    row_w.push(wg[id]);
                }
                let row_v2 = accumulate_array(row_v, h.clone(), nx, rect_len, iters);
                let row_w2 = accumulate_array(row_w, h.clone(), nx, rect_len, iters);
                for i in 0..nx {
                    let id = idx2(j, i, nx);
                    vg[id] = row_v2[i];
                    wg[id] = row_w2[i];
                }
            }
        }

        // y-direction: each column x
        {
            let rect_len = kernel_size[1] as usize;
            let h = vec![0.0; ny];

            for i in 0..nx {
                let mut col_v = Vec::with_capacity(ny);
                let mut col_w = Vec::with_capacity(ny);
                for j in 0..ny {
                    let id = idx2(j, i, nx);
                    col_v.push(vg[id]);
                    col_w.push(wg[id]);
                }
                let col_v2 = accumulate_array(col_v, h.clone(), ny, rect_len, iters);
                let col_w2 = accumulate_array(col_w, h.clone(), ny, rect_len, iters);
                for j in 0..ny {
                    let id = idx2(j, i, nx);
                    vg[id] = col_v2[j];
                    wg[id] = col_w2[j];
                }
            }
        }

        let csf = product_conv_scale_factor_convolution(&kernel_size, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    } else {
        let nx = size[0];
        let ny = size[1];
        let nz = size[2];

        // x-direction
        {
            let rect_len = kernel_size[0] as usize;
            let h = vec![0.0; nx];

            for k in 0..nz {
                for j in 0..ny {
                    let mut line_v = Vec::with_capacity(nx);
                    let mut line_w = Vec::with_capacity(nx);
                    for i in 0..nx {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_array(line_v, h.clone(), nx, rect_len, iters);
                    let w2 = accumulate_array(line_w, h.clone(), nx, rect_len, iters);
                    for i in 0..nx {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[i];
                        wg[id] = w2[i];
                    }
                }
            }
        }

        // y-direction
        {
            let rect_len = kernel_size[1] as usize;
            let h = vec![0.0; ny];

            for k in 0..nz {
                for i in 0..nx {
                    let mut line_v = Vec::with_capacity(ny);
                    let mut line_w = Vec::with_capacity(ny);
                    for j in 0..ny {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_array(line_v, h.clone(), ny, rect_len, iters);
                    let w2 = accumulate_array(line_w, h.clone(), ny, rect_len, iters);
                    for j in 0..ny {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[j];
                        wg[id] = w2[j];
                    }
                }
            }
        }

        // z-direction
        {
            let rect_len = kernel_size[2] as usize;
            let h = vec![0.0; nz];

            for j in 0..ny {
                for i in 0..nx {
                    let mut line_v = Vec::with_capacity(nz);
                    let mut line_w = Vec::with_capacity(nz);
                    for k in 0..nz {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_array(line_v, h.clone(), nz, rect_len, iters);
                    let w2 = accumulate_array(line_w, h.clone(), nz, rect_len, iters);
                    for k in 0..nz {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[k];
                        wg[id] = w2[k];
                    }
                }
            }
        }

        let csf = product_conv_scale_factor_convolution(&kernel_size, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    }

    // finalize: (vg/wg + offset).astype(float32)
    let mut out = vec![0.0_f32; ngrid];
    for i in 0..ngrid {
        out[i] = (vg[i] / wg[i] + offset) as f32;
    }
    ArrayD::from_shape_vec(IxDyn(&rsize), out).unwrap()
}

// --------------------------------------
// algorithm B (optimized): convolution + tail
// --------------------------------------

pub(crate) fn interpolate_opt_convolution(
    pts: &[(f64, f64, f64)],
    mut val: Vec<f64>,
    sigma: &[f64],
    x0: &[f64],
    step: &[f64],
    size: &[usize],
    num_iter: i32,
    max_dist_weight: f64,
) -> ArrayD<f32> {
    let dim = size.len();
    let offset = normalize_values_inplace(&mut val);

    let rsize: Vec<usize> = size.iter().cloned().rev().collect();
    let ngrid: usize = rsize.iter().product();

    let mut vg = vec![0.0_f64; ngrid];
    let mut wg = vec![0.0_f64; ngrid];

    match dim {
        1 => inject_data_1d(&mut vg, &mut wg, pts, &val, x0, step, size),
        2 => inject_data_2d(&mut vg, &mut wg, pts, &val, x0, step, size),
        3 => inject_data_3d(&mut vg, &mut wg, pts, &val, x0, step, size),
        _ => unreachable!(),
    }

    let half = get_half_kernel_size_opt(sigma, step, num_iter);
    let kernel_size: Vec<i32> = half.iter().map(|&t| 2 * t + 1).collect();
    let tail_value = get_tail_value(sigma, step, num_iter);

    let iters = num_iter as usize;

    if dim == 1 {
        let arr_len = size[0];
        let rect_len = kernel_size[0] as usize;
        let h = vec![0.0; arr_len];

        let v_new = accumulate_tail_array(vg.clone(), h.clone(), arr_len, rect_len, iters, tail_value[0]);
        let w_new = accumulate_tail_array(wg.clone(), h.clone(), arr_len, rect_len, iters, tail_value[0]);
        vg = v_new;
        wg = w_new;

        let csf = product_conv_scale_factor_opt(&kernel_size, &tail_value, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    } else if dim == 2 {
        let nx = size[0];
        let ny = size[1];

        // x-direction
        {
            let rect_len = kernel_size[0] as usize;
            let h = vec![0.0; nx];

            for j in 0..ny {
                let mut row_v = Vec::with_capacity(nx);
                let mut row_w = Vec::with_capacity(nx);
                for i in 0..nx {
                    let id = idx2(j, i, nx);
                    row_v.push(vg[id]);
                    row_w.push(wg[id]);
                }
                let row_v2 = accumulate_tail_array(row_v, h.clone(), nx, rect_len, iters, tail_value[0]);
                let row_w2 = accumulate_tail_array(row_w, h.clone(), nx, rect_len, iters, tail_value[0]);
                for i in 0..nx {
                    let id = idx2(j, i, nx);
                    vg[id] = row_v2[i];
                    wg[id] = row_w2[i];
                }
            }
        }

        // y-direction
        {
            let rect_len = kernel_size[1] as usize;
            let h = vec![0.0; ny];

            for i in 0..nx {
                let mut col_v = Vec::with_capacity(ny);
                let mut col_w = Vec::with_capacity(ny);
                for j in 0..ny {
                    let id = idx2(j, i, nx);
                    col_v.push(vg[id]);
                    col_w.push(wg[id]);
                }
                let col_v2 = accumulate_tail_array(col_v, h.clone(), ny, rect_len, iters, tail_value[1]);
                let col_w2 = accumulate_tail_array(col_w, h.clone(), ny, rect_len, iters, tail_value[1]);
                for j in 0..ny {
                    let id = idx2(j, i, nx);
                    vg[id] = col_v2[j];
                    wg[id] = col_w2[j];
                }
            }
        }

        let csf = product_conv_scale_factor_opt(&kernel_size, &tail_value, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    } else {
        let nx = size[0];
        let ny = size[1];
        let nz = size[2];

        // x-direction
        {
            let rect_len = kernel_size[0] as usize;
            let h = vec![0.0; nx];

            for k in 0..nz {
                for j in 0..ny {
                    let mut line_v = Vec::with_capacity(nx);
                    let mut line_w = Vec::with_capacity(nx);
                    for i in 0..nx {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_tail_array(line_v, h.clone(), nx, rect_len, iters, tail_value[0]);
                    let w2 = accumulate_tail_array(line_w, h.clone(), nx, rect_len, iters, tail_value[0]);
                    for i in 0..nx {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[i];
                        wg[id] = w2[i];
                    }
                }
            }
        }

        // y-direction
        {
            let rect_len = kernel_size[1] as usize;
            let h = vec![0.0; ny];

            for k in 0..nz {
                for i in 0..nx {
                    let mut line_v = Vec::with_capacity(ny);
                    let mut line_w = Vec::with_capacity(ny);
                    for j in 0..ny {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_tail_array(line_v, h.clone(), ny, rect_len, iters, tail_value[1]);
                    let w2 = accumulate_tail_array(line_w, h.clone(), ny, rect_len, iters, tail_value[1]);
                    for j in 0..ny {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[j];
                        wg[id] = w2[j];
                    }
                }
            }
        }

        // z-direction
        {
            let rect_len = kernel_size[2] as usize;
            let h = vec![0.0; nz];

            for j in 0..ny {
                for i in 0..nx {
                    let mut line_v = Vec::with_capacity(nz);
                    let mut line_w = Vec::with_capacity(nz);
                    for k in 0..nz {
                        let id = idx3(k, j, i, ny, nx);
                        line_v.push(vg[id]);
                        line_w.push(wg[id]);
                    }
                    let v2 = accumulate_tail_array(line_v, h.clone(), nz, rect_len, iters, tail_value[2]);
                    let w2 = accumulate_tail_array(line_w, h.clone(), nz, rect_len, iters, tail_value[2]);
                    for k in 0..nz {
                        let id = idx3(k, j, i, ny, nx);
                        vg[id] = v2[k];
                        wg[id] = w2[k];
                    }
                }
            }
        }

        let csf = product_conv_scale_factor_opt(&kernel_size, &tail_value, sigma, step, num_iter, max_dist_weight);
        threshold_weights(&mut wg, csf);
    }

    let mut out = vec![0.0_f32; ngrid];
    for i in 0..ngrid {
        out[i] = (vg[i] / wg[i] + offset) as f32;
    }
    ArrayD::from_shape_vec(IxDyn(&rsize), out).unwrap()
}

// --------------------------------------
// naive (algorithm A)
// --------------------------------------

fn interpolate_naive(
    pts: &[(f64, f64, f64)],
    mut val: Vec<f64>,
    sigma: &[f64],
    x0: &[f64],
    step: &[f64],
    size: &[usize],
) -> ArrayD<f32> {
    let dim = size.len();
    let offset = normalize_values_inplace(&mut val);

    let rsize: Vec<usize> = size.iter().cloned().rev().collect();
    let ngrid: usize = rsize.iter().product();

    let mut out = vec![0.0_f32; ngrid];
    let scale: Vec<f64> = sigma.iter().map(|&s| 2.0 * s * s).collect();

    if dim == 1 {
        let nx = size[0];
        for i in 0..nx {
            let xc = x0[0] + (i as f64) * step[0];
            let mut ws = 0.0;
            let mut wt = 0.0;
            for (k, &(px, _, _)) in pts.iter().enumerate() {
                let d = (px - xc) * (px - xc) / scale[0];
                let w = (-d).exp();
                ws += w * val[k];
                wt += w;
            }
            out[i] = if wt > 0.0 { (ws / wt + offset) as f32 } else { f32::NAN };
        }
    } else if dim == 2 {
        let nx = size[0];
        let ny = size[1];
        for j in 0..ny {
            let yc = x0[1] + (j as f64) * step[1];
            for i in 0..nx {
                let xc = x0[0] + (i as f64) * step[0];
                let mut ws = 0.0;
                let mut wt = 0.0;
                for (k, &(px, py, _)) in pts.iter().enumerate() {
                    let d = (px - xc) * (px - xc) / scale[0] + (py - yc) * (py - yc) / scale[1];
                    let w = (-d).exp();
                    ws += w * val[k];
                    wt += w;
                }
                out[idx2(j, i, nx)] = if wt > 0.0 { (ws / wt + offset) as f32 } else { f32::NAN };
            }
        }
    } else {
        let nx = size[0];
        let ny = size[1];
        let nz = size[2];
        for k in 0..nz {
            let zc = x0[2] + (k as f64) * step[2];
            for j in 0..ny {
                let yc = x0[1] + (j as f64) * step[1];
                for i in 0..nx {
                    let xc = x0[0] + (i as f64) * step[0];
                    let mut ws = 0.0;
                    let mut wt = 0.0;
                    for (n, &(px, py, pz)) in pts.iter().enumerate() {
                        let d = (px - xc) * (px - xc) / scale[0]
                            + (py - yc) * (py - yc) / scale[1]
                            + (pz - zc) * (pz - zc) / scale[2];
                        let w = (-d).exp();
                        ws += w * val[n];
                        wt += w;
                    }
                    out[idx3(k, j, i, ny, nx)] = if wt > 0.0 { (ws / wt + offset) as f32 } else { f32::NAN };
                }
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&rsize), out).unwrap()
}

// --------------------------------------
// radius (2D, scalar sigma) -- brute-force, result matches kd-tree version
// --------------------------------------
#[derive(Clone, Copy, Debug)]
struct KdNode {
    pt_index: usize,
    left: Option<usize>,
    right: Option<usize>,
    axis: u8, // 0=x, 1=y
}

struct KdTree2D {
    pts: Vec<[f64; 2]>,
    nodes: Vec<KdNode>,
    root: Option<usize>,
}

impl KdTree2D {
    /// Build a balanced kd-tree (median split) from points (x,y).
    fn build(pts_xy: &[[f64; 2]]) -> Self {
        let mut tree = KdTree2D {
            pts: pts_xy.to_vec(),
            nodes: Vec::with_capacity(pts_xy.len()),
            root: None,
        };

        let mut indices: Vec<usize> = (0..pts_xy.len()).collect();
        tree.root = tree.build_rec(&mut indices, 0);
        tree
    }

    fn build_rec(&mut self, indices: &mut [usize], axis: u8) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }
        // sort by axis and pick median; deterministic
        indices.sort_unstable_by(|&a, &b| {
            let va = self.pts[a][axis as usize];
            let vb = self.pts[b][axis as usize];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let pt_index = indices[mid];

        let node_index = self.nodes.len();
        self.nodes.push(KdNode {
            pt_index,
            left: None,
            right: None,
            axis,
        });

        let (left_slice, right_slice) = indices.split_at_mut(mid);
        // right_slice includes median at [0], so skip it
        let right_rest = &mut right_slice[1..];

        let next_axis = 1 - axis;
        let left_child = self.build_rec(left_slice, next_axis);
        let right_child = self.build_rec(right_rest, next_axis);

        self.nodes[node_index].left = left_child;
        self.nodes[node_index].right = right_child;

        Some(node_index)
    }

    /// Radius search: collect neighbours within radius^2.
    /// Output: (pt_index, sqr_dist). Order is generally unordered (same as python).
    fn radius_search(&self, qx: f64, qy: f64, radius_sqr: f64, out: &mut Vec<(usize, f64)>) {
        out.clear();
        if let Some(root) = self.root {
            self.radius_rec(root, qx, qy, radius_sqr, out);
        }
    }

    fn radius_rec(
        &self,
        node_idx: usize,
        qx: f64,
        qy: f64,
        radius_sqr: f64,
        out: &mut Vec<(usize, f64)>,
    ) {
        let node = self.nodes[node_idx];
        let p = self.pts[node.pt_index];
        let axis = node.axis as usize;

        let q = [qx, qy];

        // choose near/far side like python: if coor[c_ind] < pts[node,c_ind] go left first
        let (near, far) = if q[axis] < p[axis] {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(n) = near {
            self.radius_rec(n, qx, qy, radius_sqr, out);
        }

        // hyperplane check: (q[axis] - p[axis])^2 <= radius_sqr
        let da = q[axis] - p[axis];
        if da * da <= radius_sqr {
            // exact sqr dist check
            let dx = qx - p[0];
            let dy = qy - p[1];
            let d2 = dx * dx + dy * dy;
            if d2 <= radius_sqr {
                out.push((node.pt_index, d2));
            }

            if let Some(f) = far {
                self.radius_rec(f, qx, qy, radius_sqr, out);
            }
        }
    }
}

fn interpolate_radius_2d(
    pts: &[(f64, f64, f64)],
    mut val: Vec<f64>,
    sigma: f64,
    x0: &[f64],
    step: &[f64],
    size: &[usize],
    max_dist_weight: f64,
    min_weight: f64,
) -> ArrayD<f32> {
    // normalize
    let offset = normalize_values_inplace(&mut val);

    let nx = size[0];
    let ny = size[1];
    let rsize = vec![ny, nx]; // [y, x]
    let ngrid = nx * ny;

    // Python: search_radius = sqrt(-2 ln(min_weight)) * sigma
    let search_radius = (-2.0 * min_weight.ln()).sqrt() * sigma;
    let radius_sqr = search_radius * search_radius;

    // weight: exp(-d2 / (2*sigma^2))
    let scale = 2.0 * sigma * sigma;

    // build KDTree (filter non-finite consistently)
    let mut pts_xy: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    let mut val_filt: Vec<f64> = Vec::with_capacity(pts.len());

    for (k, &(px, py, _)) in pts.iter().enumerate() {
        let vk = val[k];
        if px.is_finite() && py.is_finite() && vk.is_finite() {
            pts_xy.push([px, py]);
            val_filt.push(vk);
        }
    }
    val = val_filt;

    // Shared read-only KDTree (Your KdTree2D can be synchronized as long as the radius_search does not modify the internal data)
    let kdtree = KdTree2D::build(&pts_xy);

    // output buffer
    let mut out = vec![f32::NAN; ngrid];

    // Parallel fill by "row": out: out[j*nx .. (j+1)*nx]
    out.par_chunks_mut(nx)
        .enumerate()
        .for_each(|(j, row)| {
            let yc = x0[1] + (j as f64) * step[1];

            // Each thread/reuse the neighbor buffer per line to avoid frequent allocation.
            let mut neigh: Vec<(usize, f64)> = Vec::new();

            for i in 0..nx {
                let xc = x0[0] + (i as f64) * step[0];

                kdtree.radius_search(xc, yc, radius_sqr, &mut neigh);

                let mut ws = 0.0_f64;
                let mut wt = 0.0_f64;

                for (k, d2) in neigh.iter().copied() {
                    let w = (-d2 / scale).exp();
                    ws += w * val[k];
                    wt += w;
                }

                row[i] = if wt >= max_dist_weight {
                    (ws / wt + offset) as f32
                } else {
                    f32::NAN
                };
            }
        });

    ArrayD::from_shape_vec(IxDyn(&rsize), out).unwrap()
}


// --------------------------------------
// public PyO3 API: barnes
// --------------------------------------

#[pyfunction]
#[pyo3(signature = (pts, val, sigma, x0, step, size, method="optimized_convolution", num_iter=4, max_dist=3.5, min_weight=0.001))]
pub fn barnes<'py>(
    py: Python<'py>,
    pts: PyReadonlyArrayDyn<f64>,
    val: PyReadonlyArray1<f64>,
    sigma: &Bound<'py, PyAny>,
    x0: &Bound<'py, PyAny>,
    step: &Bound<'py, PyAny>,
    size: &Bound<'py, PyAny>,
    method: &str,
    num_iter: i32,
    max_dist: f64,
    min_weight: f64,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    // pts: accept 1D (N,) => dim=1, or 2D (N,M)
    let pts_arr = pts.as_array();
    let (n, dim) = match pts_arr.ndim() {
        1 => (pts_arr.len(), 1_usize),
        2 => (pts_arr.shape()[0], pts_arr.shape()[1]),
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "expected pts array of shape (N,M) or (N,), got ndim={}",
                pts_arr.ndim()
            )))
        }
    };

    if dim < 1 || dim > 3 {
        return Err(PyRuntimeError::new_err("Barnes interpolation supports only dimensions 1, 2 or 3"));
    }

    let nval = val.len()?; // 现在是 PyResult<usize>
    if nval != n {
        return Err(PyRuntimeError::new_err(format!(
            "pts and val arrays have inconsistent lengths: Npts={n}, Nval={nval}",
        )));
    }

    let sigma_v = py_to_vec_f64(sigma, dim, "sigma")?;
    let x0_v    = py_to_vec_f64(x0,    dim, "x0")?;
    let step_v  = py_to_vec_f64(step,  dim, "step")?;
    let size_v  = py_to_size(size, dim)?;

    // Flatten pts into Vec<(x,y,z)>, fill missing dims with 0.0
    let mut pts_v: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    if dim == 1 {
        // pts can be (N,) or (N,1)
        if pts_arr.ndim() == 1 {
            for i in 0..n {
                pts_v.push((pts_arr[[i]], 0.0, 0.0));
            }
        } else {
            for i in 0..n {
                pts_v.push((pts_arr[[i, 0]], 0.0, 0.0));
            }
        }
    } else if dim == 2 {
        for i in 0..n {
            pts_v.push((pts_arr[[i, 0]], pts_arr[[i, 1]], 0.0));
        }
    } else {
        for i in 0..n {
            pts_v.push((pts_arr[[i, 0]], pts_arr[[i, 1]], pts_arr[[i, 2]]));
        }
    }

    let val_v: Vec<f64> = val.as_slice()?.to_vec();
    let mdw = max_dist_weight(max_dist);

    let out_arr = match method {
        "optimized_convolution" => {
            // same kernel-size < grid checks as python (你也可以加回去更严格)
            interpolate_opt_convolution(&pts_v, val_v, &sigma_v, &x0_v, &step_v, &size_v, num_iter, mdw)
        }
        "convolution" => interpolate_convolution(&pts_v, val_v, &sigma_v, &x0_v, &step_v, &size_v, num_iter, mdw),
        "naive" => interpolate_naive(&pts_v, val_v, &sigma_v, &x0_v, &step_v, &size_v),
        "radius" => {
            if dim != 2 {
                return Err(PyRuntimeError::new_err(format!("radius algorithm works only in 2D but data is {dim}D")));
            }
            if (sigma_v[0] - sigma_v[1]).abs() > 0.0 {
                return Err(PyRuntimeError::new_err(format!(
                    "radius algorithm in 2D works only for scalar sigma value but sigma is: {:?}",
                    sigma_v
                )));
            }
            interpolate_radius_2d(&pts_v, val_v, sigma_v[0], &x0_v, &step_v, &size_v, mdw, min_weight)
        }
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "encountered invalid Barnes interpolation method: {method}"
            )))
        }
    };

    Ok(out_arr.into_pyarray(py).to_owned().into())
}

/// Build a boolean mask for 2D grid:
/// mask[j,i] = True if there exists at least one station within `radius` (in same units as pts/x0/step).
#[pyfunction]
#[pyo3(signature = (pts, x0, step, size, radius))]
pub fn radius_mask_2d<'py>(
    py: Python<'py>,
    pts: PyReadonlyArrayDyn<f64>,   // (N,2) ideally; tolerate (N,>=2)
    x0: PyReadonlyArrayDyn<f64>,    // (2,)
    step: PyReadonlyArrayDyn<f64>,  // (2,) or scalar packed as len=1 (you can enforce (2,) if you want)
    size: PyReadonlyArrayDyn<i64>,  // (2,) [nx, ny] like python wrapper uses
    radius: f64,
) -> PyResult<Py<PyArrayDyn<bool>>> {
    let pts_arr = pts.as_array();
    if pts_arr.ndim() != 2 || pts_arr.shape()[1] < 2 {
        return Err(PyRuntimeError::new_err(
            "pts must be 2D array with shape (N,2) or (N,M>=2)",
        ));
    }

    let x0v = x0.as_array();
    if x0v.len() < 2 {
        return Err(PyRuntimeError::new_err("x0 must have length >= 2"));
    }
    let x0x = x0v[[0]];
    let x0y = x0v[[1]];

    let stepv = step.as_array();
    let (sx, sy) = if stepv.len() == 1 {
        (stepv[[0]], stepv[[0]])
    } else if stepv.len() >= 2 {
        (stepv[[0]], stepv[[1]])
    } else {
        return Err(PyRuntimeError::new_err("step must have length 1 or >= 2"));
    };

    let sizev = size.as_array();
    if sizev.len() < 2 {
        return Err(PyRuntimeError::new_err("size must have length >= 2 (nx, ny)"));
    }
    let nx = sizev[[0]] as usize;
    let ny = sizev[[1]] as usize;

    if nx == 0 || ny == 0 {
        return Err(PyRuntimeError::new_err("nx/ny must be > 0"));
    }

    let radius_sqr = radius * radius;

    // Filter finite points and build KDTree points
    let mut pts_xy: Vec<[f64; 2]> = Vec::with_capacity(pts_arr.shape()[0]);
    for k in 0..pts_arr.shape()[0] {
        let px = pts_arr[[k, 0]];
        let py_ = pts_arr[[k, 1]];
        if px.is_finite() && py_.is_finite() {
            pts_xy.push([px, py_]);
        }
    }

    let kdtree = KdTree2D::build(&pts_xy);

    // output mask (ny, nx)
    let mut out = vec![false; nx * ny];

    out.par_chunks_mut(nx)
        .enumerate()
        .for_each(|(j, row)| {
            let yc = x0y + (j as f64) * sy;

            let mut neigh: Vec<(usize, f64)> = Vec::new();

            for i in 0..nx {
                let xc = x0x + (i as f64) * sx;
                kdtree.radius_search(xc, yc, radius_sqr, &mut neigh);
                row[i] = !neigh.is_empty();
            }
        });

    let arr = ArrayD::from_shape_vec(IxDyn(&[ny, nx]), out)
        .map_err(|e| PyRuntimeError::new_err(format!("shape error: {e}")))?;

    Ok(arr.into_pyarray(py).to_owned().into())
}
