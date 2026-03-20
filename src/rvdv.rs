use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn is_missing(v: f64, xmsg: f64) -> bool {
    if xmsg.is_nan() { v.is_nan() } else { v == xmsg }
}

#[inline]
fn set_missing(out: &mut [f64], xmsg: f64) {
    if xmsg.is_nan() {
        for o in out.iter_mut() {
            *o = f64::NAN;
        }
    } else {
        for o in out.iter_mut() {
            *o = xmsg;
        }
    }
}

/// Fortran: DUSEAVE(X, MLON, XMSG)
/// +/-90° 行：对非缺测取平均，然后整行填充该平均值
fn duseave_row(row: &mut [f64], xmsg: f64) {
    let mut ave = 0.0f64;
    let mut cnt = 0.0f64;
    for &v in row.iter() {
        if !is_missing(v, xmsg) {
            ave += v;
            cnt += 1.0;
        }
    }
    if cnt != 0.0 {
        ave /= cnt;
        for v in row.iter_mut() {
            *v = ave;
        }
    }
}

/// Fortran: DLNEXTRP(X, MLON, NLAT, XMSG)
/// 仅用于 JOPT==2 时角点缺测的线性外推（两方向平均）
fn dlnextrp_corners(x: &mut Array2<f64>, xmsg: f64) {
    let (nlat, mlon) = x.dim();
    if nlat < 3 || mlon < 3 {
        return;
    }

    // corners in Fortran are (ml,nl): (1,1), (mlon,1), (1,nlat), (mlon,nlat)
    // Here ndarray is (nl,ml): (0,0), (0,mlon-1), (nlat-1,0), (nlat-1,mlon-1)
    let corners = [
        (0usize, 0usize),
        (0usize, mlon - 1),
        (nlat - 1, 0usize),
        (nlat - 1, mlon - 1),
    ];

    for (nl, ml) in corners {
        if !is_missing(x[(nl, ml)], xmsg) {
            continue;
        }

        if nl == 0 && ml == 0 {
            // (1,1): need (1,2),(1,3),(2,1),(3,1) in Fortran
            let a1 = x[(1, 0)];
            let a2 = x[(2, 0)];
            let b1 = x[(0, 1)];
            let b2 = x[(0, 2)];
            if !is_missing(a1, xmsg)
                && !is_missing(a2, xmsg)
                && !is_missing(b1, xmsg)
                && !is_missing(b2, xmsg)
            {
                x[(nl, ml)] = (2.0 * a1 - a2 + 2.0 * b1 - b2) * 0.5;
            }
        } else if nl == 0 && ml == mlon - 1 {
            // (mlon,1): need (mlon,2),(mlon,3),(mlon-1,1),(mlon-2,1)
            let a1 = x[(1, ml)];
            let a2 = x[(2, ml)];
            let b1 = x[(0, ml - 1)];
            let b2 = x[(0, ml - 2)];
            if !is_missing(a1, xmsg)
                && !is_missing(a2, xmsg)
                && !is_missing(b1, xmsg)
                && !is_missing(b2, xmsg)
            {
                x[(nl, ml)] = (2.0 * a1 - a2 + 2.0 * b1 - b2) * 0.5;
            }
        } else if nl == nlat - 1 && ml == 0 {
            // (1,nlat): need (1,nlat-1),(1,nlat-2),(2,nlat),(3,nlat)
            let a1 = x[(nl - 1, 0)];
            let a2 = x[(nl - 2, 0)];
            let b1 = x[(nl, 1)];
            let b2 = x[(nl, 2)];
            if !is_missing(a1, xmsg)
                && !is_missing(a2, xmsg)
                && !is_missing(b1, xmsg)
                && !is_missing(b2, xmsg)
            {
                x[(nl, ml)] = (2.0 * a1 - a2 + 2.0 * b1 - b2) * 0.5;
            }
        } else if nl == nlat - 1 && ml == mlon - 1 {
            // (mlon,nlat): need (mlon,nlat-1),(mlon,nlat-2),(mlon-1,nlat),(mlon-2,nlat)
            let a1 = x[(nl - 1, ml)];
            let a2 = x[(nl - 2, ml)];
            let b1 = x[(nl, ml - 1)];
            let b2 = x[(nl, ml - 2)];
            if !is_missing(a1, xmsg)
                && !is_missing(a2, xmsg)
                && !is_missing(b1, xmsg)
                && !is_missing(b2, xmsg)
            {
                x[(nl, ml)] = (2.0 * a1 - a2 + 2.0 * b1 - b2) * 0.5;
            }
        }
    }
}

fn calc_dvrfidf_core(
    u: &ndarray::ArrayView2<f64>, // (nlat, mlon)
    v: &ndarray::ArrayView2<f64>, // (nlat, mlon)
    glat: &[f64],
    glon: &[f64],
    xmsg: f64,
    iopt: i32,
    re: f64,
) -> (Array2<f64>, i32) {
    let (nlat, mlon) = u.dim();
    if mlon < 1 || nlat < 1 {
        return (Array2::<f64>::zeros((nlat, mlon)), 1);
    }
    if glat.len() != nlat || glon.len() != mlon {
        return (Array2::<f64>::zeros((nlat, mlon)), 10);
    }
    if glat.first().unwrap().abs() > 90.0 || glat.last().unwrap().abs() > 90.0 {
        return (Array2::<f64>::zeros((nlat, mlon)), 2);
    }
    if mlon < 3 || nlat < 2 {
        // Fortran 会用到 GLON(3)、GLAT(2) 等
        return (Array2::<f64>::zeros((nlat, mlon)), 11);
    }

    let rad = 4.0 * (1.0f64).atan() / 180.0;
    let rcon = re * rad;
    let jopt = iopt.abs();

    let mut clat = vec![0.0f64; nlat];
    let mut tlatre = vec![0.0f64; nlat];
    for nl in 0..nlat {
        clat[nl] = (rad * glat[nl]).cos();
    }
    for nl in 0..nlat {
        if glat[nl].abs() < 90.0 {
            tlatre[nl] = (rad * glat[nl]).tan() / re;
        } else {
            // pole: use neighbor average latitude like Fortran
            if glat[nl] == 90.0 {
                let polat = 0.5 * (glat[nl] + glat[nl - 1]);
                tlatre[nl] = (rad * polat).tan() / re;
            } else {
                let polat = 0.5 * (glat[nl] + glat[nl + 1]);
                tlatre[nl] = (rad * polat).tan() / re;
            }
        }
    }

    let mut rv = Array2::<f64>::zeros((nlat, mlon));
    {
        // init to msg
        let slice = rv.as_slice_mut().unwrap();
        set_missing(slice, xmsg);
    }

    let dybot = 1.0 / (rcon * (glat[1] - glat[0]));
    let dytop = 1.0 / (rcon * (glat[nlat - 1] - glat[nlat - 2]));
    let mut dy2 = vec![0.0f64; nlat];
    for nl in 1..(nlat - 1) {
        dy2[nl] = 1.0 / (rcon * (glat[nl + 1] - glat[nl - 1]));
    }

    let dlon = glon[1] - glon[0];
    let dlon2 = glon[2] - glon[0];
    let mut dx = vec![0.0f64; nlat];
    let mut dx2 = vec![0.0f64; nlat];
    for nl in 0..nlat {
        if glat[nl].abs() != 90.0 {
            dx[nl] = 1.0 / (rcon * dlon * clat[nl]);
            dx2[nl] = 1.0 / (rcon * dlon2 * clat[nl]);
        } else {
            dx[nl] = 0.0;
            dx2[nl] = 0.0;
        }
    }

    let (mlstrt, mlend) = if jopt == 1 || jopt == 3 {
        (0usize, mlon - 1)
    } else {
        (1usize, mlon - 2)
    };

    // longitude loop
    for ml in mlstrt..=mlend {
        let mlm1 = if ml == 0 { mlon - 1 } else { ml - 1 };
        let mlp1 = if ml == mlon - 1 { 0 } else { ml + 1 };
        // For non-cyclic case, when mlstrt=1..mlon-2, mlm1/mlp1 are safe too

        // body: nl=2..nlat-1 (Fortran) => 1..nlat-2
        for nl in 1..(nlat - 1) {
            let cond = !is_missing(v[(nl, mlp1)], xmsg)
                && !is_missing(v[(nl, mlm1)], xmsg)
                && !is_missing(u[(nl + 1, ml)], xmsg)
                && !is_missing(u[(nl - 1, ml)], xmsg)
                && !is_missing(u[(nl, ml)], xmsg);

            if cond {
                rv[(nl, ml)] = (v[(nl, mlp1)] - v[(nl, mlm1)]) * dx2[nl]
                    - (u[(nl + 1, ml)] - u[(nl - 1, ml)]) * dy2[nl]
                    + u[(nl, ml)] * tlatre[nl];
            }
        }

        if jopt >= 2 {
            // bottom (nl=1 => 0)
            let nl = 0usize;
            let cond = !is_missing(v[(nl, mlp1)], xmsg)
                && !is_missing(v[(nl, mlm1)], xmsg)
                && !is_missing(u[(nl + 1, ml)], xmsg)
                && !is_missing(u[(nl, ml)], xmsg);
            if cond {
                rv[(nl, ml)] = (v[(nl, mlp1)] - v[(nl, mlm1)]) * dx2[nl]
                    - (u[(nl + 1, ml)] - u[(nl, ml)]) * dybot
                    + u[(nl, ml)] * tlatre[nl];
            }

            // top (nl=nlat => nlat-1)
            let nl = nlat - 1;
            let cond = !is_missing(v[(nl, mlp1)], xmsg)
                && !is_missing(v[(nl, mlm1)], xmsg)
                && !is_missing(u[(nl, ml)], xmsg)
                && !is_missing(u[(nl - 1, ml)], xmsg);
            if cond {
                rv[(nl, ml)] = (v[(nl, mlp1)] - v[(nl, mlm1)]) * dx2[nl]
                    - (u[(nl, ml)] - u[(nl - 1, ml)]) * dytop
                    + u[(nl, ml)] * tlatre[nl];
            }
        }
    }

    // left/right bound for jopt==2
    if jopt == 2 {
        for nl in 1..(nlat - 1) {
            // left ml=0
            let cond = !is_missing(v[(nl, 1)], xmsg)
                && !is_missing(v[(nl, 0)], xmsg)
                && !is_missing(u[(nl + 1, 0)], xmsg)
                && !is_missing(u[(nl - 1, 0)], xmsg)
                && !is_missing(u[(nl, 0)], xmsg);
            if cond {
                rv[(nl, 0)] = (v[(nl, 1)] - v[(nl, 0)]) * dx[nl]
                    - (u[(nl + 1, 0)] - u[(nl - 1, 0)]) * dy2[nl]
                    + u[(nl, 0)] * tlatre[nl];
            }

            // right ml=mlon-1
            let ml = mlon - 1;
            let cond = !is_missing(v[(nl, ml)], xmsg)
                && !is_missing(v[(nl, ml - 1)], xmsg)
                && !is_missing(u[(nl + 1, ml)], xmsg)
                && !is_missing(u[(nl - 1, ml)], xmsg)
                && !is_missing(u[(nl, ml)], xmsg);
            if cond {
                rv[(nl, ml)] = (v[(nl, ml)] - v[(nl, ml - 1)]) * dx[nl]
                    - (u[(nl + 1, ml)] - u[(nl - 1, ml)]) * dy2[nl]
                    + u[(nl, ml)] * tlatre[nl];
            }
        }
    }

    // special at +/-90: use average across lon
    for &nl in &[0usize, nlat - 1] {
        if glat[nl].abs() == 90.0 {
            let mut row = rv.row_mut(nl);
            duseave_row(row.as_slice_mut().unwrap(), xmsg);
        }
    }

    // corners extrapolation for jopt==2
    if jopt == 2 {
        dlnextrp_corners(&mut rv, xmsg);
    }

    (rv, 0)
}

fn calc_ddvfidf_core(
    u: &ndarray::ArrayView2<f64>, // (nlat, mlon)
    v: &ndarray::ArrayView2<f64>, // (nlat, mlon)
    glat: &[f64],
    glon: &[f64],
    xmsg: f64,
    iopt: i32,
    re: f64,
) -> (Array2<f64>, i32) {
    let (nlat, mlon) = u.dim();
    if mlon < 1 || nlat < 1 {
        return (Array2::<f64>::zeros((nlat, mlon)), 1);
    }
    if glat.len() != nlat || glon.len() != mlon {
        return (Array2::<f64>::zeros((nlat, mlon)), 10);
    }
    if glat.first().unwrap().abs() > 90.0 || glat.last().unwrap().abs() > 90.0 {
        return (Array2::<f64>::zeros((nlat, mlon)), 2);
    }
    if mlon < 3 || nlat < 2 {
        return (Array2::<f64>::zeros((nlat, mlon)), 11);
    }

    let rad = 4.0 * (1.0f64).atan() / 180.0;
    let rcon = re * rad;
    let jopt = iopt.abs();

    let mut clat = vec![0.0f64; nlat];
    let mut tlatre = vec![0.0f64; nlat];
    for nl in 0..nlat {
        clat[nl] = (rad * glat[nl]).cos();
    }
    for nl in 0..nlat {
        if glat[nl].abs() < 90.0 {
            tlatre[nl] = (rad * glat[nl]).tan() / re;
        } else {
            if glat[nl] == 90.0 {
                let polat = 0.5 * (glat[nl] + glat[nl - 1]);
                tlatre[nl] = (rad * polat).tan() / re;
            } else {
                let polat = 0.5 * (glat[nl] + glat[nl + 1]);
                tlatre[nl] = (rad * polat).tan() / re;
            }
        }
    }

    let mut dv = Array2::<f64>::zeros((nlat, mlon));
    set_missing(dv.as_slice_mut().unwrap(), xmsg);

    let dybot = 1.0 / (rcon * (glat[1] - glat[0]));
    let dytop = 1.0 / (rcon * (glat[nlat - 1] - glat[nlat - 2]));
    let mut dy2 = vec![0.0f64; nlat];
    for nl in 1..(nlat - 1) {
        dy2[nl] = 1.0 / (rcon * (glat[nl + 1] - glat[nl - 1]));
    }

    let dlon = glon[1] - glon[0];
    let dlon2 = glon[2] - glon[0];
    let mut dx = vec![0.0f64; nlat];
    let mut dx2 = vec![0.0f64; nlat];
    for nl in 0..nlat {
        if glat[nl].abs() != 90.0 {
            dx[nl] = 1.0 / (rcon * dlon * clat[nl]);
            dx2[nl] = 1.0 / (rcon * dlon2 * clat[nl]);
        } else {
            dx[nl] = 0.0;
            dx2[nl] = 0.0;
        }
    }

    let (mlstrt, mlend) = if jopt == 1 || jopt == 3 {
        (0usize, mlon - 1)
    } else {
        (1usize, mlon - 2)
    };

    for ml in mlstrt..=mlend {
        let mlm1 = if ml == 0 { mlon - 1 } else { ml - 1 };
        let mlp1 = if ml == mlon - 1 { 0 } else { ml + 1 };

        for nl in 1..(nlat - 1) {
            let cond = !is_missing(v[(nl + 1, ml)], xmsg)
                && !is_missing(v[(nl - 1, ml)], xmsg)
                && !is_missing(u[(nl, mlp1)], xmsg)
                && !is_missing(u[(nl, mlm1)], xmsg)
                && !is_missing(v[(nl, ml)], xmsg);
            if cond {
                dv[(nl, ml)] = (v[(nl + 1, ml)] - v[(nl - 1, ml)]) * dy2[nl]
                    + (u[(nl, mlp1)] - u[(nl, mlm1)]) * dx2[nl]
                    - v[(nl, ml)] * tlatre[nl];
            }
        }

        if jopt >= 2 {
            // bottom nl=0
            let cond = !is_missing(v[(1, ml)], xmsg)
                && !is_missing(v[(0, ml)], xmsg)
                && !is_missing(u[(0, mlp1)], xmsg)
                && !is_missing(u[(0, mlm1)], xmsg);
            if cond {
                dv[(0, ml)] = (v[(1, ml)] - v[(0, ml)]) * dybot
                    + (u[(0, mlp1)] - u[(0, mlm1)]) * dx2[0]
                    - v[(0, ml)] * tlatre[0];
            }

            // top nl=nlat-1
            let nl = nlat - 1;
            let cond = !is_missing(v[(nl, ml)], xmsg)
                && !is_missing(v[(nl - 1, ml)], xmsg)
                && !is_missing(u[(nl, mlp1)], xmsg)
                && !is_missing(u[(nl, mlm1)], xmsg);
            if cond {
                dv[(nl, ml)] = (v[(nl, ml)] - v[(nl - 1, ml)]) * dytop
                    + (u[(nl, mlp1)] - u[(nl, mlm1)]) * dx2[nl]
                    - v[(nl, ml)] * tlatre[nl];
            }
        }
    }

    if jopt == 2 {
        for nl in 1..(nlat - 1) {
            // left ml=0
            let cond = !is_missing(v[(nl + 1, 0)], xmsg)
                && !is_missing(v[(nl - 1, 0)], xmsg)
                && !is_missing(u[(nl, 1)], xmsg)
                && !is_missing(u[(nl, 0)], xmsg)
                && !is_missing(v[(nl, 0)], xmsg);
            if cond {
                dv[(nl, 0)] = (v[(nl + 1, 0)] - v[(nl - 1, 0)]) * dy2[nl]
                    + (u[(nl, 1)] - u[(nl, 0)]) * dx[nl]
                    - v[(nl, 0)] * tlatre[nl];
            }

            // right ml=mlon-1
            let m = mlon - 1;
            let cond = !is_missing(v[(nl + 1, m)], xmsg)
                && !is_missing(v[(nl - 1, m)], xmsg)
                && !is_missing(u[(nl, m)], xmsg)
                && !is_missing(u[(nl, m - 1)], xmsg)
                && !is_missing(v[(nl, m)], xmsg);
            if cond {
                dv[(nl, m)] = (v[(nl + 1, m)] - v[(nl - 1, m)]) * dy2[nl]
                    + (u[(nl, m)] - u[(nl, m - 1)]) * dx[nl]
                    - v[(nl, m)] * tlatre[nl];
            }
        }
    }

    for &nl in &[0usize, nlat - 1] {
        if glat[nl].abs() == 90.0 {
            let mut row = dv.row_mut(nl);
            duseave_row(row.as_slice_mut().unwrap(), xmsg);
        }
    }

    if jopt == 2 {
        dlnextrp_corners(&mut dv, xmsg);
    }

    (dv, 0)
}

#[pyfunction]
pub fn dvrfidf(
    py: Python<'_>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    glat: PyReadonlyArray1<f64>,
    glon: PyReadonlyArray1<f64>,
    iopt: i32,
    xmsg: Option<f64>,
    re: f64,
) -> PyResult<(Py<PyArray2<f64>>, i32)> {
    let u = u.as_array();
    let v = v.as_array();
    if u.dim() != v.dim() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u and v must have the same shape (nlat, mlon).",
        ));
    }
    let xmsg = xmsg.unwrap_or(f64::NAN);
    let (out, ier) = calc_dvrfidf_core(&u, &v, glat.as_slice()?, glon.as_slice()?, xmsg, iopt, re);
    Ok((out.into_pyarray(py).to_owned().into(), ier))
}

#[pyfunction]
pub fn ddvfidf(
    py: Python<'_>,
    u: PyReadonlyArray2<f64>,
    v: PyReadonlyArray2<f64>,
    glat: PyReadonlyArray1<f64>,
    glon: PyReadonlyArray1<f64>,
    iopt: i32,
    xmsg: Option<f64>,
    re: f64,
) -> PyResult<(Py<PyArray2<f64>>, i32)> {
    let u = u.as_array();
    let v = v.as_array();
    if u.dim() != v.dim() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u and v must have the same shape (nlat, mlon).",
        ));
    }
    let xmsg = xmsg.unwrap_or(f64::NAN);
    let (out, ier) = calc_ddvfidf_core(&u, &v, glat.as_slice()?, glon.as_slice()?, xmsg, iopt, re);
    Ok((out.into_pyarray(py).to_owned().into(), ier))
}

#[pyfunction]
pub fn ddvfidf_batch(
    py: Python<'_>,
    u: PyReadonlyArrayDyn<f64>, // (..., nlat, mlon)
    v: PyReadonlyArrayDyn<f64>,
    glat: PyReadonlyArray1<f64>, // (nlat,)
    glon: PyReadonlyArray1<f64>, // (mlon,)
    iopt: i32,
    xmsg: f64, // 通常 np.nan
    re: f64,   // ✅ R 从 python 传入
) -> PyResult<(Py<PyArrayDyn<f64>>, i32)> {
    let u = u.as_array();
    let v = v.as_array();
    if u.shape() != v.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u and v must have the same shape",
        ));
    }
    if u.ndim() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u/v must have ndim>=2",
        ));
    }
    let nlat = u.shape()[u.ndim() - 2];
    let mlon = u.shape()[u.ndim() - 1];

    let glat = glat.as_slice()?;
    let glon = glon.as_slice()?;

    // 展平 batch 维度：batch = prod(shape[..-2])
    let batch: usize = u.shape()[..u.ndim() - 2].iter().product();
    let stride = nlat * mlon;

    // 要求 C contiguous；如果不是，强制拷贝成 contiguous（避免 stride 坑）
    let u_cow = if u.is_standard_layout() {
        None
    } else {
        Some(u.to_owned())
    };
    let v_cow = if v.is_standard_layout() {
        None
    } else {
        Some(v.to_owned())
    };

    let u_data: &[f64] = match &u_cow {
        Some(a) => a.as_slice().unwrap(),
        None => u
            .as_slice()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("u must be C-contiguous"))?,
    };
    let v_data: &[f64] = match &v_cow {
        Some(a) => a.as_slice().unwrap(),
        None => v
            .as_slice()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("v must be C-contiguous"))?,
    };

    let mut out = vec![0.0f64; batch * stride];

    // ✅ 释放 GIL + ✅ 并行按 batch 切片计算
    let iers: Vec<i32> = {
        out.par_chunks_mut(stride)
            .enumerate()
            .map(|(b, out_chunk)| {
                let s = b * stride;
                let u_view =
                    ndarray::ArrayView2::from_shape((nlat, mlon), &u_data[s..s + stride]).unwrap();
                let v_view =
                    ndarray::ArrayView2::from_shape((nlat, mlon), &v_data[s..s + stride]).unwrap();

                let (res, ier) = calc_ddvfidf_core(&u_view, &v_view, glat, glon, xmsg, iopt, re);

                out_chunk.copy_from_slice(res.as_slice().unwrap());
                ier
            })
            .collect()
    };

    let ier = iers.into_iter().find(|&x| x != 0).unwrap_or(0);

    // reshape 回原 shape
    let out_shape = u.shape().to_vec();
    let out_arr = ndarray::ArrayD::from_shape_vec(out_shape, out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("reshape failed: {e}")))?;

    Ok((out_arr.into_pyarray(py).to_owned().into(), ier))
}

#[pyfunction]
pub fn dvrfidf_batch(
    py: Python<'_>,
    u: PyReadonlyArrayDyn<f64>,
    v: PyReadonlyArrayDyn<f64>,
    glat: PyReadonlyArray1<f64>,
    glon: PyReadonlyArray1<f64>,
    iopt: i32,
    xmsg: f64,
    re: f64,
) -> PyResult<(Py<PyArrayDyn<f64>>, i32)> {
    let u = u.as_array();
    let v = v.as_array();
    if u.shape() != v.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u and v must have the same shape",
        ));
    }
    if u.ndim() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u/v must have ndim>=2",
        ));
    }
    let nlat = u.shape()[u.ndim() - 2];
    let mlon = u.shape()[u.ndim() - 1];

    let glat = glat.as_slice()?;
    let glon = glon.as_slice()?;

    // 展平 batch 维度：batch = prod(shape[..-2])
    let batch: usize = u.shape()[..u.ndim() - 2].iter().product();
    let stride = nlat * mlon;

    // 要求 C contiguous；如果不是，强制拷贝成 contiguous（避免 stride 坑）
    let u_cow = if u.is_standard_layout() {
        None
    } else {
        Some(u.to_owned())
    };
    let v_cow = if v.is_standard_layout() {
        None
    } else {
        Some(v.to_owned())
    };

    let u_data: &[f64] = match &u_cow {
        Some(a) => a.as_slice().unwrap(),
        None => u
            .as_slice()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("u must be C-contiguous"))?,
    };
    let v_data: &[f64] = match &v_cow {
        Some(a) => a.as_slice().unwrap(),
        None => v
            .as_slice()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("v must be C-contiguous"))?,
    };

    let mut out = vec![0.0f64; batch * stride];

    // ✅ 释放 GIL + ✅ 并行按 batch 切片计算
    let iers: Vec<i32> = {
        out.par_chunks_mut(stride)
            .enumerate()
            .map(|(b, out_chunk)| {
                let s = b * stride;
                let u_view =
                    ndarray::ArrayView2::from_shape((nlat, mlon), &u_data[s..s + stride]).unwrap();
                let v_view =
                    ndarray::ArrayView2::from_shape((nlat, mlon), &v_data[s..s + stride]).unwrap();

                let (res, ier) = calc_dvrfidf_core(&u_view, &v_view, glat, glon, xmsg, iopt, re);

                out_chunk.copy_from_slice(res.as_slice().unwrap());
                ier
            })
            .collect()
    };

    let ier = iers.into_iter().find(|&x| x != 0).unwrap_or(0);

    // reshape 回原 shape
    let out_shape = u.shape().to_vec();
    let out_arr = ndarray::ArrayD::from_shape_vec(out_shape, out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("reshape failed: {e}")))?;

    Ok((out_arr.into_pyarray(py).to_owned().into(), ier))
}
