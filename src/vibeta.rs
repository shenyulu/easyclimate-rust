use numpy::{
    PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;

/// Port of Fortran DVIBETA (double precision) from vibeta_dp.f
/// Returns (vint, ier) for a single profile
#[pyfunction]
pub fn dvibeta(
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArray1<f64>,
    xmsg: f64,
    linlog: i32,
    psfc: f64,
    xsfc: f64,
    pbot: f64,
    ptop: f64,
    plvcrt: f64,
) -> PyResult<(f64, i32)> {
    let p = p.as_slice()?;
    let x = x.as_slice()?;

    if p.len() != x.len() {
        return Ok((xmsg, 1_000_000_000)); // shape mismatch
    }
    let nlev = p.len() as i32;

    let (vint, ier) = WS.with(|ws| {
        let mut ws = ws.borrow_mut();
        dvibeta_core_ws(
            p, x, nlev, xmsg, linlog, psfc, xsfc, pbot, ptop, plvcrt, &mut *ws,
        )
    });

    Ok((vint, ier))
}

/// Batch API:
/// p: (nlev,)
/// x: (..., nlev)  MUST be C-contiguous
/// psfc: (...)     MUST be C-contiguous, shape matches x without last dim
/// Returns:
///   vint: (...) float64
///   ier:  (...) int32
///
/// Notes:
/// - computes xsfc from x[...,0] (same as your python wrapper logic)
/// - pbot = psfc, plvcrt = ptop (matches your current usage)
/// - replaces NaN/Inf in x with xmsg (Fortran-style missing handling)
#[pyfunction]
pub fn dvibeta_batch(
    py: Python<'_>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArrayDyn<f64>,
    psfc: PyReadonlyArrayDyn<f64>,
    xmsg: f64,
    linlog: i32,
    ptop: f64,
) -> PyResult<(Py<PyArrayDyn<f64>>, Py<PyArrayDyn<i32>>)> {
    let p = p.as_slice()?;
    let nlev = p.len();
    if nlev == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("p is empty"));
    }

    // Require contiguous
    let x_slice = x
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("x must be C-contiguous"))?;
    let psfc_slice = psfc
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("psfc must be C-contiguous"))?;

    // shape() needs PyUntypedArrayMethods in scope
    let x_shape = x.shape().to_vec(); // Vec<usize>
    let ps_shape = psfc.shape().to_vec(); // Vec<usize>

    if x_shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least 1 dimension (..., nlev)",
        ));
    }
    let x_nlev = x_shape[x_shape.len() - 1];
    if x_nlev != nlev {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x last dim ({x_nlev}) != p length ({nlev})"
        )));
    }

    // psfc shape must equal x shape without last dim
    if ps_shape.len() + 1 != x_shape.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "psfc shape must match x shape without last dim",
        ));
    }
    for i in 0..ps_shape.len() {
        if ps_shape[i] != x_shape[i] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "psfc shape must match x shape without last dim",
            ));
        }
    }

    let npoints = psfc_slice.len();
    if npoints * nlev != x_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x size is inconsistent with psfc size and p length",
        ));
    }

    // outputs (flat)
    let mut out = vec![f64::NAN; npoints];
    let mut ier_out = vec![0i32; npoints];

    // Parallel over points
    out.par_iter_mut()
        .zip(ier_out.par_iter_mut())
        .enumerate()
        .for_each(|(i, (out_i, ier_i))| {
            let ps = psfc_slice[i];
            if !(ps > ptop) {
                *out_i = f64::NAN;
                *ier_i = -10;
                return;
            }

            let base = i * nlev;
            let prof = &x_slice[base..base + nlev];

            let (vint, ier) = WS.with(|ws_cell| {
                let mut ws = ws_cell.borrow_mut();

                // copy profile into workspace buffer, replace NaN/Inf
                for k in 0..nlev {
                    let v = prof[k];
                    ws.prof[k] = if v.is_finite() { v } else { xmsg };
                }

                let xsfc = {
                    let v0 = ws.prof[0];
                    if (v0 - xmsg).abs() > 1e10 { v0 } else { 0.0 }
                };

                // IMPORTANT: detach x_in from ws to satisfy borrow checker
                let xbuf = ws.prof; // copy [f64; KLVL]
                let x_in = &xbuf[..nlev]; // slice into local copy

                dvibeta_core_ws(
                    p,
                    x_in,
                    nlev as i32,
                    xmsg,
                    linlog,
                    ps,
                    xsfc,
                    ps,
                    ptop,
                    ptop,
                    &mut *ws,
                )
            });

            if ier == 0 {
                *out_i = vint;
            } else {
                *out_i = f64::NAN;
            }
            *ier_i = ier;
        });

    // ---- create numpy outputs ----
    // numpy 0.27: use PyArray1::from_vec, then reshape to dyn
    let out_1d = PyArray1::<f64>::from_vec(py, out);
    let out_dyn = out_1d.reshape(ps_shape.clone())?.to_owned();

    let ier_1d = PyArray1::<i32>::from_vec(py, ier_out);
    let ier_dyn = ier_1d.reshape(ps_shape)?.to_owned();

    Ok((out_dyn.into(), ier_dyn.into()))
}

// =======================
// Internal implementation
// =======================

const KLVL: usize = 300;
const NMAX: usize = 200;

#[derive(Clone)]
struct VibetaWorkspace {
    // dvibeta core arrays (Fortran-like 1-based usage)
    pp: [f64; KLVL + 1],
    xx: [f64; KLVL + 1],
    pi: [f64; KLVL + 1],
    xi: [f64; KLVL + 1],
    delp: [f64; KLVL + 1],
    beta: [f64; KLVL + 1],

    // temp buffers for insertion/copy
    tmp1: [f64; KLVL + 1],
    tmp2: [f64; KLVL + 1],

    // interpolation output buffer
    xout: [f64; KLVL + 1],

    // dint2p2 filtered arrays
    p200: [f64; NMAX],
    x200: [f64; NMAX],
    pln200: [f64; NMAX],

    // profile buffer (batch copies here)
    prof: [f64; KLVL],

    // constant-1 profile for normalization (avoid per-call alloc)
    ones: [f64; KLVL],
}

impl Default for VibetaWorkspace {
    fn default() -> Self {
        Self {
            pp: [0.0; KLVL + 1],
            xx: [0.0; KLVL + 1],
            pi: [0.0; KLVL + 1],
            xi: [0.0; KLVL + 1],
            delp: [0.0; KLVL + 1],
            beta: [0.0; KLVL + 1],
            tmp1: [0.0; KLVL + 1],
            tmp2: [0.0; KLVL + 1],
            xout: [0.0; KLVL + 1],
            p200: [0.0; NMAX],
            x200: [0.0; NMAX],
            pln200: [0.0; NMAX],
            prof: [0.0; KLVL],
            ones: [1.0; KLVL],
        }
    }
}

thread_local! {
    static WS: RefCell<VibetaWorkspace> = RefCell::new(VibetaWorkspace::default());
}

#[inline(always)]
fn aint(v: f64) -> f64 {
    v.trunc()
}

#[inline]
fn dint2p2_ws(
    pin: &[f64],
    xin: &[f64],
    pout: &[f64],
    xout: &mut [f64],
    iflag: i32,
    xmsg: f64,
    p200: &mut [f64; NMAX],
    x200: &mut [f64; NMAX],
    pln200: &mut [f64; NMAX],
) -> i32 {
    let npin = pin.len();
    let npout = xout.len();

    let mut ier = 0;
    if npin < 1 || npout < 1 {
        ier += 1;
    }
    if iflag < 1 || iflag > 2 {
        ier += 10;
    }
    if npin > NMAX {
        ier += 100;
    }
    if ier != 0 {
        for v in xout.iter_mut() {
            *v = xmsg;
        }
        return ier;
    }

    // filter missing
    let mut nlmax = 0usize;
    for i in 0..npin {
        let pi = pin[i];
        let xi = xin[i];
        if xi != xmsg && pi != xmsg {
            p200[nlmax] = pi;
            x200[nlmax] = xi;
            nlmax += 1;
        }
    }
    if nlmax == 0 {
        for v in xout.iter_mut() {
            *v = xmsg;
        }
        return 1000;
    }

    if iflag == 2 {
        for i in 0..nlmax {
            pln200[i] = p200[i].ln();
        }
    }

    for np in 0..npout {
        let pp = pout[np];
        xout[np] = xmsg;

        if iflag == 2 {
            let pp_ln = pp.ln();
            for l in 0..nlmax {
                if pp == p200[l] {
                    xout[np] = x200[l];
                } else if l + 1 < nlmax && pp < p200[l] && pp > p200[l + 1] {
                    let pa = pln200[l];
                    let pc = pln200[l + 1];
                    let slope = (x200[l] - x200[l + 1]) / (pa - pc);
                    xout[np] = x200[l + 1] + slope * (pp_ln - pc);
                    break;
                }
            }
        } else {
            for l in 0..nlmax {
                if pp == p200[l] {
                    xout[np] = x200[l];
                } else if l + 1 < nlmax && pp < p200[l] && pp > p200[l + 1] {
                    let slope = (x200[l] - x200[l + 1]) / (p200[l] - p200[l + 1]);
                    xout[np] = x200[l + 1] + slope * (pp - p200[l + 1]);
                    break;
                }
            }
        }
    }

    0
}

fn dvibeta_core_ws(
    p_in: &[f64],
    x_in: &[f64],
    nlev: i32,
    xmsg: f64,
    linlog: i32,
    psfc: f64,
    _xsfc: f64,
    pbot: f64,
    ptop: f64,
    _plvcrt: f64,
    ws: &mut VibetaWorkspace,
) -> (f64, i32) {
    let zero = 0.0f64;
    let pzero = 0.0f64;
    let xzero = 0.0f64;

    let mut ier = 0i32;
    if nlev < 3 {
        ier += 1;
    }
    if KLVL < 2 * (nlev as usize) {
        ier += 10;
    }
    if psfc == xmsg {
        ier += 100;
    }
    if ptop >= pbot {
        ier += 10000;
    }
    if ier != 0 {
        return (xmsg, ier);
    }

    ws.pp.fill(xmsg);
    ws.xx.fill(0.0);

    let mut nll = 0usize;
    let mut nlx = 0usize;
    let mut pflag = false;

    for nl in 0..(nlev as usize) {
        let p = p_in[nl];
        let x = x_in[nl];
        if p != xmsg && x != xmsg && p < psfc {
            nlx = nl + 1;
            nll += 1;
            if nll >= KLVL {
                return (xmsg, 9_999_999);
            }
            ws.pp[nll] = p;
            ws.xx[nll] = x;
            if aint(ws.pp[nll]) == aint(ptop) {
                pflag = true;
            }
        }
    }
    let mut nlmax = nll;

    if nlmax <= 3 {
        return (xmsg, -999);
    }

    if nlmax < (nlev as usize) {
        for nl in nlx..(nlev as usize) {
            nll += 1;
            if nll >= KLVL {
                return (xmsg, 9_999_998);
            }
            ws.pp[nll] = p_in[nl];
            ws.xx[nll] = 0.0;
        }
        nlmax = nll;
    }

    if ws.pp[nlmax] > ptop {
        nlmax += 1;
        if nlmax >= KLVL {
            return (xmsg, 9_999_997);
        }
        ws.pp[nlmax] = ptop;
        if linlog == 1 {
            let slope = (ws.xx[nlmax - 1] - xzero) / (ws.pp[nlmax - 1] - pzero);
            ws.xx[nlmax] = xzero + (ws.pp[nlmax] - pzero) * slope;
        } else if linlog == 2 {
            let slope = ws.xx[nlmax - 1] / ws.pp[nlmax - 1].ln();
            ws.xx[nlmax] = xzero + ws.pp[nlmax].ln() * slope;
        }
    } else if !pflag {
        ws.tmp1[..=nlmax].copy_from_slice(&ws.pp[..=nlmax]);
        ws.tmp2[..=nlmax].copy_from_slice(&ws.xx[..=nlmax]);

        let mut nltop: usize = 0;
        for nl in 1..nlmax {
            if aint(ws.pp[nl]) > ptop && aint(ws.pp[nl + 1]) < ptop {
                nltop = nl + 1;
                break;
            }
        }
        if nltop == 0 {
            return (xmsg, 1_000_000);
        }

        let nl1 = nltop;
        let nl2 = nltop - 1;
        if linlog == 1 {
            let slope = (ws.xx[nl2] - ws.xx[nl1]) / (ws.pp[nl2] - ws.pp[nl1]);
            ws.xx[nltop] = ws.xx[nl1] + (ptop - ws.pp[nl1]) * slope;
        } else if linlog == 2 {
            let pa = ws.pp[nl2].ln();
            let pc = ws.pp[nl1].ln();
            let slope = (ws.xx[nl2] - ws.xx[nl1]) / (pa - pc);
            ws.xx[nltop] = ws.xx[nl1] + (ptop.ln() - pc) * slope;
        }
        ws.pp[nltop] = ptop;

        for nl in (nltop..=nlmax).rev() {
            ws.xx[nl + 1] = ws.tmp2[nl];
            ws.pp[nl + 1] = ws.tmp1[nl];
        }
        nlmax += 1;
    }

    ws.pi.fill(0.0);
    ws.xi.fill(0.0);

    ws.pi[0] = pbot;
    for nl in 1..nlmax {
        let i_odd = 2 * nl - 1;
        let i_even = 2 * nl;
        if i_even >= KLVL {
            return (xmsg, 9_999_996);
        }
        ws.pi[i_odd] = ws.pp[nl];
        ws.pi[i_even] = 0.5 * (ws.pp[nl] + ws.pp[nl + 1]);
    }
    let i_last_odd = 2 * nlmax - 1;
    let i_last_even = 2 * nlmax;
    if i_last_even >= KLVL {
        return (xmsg, 9_999_995);
    }
    ws.pi[i_last_odd] = ws.pp[nlmax];
    ws.pi[i_last_even] = 0.5 * (ws.pp[nlmax] + zero);

    let mut nltop_idx: usize = 0;
    for nl in 1..=(2 * nlmax - 1) {
        if aint(ws.pi[nl]) == aint(ptop) {
            nltop_idx = nl;
        }
    }
    if nltop_idx == 0 || (nltop_idx % 2 == 0) {
        return (xmsg, 8_888_888);
    }

    let j1 = 2 * nltop_idx - 1;
    let j2 = 2 * nltop_idx;
    if j2 < ws.pi.len() {
        ws.pi[j1] = ptop;
        ws.pi[j2] = 0.5 * (ptop + zero);
    }

    let psfcx = if psfc != xmsg { psfc } else { ws.pp[1] };

    let pin = &ws.pp[1..=nlmax];
    let xin = &ws.xx[1..=nlmax];
    let pout = &ws.pi[1..=nltop_idx];

    let (p200, x200, pln200) = (&mut ws.p200, &mut ws.x200, &mut ws.pln200);
    let xout_slice = &mut ws.xout[..nltop_idx];

    let ier2 = dint2p2_ws(pin, xin, pout, xout_slice, linlog, xmsg, p200, x200, pln200);
    if ier2 != 0 {
        return (xmsg, ier2);
    }
    for i in 1..=nltop_idx {
        ws.xi[i] = xout_slice[i - 1];
    }

    ws.delp.fill(0.0);
    ws.beta.fill(0.0);

    for nl in 1..=nltop_idx {
        ws.delp[nl] = ws.pi[nl - 1] - ws.pi[nl + 1];
    }

    for nl in 1..=nltop_idx {
        if ws.pi[nl - 1] >= psfcx && ws.pi[nl + 1] < psfcx {
            ws.beta[nl] = (psfcx - ws.pi[nl + 1]) / (ws.pi[nl - 1] - ws.pi[nl + 1]);
        } else if ws.pi[nl - 1] > ptop && ws.pi[nl + 1] < ptop {
            ws.beta[nl] = (ws.pi[nl - 1] - ptop) / (ws.pi[nl - 1] - ws.pi[nl + 1]);
        } else if ws.pi[nl] < pbot && ws.pi[nl] >= ptop {
            ws.beta[nl] = 1.0;
        }
    }

    let mut vint = 0.0f64;
    let mut nl = 1usize;
    while nl <= nltop_idx {
        vint += ws.beta[nl] * ws.xi[nl] * ws.delp[nl];
        nl += 2;
    }

    (vint, 0)
}

/// Batch API (sum + normalization in ONE pass):
/// Returns:
///   vsum:  (...) float64   integral of x
///   vnorm: (...) float64   integral of 1 (same as calling vibeta on ones_like(x))
///   ier:   (...) int32     error code (0 ok; otherwise vsum/vnorm are NaN)
#[pyfunction]
pub fn dvibeta_batch_sum_norm(
    py: Python<'_>,
    p: PyReadonlyArray1<f64>,
    x: PyReadonlyArrayDyn<f64>,
    psfc: PyReadonlyArrayDyn<f64>,
    xmsg: f64,
    linlog: i32,
    ptop: f64,
) -> PyResult<(
    Py<PyArrayDyn<f64>>,
    Py<PyArrayDyn<f64>>,
    Py<PyArrayDyn<i32>>,
)> {
    let p = p.as_slice()?;
    let nlev = p.len();
    if nlev == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("p is empty"));
    }

    let x_slice = x
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("x must be C-contiguous"))?;
    let psfc_slice = psfc
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("psfc must be C-contiguous"))?;

    let x_shape = x.shape().to_vec();
    let ps_shape = psfc.shape().to_vec();

    if x_shape.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x must have at least 1 dimension (..., nlev)",
        ));
    }
    let x_nlev = x_shape[x_shape.len() - 1];
    if x_nlev != nlev {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x last dim ({x_nlev}) != p length ({nlev})"
        )));
    }

    if ps_shape.len() + 1 != x_shape.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "psfc shape must match x shape without last dim",
        ));
    }
    for i in 0..ps_shape.len() {
        if ps_shape[i] != x_shape[i] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "psfc shape must match x shape without last dim",
            ));
        }
    }

    let npoints = psfc_slice.len();
    if npoints * nlev != x_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x size is inconsistent with psfc size and p length",
        ));
    }

    let mut vsum = vec![f64::NAN; npoints];
    let mut vnorm = vec![f64::NAN; npoints];
    let mut ier_out = vec![0i32; npoints];

    vsum.par_iter_mut()
        .zip(vnorm.par_iter_mut())
        .zip(ier_out.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((vsum_i, vnorm_i), ier_i))| {
            let ps = psfc_slice[i];
            if !(ps > ptop) {
                *vsum_i = f64::NAN;
                *vnorm_i = f64::NAN;
                *ier_i = -10;
                return;
            }

            let base = i * nlev;
            let prof = &x_slice[base..base + nlev];

            let ((s, ier_s), (n, ier_n)) = WS.with(|ws_cell| {
                let mut ws = ws_cell.borrow_mut();

                // ---- prepare x profile (replace NaN/Inf with xmsg) ----
                for k in 0..nlev {
                    let v = prof[k];
                    ws.prof[k] = if v.is_finite() { v } else { xmsg };
                }

                let xsfc = {
                    let v0 = ws.prof[0];
                    if (v0 - xmsg).abs() > 1e10 { v0 } else { 0.0 }
                };

                // detach from ws to satisfy borrow checker
                let xbuf = ws.prof; // copy [f64; KLVL]
                let x_in = &xbuf[..nlev];

                let s_res = dvibeta_core_ws(
                    p,
                    x_in,
                    nlev as i32,
                    xmsg,
                    linlog,
                    ps,
                    xsfc,
                    ps, // pbot
                    ptop,
                    ptop, // plvcrt
                    &mut *ws,
                );

                // ---- normalization profile = constant 1 (same as ones_like) ----
                // xsfc for ones is 1
                let onesbuf = ws.ones; // copy [f64; KLVL]
                let ones_in = &onesbuf[..nlev];

                let n_res = dvibeta_core_ws(
                    p,
                    ones_in,
                    nlev as i32,
                    xmsg,
                    linlog,
                    ps,
                    1.0, // xsfc
                    ps,  // pbot
                    ptop,
                    ptop, // plvcrt
                    &mut *ws,
                );

                (s_res, n_res)
            });

            // if either fails -> NaN (match your python behavior: nan if ier != 0)
            if ier_s == 0 && ier_n == 0 {
                *vsum_i = s;
                *vnorm_i = n;
                *ier_i = 0;
            } else {
                *vsum_i = f64::NAN;
                *vnorm_i = f64::NAN;
                *ier_i = if ier_s != 0 { ier_s } else { ier_n };
            }
        });

    // outputs -> numpy
    let vsum_1d = PyArray1::<f64>::from_vec(py, vsum);
    let vsum_dyn = vsum_1d.reshape(ps_shape.clone())?.to_owned();

    let vnorm_1d = PyArray1::<f64>::from_vec(py, vnorm);
    let vnorm_dyn = vnorm_1d.reshape(ps_shape.clone())?.to_owned();

    let ier_1d = PyArray1::<i32>::from_vec(py, ier_out);
    let ier_dyn = ier_1d.reshape(ps_shape)?.to_owned();

    Ok((vsum_dyn.into(), vnorm_dyn.into(), ier_dyn.into()))
}
