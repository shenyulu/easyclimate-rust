use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

// Calculate saturated vapor pressure
fn e_s(t_c: f64) -> f64 {
    let a = 17.2693882;
    let b = 35.86;
    let t_k = t_c + 273.16;
    6.1078 * (a * (t_k - 273.16) / (t_k - b)).exp()
}

// Calculate the actual vapor pressure
fn e_a(t_dry: f64, rh: f64) -> f64 {
    (rh / 100.0) * e_s(t_dry)
}

// Objective function f(T_w)
fn f(t_w: f64, t_dry: f64, rh: f64, p: f64, a: f64) -> f64 {
    let e_w = e_s(t_w);
    let e_d = e_a(t_dry, rh);
    e_w - e_d - (a * p * (t_dry - t_w))
}

// Derivative of the objective function with respect to T (df/dT)
fn df_dt(t_w: f64, t_dry: f64, rh: f64, p: f64, a: f64) -> f64 {
    let dt = 0.001;
    (f(t_w + dt, t_dry, rh, p, a) - f(t_w - dt, t_dry, rh, p, a)) / (2.0 * dt)
}

// Main function for calculating wet bulb temperature
pub fn wet_bulb_temperature(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
) -> Result<f64, &'static str> {
    let mut t_guess = t_dry;
    let mut delta_t = tolerance + 1.0;
    let mut iter = 0;

    while delta_t > tolerance && iter < max_iter {
        let f_val = f(t_guess, t_dry, rh, p, a);
        let df_val = df_dt(t_guess, t_dry, rh, p, a);
        
        // Avoid zero division errors
        if df_val.abs() < f64::EPSILON {
            return Err("If the derivative value is too small, it may lead to unstable numerical results.");
        }
        
        let t_new = t_guess - f_val / df_val;
        delta_t = (t_new - t_guess).abs();
        t_guess = t_new;
        iter += 1;
    }

    if iter >= max_iter {
        return Err("Reaching the maximum number of iterations but not converging");
    }

    Ok(t_guess)
}

// Python wrapper for the wet-bulb temperature function
#[pyfunction]
pub fn calc_wet_bulb_temperature(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
) -> PyResult<f64> {
    match wet_bulb_temperature(t_dry, rh, p, tolerance, a, max_iter) {
        Ok(result) => Ok(result),
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wet_bulb_temperature() {
        let t_dry = 25.0;
        let rh = 60.0;
        let p = 1013.25;
        let tolerance = 1e-6;
        let a = 6.5e-4;
        let max_iter = 100;

        match wet_bulb_temperature(t_dry, rh, p, tolerance, a, max_iter) {
            Ok(t_wet) => {
                println!("wet bulb temperature: {:.2}°C", t_wet);
                assert!(t_wet > 15.0 && t_wet < 25.0);
            }
            Err(e) => panic!("Calculation failed: {}", e),
        }
    }

    #[test]
    fn test_e_s() {
        let result = e_s(20.0);
        assert!((result - 23.37).abs() < 0.1);
    }
}