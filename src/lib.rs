use pyo3::prelude::*;

#[pyfunction]
fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[pymodule]
#[pyo3(name="_easyclimate_rust")]
fn easyclimate_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;

    //wet_bulb module
    m.add_function(wrap_pyfunction!(wet_bulb::calc_wet_bulb_temperature, m)?)?;

    // sphere_laplacian module (NumPy interface)
    m.add_function(wrap_pyfunction!(sphere_laplacian::calc_sphere_laplacian_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_laplacian::calc_sphere_laplacian_conservative_numpy, m)?)?;

    // detrend_spatial module - high-performance spatial detrending
    m.add_function(wrap_pyfunction!(detrend_spatial::calc_detrend_spatial_3d, m)?)?;
    m.add_function(wrap_pyfunction!(detrend_spatial::calc_detrend_spatial_3d_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(detrend_spatial::calc_detrend_spatial_flexible, m)?)?;

    // interp1d
    m.add_function(wrap_pyfunction!(interp1d::interp1d_linear_core, m)?)?;
    m.add_function(wrap_pyfunction!(interp1d::interp1d_linear_2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(interp1d::interp1d_linear_3d_py, m)?)?;
    m.add_function(wrap_pyfunction!(interp1d::interp1d_linear_4d_py, m)?)?;

    Ok(())
}

// Declare module
mod wet_bulb;
mod sphere_laplacian;
mod detrend_spatial;
mod interp1d;
