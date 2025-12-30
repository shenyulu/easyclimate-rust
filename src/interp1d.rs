use numpy::{PyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyArrayMethods};
use pyo3::prelude::*;

/// Fast 1D linear interpolation for a single profile
/// 
/// # Arguments
/// * `x` - Input x coordinates (must be monotonically increasing)
/// * `y` - Input y values corresponding to x
/// * `x_new` - Target x coordinates for interpolation
/// 
/// # Returns
/// * Interpolated y values at x_new positions
#[pyfunction]
#[pyo3(name = "interp1d_linear_core")]
pub fn interp1d_linear_core(x: Vec<f64>, y: Vec<f64>, x_new: Vec<f64>) -> Vec<f64> {
    let n = x.len();
    let mut result = Vec::with_capacity(x_new.len());
    
    for &xi in &x_new {
        // 找到合适的插值区间
        let mut idx = 0;
        
        // 处理边界
        if (x[1] > x[0] && xi <= x[0]) || (x[1] < x[0] && xi >= x[0]) {
            result.push(y[0]);
            continue;
        }
        if (x[1] > x[0] && xi >= x[n-1]) || (x[1] < x[0] && xi <= x[n-1]) {
            result.push(y[n - 1]);
            continue;
        }
        
        // 查找区间（适用于升序和降序）
        for i in 0..n - 1 {
            if (x[i] <= xi && xi <= x[i + 1]) || (x[i] >= xi && xi >= x[i + 1]) {
                idx = i;
                break;
            }
        }
        
        // 线性插值
        let x0 = x[idx];
        let x1 = x[idx + 1];
        let y0 = y[idx];
        let y1 = y[idx + 1];
        
        let t = (xi - x0) / (x1 - x0);
        let yi = y0 + t * (y1 - y0);
        result.push(yi);
    }
    
    result
}

/// Fast 1D linear interpolation for 2D data (e.g., time x level)
/// Interpolates along the vertical dimension for each time step
/// 
/// # Arguments
/// * `z_data` - 2D array of z coordinates (time, level)
/// * `var_data` - 2D array of variable values (time, level)
/// * `target_heights` - 1D array of target z coordinates
/// 
/// # Returns
/// * 2D array of interpolated values (time, target_heights)
#[pyfunction]
#[pyo3(name = "interp1d_linear_2d")]
pub fn interp1d_linear_2d_py<'py>(
    py: Python<'py>,
    z_data: PyReadonlyArray2<'py, f64>,
    var_data: PyReadonlyArray2<'py, f64>,
    target_heights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let z = z_data.as_array();
    let var = var_data.as_array();
    let targets = target_heights.as_slice()?;
    
    let (ntime, _nlev) = z.dim();
    
    // 顺序处理 - 对于2D数据已经很快了
    let mut results = Vec::with_capacity(ntime);
    
    for i in 0..ntime {
        let z_profile = z.row(i).to_slice().unwrap().to_vec();
        let var_profile = var.row(i).to_slice().unwrap().to_vec();
        let interpolated = interp1d_linear_core(z_profile, var_profile, targets.to_vec());
        results.push(interpolated);
    }
    
    // Convert to 2D array using from_vec2
    let array = PyArray2::from_vec2(py, &results)?;
    Ok(array)
}

/// Fast 1D linear interpolation for 3D data (e.g., time x lat x level)
/// Interpolates along the vertical dimension for each time and lat
/// 
/// # Arguments
/// * `z_data` - 3D array of z coordinates (time, lat, level)
/// * `var_data` - 3D array of variable values (time, lat, level)
/// * `target_heights` - 1D array of target z coordinates
/// 
/// # Returns
/// * 3D array of interpolated values (time, lat, target_heights)
#[pyfunction]
#[pyo3(name = "interp1d_linear_3d")]
pub fn interp1d_linear_3d_py<'py>(
    py: Python<'py>,
    z_data: PyReadonlyArray3<'py, f64>,
    var_data: PyReadonlyArray3<'py, f64>,
    target_heights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let z = z_data.as_array();
    let var = var_data.as_array();
    let targets = target_heights.as_slice()?;
    
    let (ntime, nlat, nlev) = z.dim();
    let ntarget = targets.len();
    
    // 顺序处理
    let total_elements = ntime * nlat * ntarget;
    let mut data = Vec::with_capacity(total_elements);
    
    for i in 0..ntime {
        for j in 0..nlat {
            let mut z_profile = Vec::with_capacity(nlev);
            let mut var_profile = Vec::with_capacity(nlev);
            
            for k in 0..nlev {
                z_profile.push(z[[i, j, k]]);
                var_profile.push(var[[i, j, k]]);
            }
            
            let interpolated = interp1d_linear_core(z_profile, var_profile, targets.to_vec());
            data.extend(interpolated);
        }
    }
    
    // 创建一维数组然后reshape
    let flat_array = PyArray::from_vec(py, data);
    let reshaped = flat_array.reshape([ntime, nlat, ntarget])?;
    Ok(reshaped)
}

/// Fast 1D linear interpolation for 4D data (e.g., time x lat x lon x level)
/// Interpolates along the vertical dimension for each time, lat, and lon
/// 
/// # Arguments
/// * `z_data` - 4D array of z coordinates (time, lat, lon, level)
/// * `var_data` - 4D array of variable values (time, lat, lon, level)
/// * `target_heights` - 1D array of target z coordinates
/// 
/// # Returns
/// * 4D array of interpolated values (time, lat, lon, target_heights)
#[pyfunction]
#[pyo3(name = "interp1d_linear_4d")]
pub fn interp1d_linear_4d_py<'py>(
    py: Python<'py>,
    z_data: PyReadonlyArray4<'py, f64>,
    var_data: PyReadonlyArray4<'py, f64>,
    target_heights: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let z = z_data.as_array();
    let var = var_data.as_array();
    let targets = target_heights.as_slice()?;
    
    let (ntime, nlat, nlon, nlev) = z.dim();
    let ntarget = targets.len();
    
    // 顺序处理
    let total_elements = ntime * nlat * nlon * ntarget;
    let mut data = Vec::with_capacity(total_elements);
    
    for i in 0..ntime {
        for j in 0..nlat {
            for k in 0..nlon {
                let mut z_profile = Vec::with_capacity(nlev);
                let mut var_profile = Vec::with_capacity(nlev);
                
                for l in 0..nlev {
                    z_profile.push(z[[i, j, k, l]]);
                    var_profile.push(var[[i, j, k, l]]);
                }
                
                let interpolated = interp1d_linear_core(z_profile, var_profile, targets.to_vec());
                data.extend(interpolated);
            }
        }
    }
    
    // 创建一维数组然后reshape
    let flat_array = PyArray::from_vec(py, data);
    let reshaped = flat_array.reshape([ntime, nlat, nlon, ntarget])?;
    Ok(reshaped)
}