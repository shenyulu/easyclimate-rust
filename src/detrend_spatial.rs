use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Detrend data along the time axis for 3D array (time, lat, lon)
/// 
/// This function removes linear trends from spatial-temporal data with special value handling.
/// Uses parallel processing and optimized algorithms for maximum performance.
///
/// # Arguments
/// * `data` - Input 3D array with shape (time, lat, lon)
/// * `min_valid_fraction` - Minimum fraction of valid (finite) values required (default: 0.5)
///
/// # Returns
/// * Detrended 3D array with same shape as input
///
/// # Performance
/// - Uses Rayon for parallel processing across spatial grid points
/// - SIMD-friendly memory layout and operations
/// - Optimized linear regression implementation
#[pyfunction]
#[pyo3(signature = (data, min_valid_fraction=0.5))]
pub fn calc_detrend_spatial_3d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    min_valid_fraction: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let data_array = data.as_array();
    let shape = data_array.shape();
    let (nt, nlat, nlon) = (shape[0], shape[1], shape[2]);

    // Validate input
    if !(0.0..=1.0).contains(&min_valid_fraction) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_valid_fraction must be between 0 and 1",
        ));
    }

    // Pre-calculate time indices and statistics for linear regression
    let time_indices: Vec<f64> = (0..nt).map(|i| i as f64).collect();
    let t_mean = time_indices.iter().sum::<f64>() / nt as f64;
    let t_centered: Vec<f64> = time_indices.iter().map(|&t| t - t_mean).collect();
    let t_var: f64 = t_centered.iter().map(|&t| t * t).sum();

    // Create output array using zeros_bound
    let output = PyArray3::<f64>::zeros(py, [nt, nlat, nlon], false);
    let mut output_slice = unsafe { output.as_array_mut() };

    // Process each spatial location in parallel
    let spatial_points: Vec<(usize, usize)> = (0..nlat)
        .flat_map(|lat| (0..nlon).map(move |lon| (lat, lon)))
        .collect();

    let results: Vec<_> = spatial_points
        .par_iter()
        .map(|&(lat, lon)| {
            // Extract time series for this location
            let mut time_series = Vec::with_capacity(nt);
            for t in 0..nt {
                time_series.push(data_array[[t, lat, lon]]);
            }

            // Detrend the time series
            detrend_timeseries(&time_series, &t_centered, t_var, min_valid_fraction)
        })
        .collect();

    // Write results back to output array
    for (idx, &(lat, lon)) in spatial_points.iter().enumerate() {
        let detrended = &results[idx];
        for (t, &value) in detrended.iter().enumerate() {
            output_slice[[t, lat, lon]] = value;
        }
    }

    Ok(output)
}

/// Optimized detrend for a single time series
/// 
/// Uses efficient linear regression with special value handling
#[inline]
fn detrend_timeseries(
    data: &[f64],
    t_centered: &[f64],
    t_var: f64,
    min_valid_fraction: f64,
) -> Vec<f64> {
    let n = data.len();
    let min_valid_count = (n as f64 * min_valid_fraction).ceil() as usize;

    // Count valid (finite) values and calculate statistics
    let mut valid_count = 0;
    let mut sum_y = 0.0;
    let mut sum_ty = 0.0;

    for i in 0..n {
        let y = data[i];
        if y.is_finite() {
            valid_count += 1;
            sum_y += y;
            sum_ty += t_centered[i] * y;
        }
    }

    // If not enough valid data, return NaN array
    if valid_count < min_valid_count {
        return vec![f64::NAN; n];
    }

    // Calculate linear regression parameters
    let y_mean = sum_y / valid_count as f64;
    let slope = sum_ty / t_var;

    // Detrend: remove linear trend
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let y = data[i];
        if y.is_finite() {
            let trend = slope * t_centered[i] + y_mean;
            result.push(y - trend);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Fast detrend for 3D array with chunked processing
/// 
/// This version processes data in chunks for better cache locality
/// Recommended for very large datasets
#[pyfunction]
#[pyo3(signature = (data, min_valid_fraction=0.5, chunk_size=1000))]
pub fn calc_detrend_spatial_3d_chunked<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    min_valid_fraction: f64,
    chunk_size: usize,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let data_array = data.as_array();
    let shape = data_array.shape();
    let (nt, nlat, nlon) = (shape[0], shape[1], shape[2]);

    if !(0.0..=1.0).contains(&min_valid_fraction) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_valid_fraction must be between 0 and 1",
        ));
    }

    // Pre-calculate time statistics
    let time_indices: Vec<f64> = (0..nt).map(|i| i as f64).collect();
    let t_mean = time_indices.iter().sum::<f64>() / nt as f64;
    let t_centered: Vec<f64> = time_indices.iter().map(|&t| t - t_mean).collect();
    let t_var: f64 = t_centered.iter().map(|&t| t * t).sum();

    let output = PyArray3::<f64>::zeros(py, [nt, nlat, nlon], false);
    let mut output_slice = unsafe { output.as_array_mut() };

    // Process in chunks for better cache performance
    let total_points = nlat * nlon;
    let num_chunks = (total_points + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = ((chunk_idx + 1) * chunk_size).min(total_points);

        let chunk_points: Vec<(usize, usize)> = (start_idx..end_idx)
            .map(|idx| (idx / nlon, idx % nlon))
            .collect();

        let results: Vec<_> = chunk_points
            .par_iter()
            .map(|&(lat, lon)| {
                let mut time_series = Vec::with_capacity(nt);
                for t in 0..nt {
                    time_series.push(data_array[[t, lat, lon]]);
                }
                detrend_timeseries(&time_series, &t_centered, t_var, min_valid_fraction)
            })
            .collect();

        // Write results
        for (idx, &(lat, lon)) in chunk_points.iter().enumerate() {
            let detrended = &results[idx];
            for (t, &value) in detrended.iter().enumerate() {
                output_slice[[t, lat, lon]] = value;
            }
        }
    }

    Ok(output)
}

/// Detrend with custom time dimension axis
/// 
/// Supports different axis orderings: (time, lat, lon), (lat, lon, time), (lon, lat, time)
#[pyfunction]
#[pyo3(signature = (data, time_axis=0, min_valid_fraction=0.5))]
pub fn calc_detrend_spatial_flexible<'py>(
    py: Python<'py>,
    data: PyReadonlyArray3<'py, f64>,
    time_axis: usize,
    min_valid_fraction: f64,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let data_array = data.as_array();
    let shape = data_array.shape();

    if time_axis >= 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time_axis must be 0, 1, or 2",
        ));
    }

    if !(0.0..=1.0).contains(&min_valid_fraction) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_valid_fraction must be between 0 and 1",
        ));
    }

    let nt = shape[time_axis];
    let other_dims: Vec<usize> = (0..3).filter(|&i| i != time_axis).collect();
    let dim1 = shape[other_dims[0]];
    let dim2 = shape[other_dims[1]];

    // Pre-calculate time statistics
    let time_indices: Vec<f64> = (0..nt).map(|i| i as f64).collect();
    let t_mean = time_indices.iter().sum::<f64>() / nt as f64;
    let t_centered: Vec<f64> = time_indices.iter().map(|&t| t - t_mean).collect();
    let t_var: f64 = t_centered.iter().map(|&t| t * t).sum();

    let output = PyArray3::<f64>::zeros(py, [shape[0], shape[1], shape[2]], false);
    let mut output_slice = unsafe { output.as_array_mut() };

    let spatial_points: Vec<(usize, usize)> = (0..dim1)
        .flat_map(|i| (0..dim2).map(move |j| (i, j)))
        .collect();

    let results: Vec<_> = spatial_points
        .par_iter()
        .map(|&(i, j)| {
            let mut time_series = Vec::with_capacity(nt);
            
            // Extract time series based on axis configuration
            for t in 0..nt {
                let value = match time_axis {
                    0 => data_array[[t, i, j]],
                    1 => data_array[[i, t, j]],
                    2 => data_array[[i, j, t]],
                    _ => unreachable!(),
                };
                time_series.push(value);
            }

            detrend_timeseries(&time_series, &t_centered, t_var, min_valid_fraction)
        })
        .collect();

    // Write results back
    for (idx, &(i, j)) in spatial_points.iter().enumerate() {
        let detrended = &results[idx];
        for (t, &value) in detrended.iter().enumerate() {
            match time_axis {
                0 => output_slice[[t, i, j]] = value,
                1 => output_slice[[i, t, j]] = value,
                2 => output_slice[[i, j, t]] = value,
                _ => unreachable!(),
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detrend_simple() {
        // Create simple linear trend
        let t_centered = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect linear trend
        
        let result = detrend_timeseries(&data, &t_centered, 10.0, 0.5);
        
        // After detrending, should be approximately zero (with floating point tolerance)
        for val in result.iter() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_detrend_with_nan() {
        let t_centered = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        
        let result = detrend_timeseries(&data, &t_centered, 10.0, 0.5);
        
        // Check that NaN positions are preserved
        assert!(result[1].is_nan());
        assert!(result[3].is_nan());
        assert!(result[0].is_finite());
    }

    #[test]
    fn test_insufficient_valid_data() {
        let t_centered = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let data = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 5.0];
        
        let result = detrend_timeseries(&data, &t_centered, 10.0, 0.5);
        
        // Should return all NaN when insufficient valid data
        assert!(result.iter().all(|x| x.is_nan()));
    }
}