use rayon::prelude::*;
use std::f64::consts::PI;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray2};

/// Spherical Laplacian calculation
pub struct SphereLaplacian {
    earth_radius: f64,
}

impl SphereLaplacian {
    pub fn new(earth_radius: f64) -> Self {
        Self { earth_radius }
    }

    // pub fn default() -> Self {
    //     Self::new(6.371e6)
    // }

    /// Core computing functions - Using flat array (compatible with NumPy)
    /// 
    /// # Arguments
    /// * `t_flat` - Flat-lying temperature field [nlat * nlon]
    /// * `lat` - Latitude array (degrees) [nlat]
    /// * `lon` - Longitude array (degrees) [nlon]
    /// * `cyclic_boundary` - Whether to use the loop boundary (to handle the connection at lon = 0 and lon = 360)
    pub fn compute_flat(
        &self,
        t_flat: &[f64],
        lat: &[f64],
        lon: &[f64],
        cyclic_boundary: bool,
    ) -> Result<Vec<f64>, String> {
        let nlat = lat.len();
        let nlon = lon.len();

        if t_flat.len() != nlat * nlon {
            return Err(format!(
                // "数据长度 {} 不等于 nlat({}) * nlon({})",
                "The data length {} is not equal to nlat({}) multiplied by nlon({}).",
                t_flat.len(),
                nlat,
                nlon
            ));
        }
        if nlat < 3 {
            // return Err("纬度点数必须至少为3".to_string());
            return Err("The number of latitude points must be at least 3.".to_string());
        }
        if nlon < 3 {
            // return Err("经度点数必须至少为3".to_string());
            return Err("The longitude value must be at least 3.".to_string());
        }

        // 检查经度间距一致性
        // Check the consistency of longitude intervals
        let dlon = lon[1] - lon[0];
        for i in 2..lon.len() {
            if (lon[i] - lon[i - 1] - dlon).abs() > 1e-6 {
                // return Err("经度间距必须均匀".to_string());
                return Err("The distance between longitudes must be uniform.".to_string());
            }
        }

        // 如果使用循环边界,检查是否真的是全球数据
        // If using the loop boundary, check whether it is indeed global data
        if cyclic_boundary {
            let lon_range = lon[lon.len() - 1] - lon[0] + dlon;
            if (lon_range - 360.0).abs() > 1e-6 {
                return Err(format!(
                    // "循环边界要求经度覆盖 360°, 当前覆盖 {:.2}°",
                    "The loop boundary requires the longitude to cover 360°. Currently, it covers {:.2}°.",
                    lon_range
                ));
            }
        }

        // 计算网格间距(转换为弧度)
        // Calculate the grid spacing (converted to radians)
        let dlambda = dlon * PI / 180.0;
        let dphi = (lat[1] - lat[0]) * PI / 180.0;

        // 预计算三角函数值
        // Pre-computed trigonometric function values
        let phi: Vec<f64> = lat.iter().map(|&x| x * PI / 180.0).collect();
        let cos_phi: Vec<f64> = phi.iter().map(|&x| x.cos()).collect();
        let tan_phi: Vec<f64> = phi.iter().map(|&x| x.tan()).collect();

        let r_sq = self.earth_radius * self.earth_radius;
        let dlambda_sq = dlambda * dlambda;
        let dphi_sq = dphi * dphi;
        let two_dphi = 2.0 * dphi;

        // 并行计算拉普拉斯算子
        // Parallel computation of the Laplacian operator
        let mut laplacian = vec![0.0; nlat * nlon];

        laplacian
            .par_chunks_mut(nlon)
            .enumerate()
            .for_each(|(j, row)| {
                if j == 0 || j == nlat - 1 {
                    // 极点处设为 NaN
                    // Set the extreme point to NaN
                    row.iter_mut().for_each(|x| *x = f64::NAN);
                } else {
                    let cos_phi_sq = cos_phi[j] * cos_phi[j];
                    let tan_phi_j = tan_phi[j];

                    for i in 0..nlon {
                        // 处理经度边界
                        // Handling the longitude boundaries
                        let (i_plus, i_minus) = if cyclic_boundary {
                            // 循环边界: 0 连接到 nlon-1
                            // Loop boundary: 0 is connected to nlon-1
                            ((i + 1) % nlon, if i == 0 { nlon - 1 } else { i - 1 })
                        } else {
                            // 非循环边界: 边界点设为 NaN
                            // Non-cyclic boundary: Set boundary points to NaN
                            if i == 0 || i == nlon - 1 {
                                row[i] = f64::NAN;
                                continue;
                            }
                            (i + 1, i - 1)
                        };

                        // 获取相邻点的值
                        // Obtain the values of adjacent points
                        let t_center = t_flat[j * nlon + i];
                        let t_east = t_flat[j * nlon + i_plus];
                        let t_west = t_flat[j * nlon + i_minus];
                        let t_north = t_flat[(j + 1) * nlon + i];
                        let t_south = t_flat[(j - 1) * nlon + i];

                        // 经度方向二阶导数
                        // Second derivative in the longitude direction
                        let d2t_dlambda2 = (t_east - 2.0 * t_center + t_west) / dlambda_sq;

                        // 纬度方向二阶导数
                        // Second-order derivative in the latitude direction
                        let d2t_dphi2 = (t_north - 2.0 * t_center + t_south) / dphi_sq;

                        // 纬度方向一阶导数
                        // First-order derivative in the latitude direction
                        let dt_dphi = (t_north - t_south) / two_dphi;

                        // 拉普拉斯算子
                        // Laplace operator
                        row[i] = (d2t_dlambda2 / cos_phi_sq + d2t_dphi2 
                            - tan_phi_j * dt_dphi) / r_sq;
                    }
                }
            });

        Ok(laplacian)
    }

    /// 守恒形式 - flat array 版本
    /// Conservation form - flat array version
    pub fn compute_conservative_flat(
        &self,
        t_flat: &[f64],
        lat: &[f64],
        lon: &[f64],
        cyclic_boundary: bool,
    ) -> Result<Vec<f64>, String> {
        let nlat = lat.len();
        let nlon = lon.len();

        if t_flat.len() != nlat * nlon || nlat < 3 || nlon < 3 {
            // return Err("维度错误".to_string());
            return Err("Dimension error".to_string());
        }

        let dlambda = (lon[1] - lon[0]) * PI / 180.0;
        let dphi = (lat[1] - lat[0]) * PI / 180.0;

        let phi: Vec<f64> = lat.iter().map(|&x| x * PI / 180.0).collect();
        let cos_phi: Vec<f64> = phi.iter().map(|&x| x.cos()).collect();

        let r_sq = self.earth_radius * self.earth_radius;
        let dlambda_sq = dlambda * dlambda;
        let dphi_sq = dphi * dphi;

        let mut laplacian = vec![0.0; nlat * nlon];

        laplacian
            .par_chunks_mut(nlon)
            .enumerate()
            .for_each(|(j, row)| {
                if j == 0 || j == nlat - 1 {
                    row.iter_mut().for_each(|x| *x = f64::NAN);
                } else {
                    let cos_phi_half_plus = ((phi[j] + phi[j + 1]) / 2.0).cos();
                    let cos_phi_half_minus = ((phi[j] + phi[j - 1]) / 2.0).cos();
                    let cos_phi_j = cos_phi[j];

                    for i in 0..nlon {
                        let (i_plus, i_minus) = if cyclic_boundary {
                            ((i + 1) % nlon, if i == 0 { nlon - 1 } else { i - 1 })
                        } else {
                            if i == 0 || i == nlon - 1 {
                                row[i] = f64::NAN;
                                continue;
                            }
                            (i + 1, i - 1)
                        };

                        let t_center = t_flat[j * nlon + i];
                        let t_east = t_flat[j * nlon + i_plus];
                        let t_west = t_flat[j * nlon + i_minus];
                        let t_north = t_flat[(j + 1) * nlon + i];
                        let t_south = t_flat[(j - 1) * nlon + i];

                        // 经度方向贡献
                        // Longitude direction contribution
                        let lambda_contrib = cos_phi_j * (t_east - t_center) 
                            - cos_phi_j * (t_center - t_west);
                        let lambda_term = lambda_contrib 
                            / (cos_phi_j * dlambda_sq * r_sq);

                        // 纬度方向贡献
                        // Latitude direction contribution
                        let phi_contrib = cos_phi_half_plus * (t_north - t_center) 
                            - cos_phi_half_minus * (t_center - t_south);
                        let phi_term = phi_contrib / (cos_phi_j * dphi_sq * r_sq);

                        row[i] = lambda_term + phi_term;
                    }
                }
            });

        Ok(laplacian)
    }
}

// NumPy 数组接口 - 标准形式
#[pyfunction]
pub fn calc_sphere_laplacian_numpy<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    lon: PyReadonlyArray1<'py, f64>,
    cyclic_boundary: bool,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_array = t.as_array();
    let lat_slice = lat.as_slice()?;
    let lon_slice = lon.as_slice()?;

    let (_nlat, nlon) = (t_array.nrows(), t_array.ncols());

    // 转换为 flat array
    let t_flat: Vec<f64> = t_array.iter().copied().collect();

    let calculator = SphereLaplacian::new(earth_radius);

    let result_flat = calculator
        .compute_flat(&t_flat, lat_slice, lon_slice, cyclic_boundary)
        .map_err(|e| PyValueError::new_err(e))?;

    // 转换回 2D NumPy 数组
    let result_vec2d: Vec<Vec<f64>> = result_flat
        .chunks(nlon)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    let result_array = PyArray2::from_vec2(py, &result_vec2d)
        .map_err(|e| PyValueError::new_err(format!("创建数组失败: {}", e)))?;

    Ok(result_array)
}

// NumPy 数组接口 - 守恒形式
#[pyfunction]
pub fn calc_sphere_laplacian_conservative_numpy<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<'py, f64>,
    lat: PyReadonlyArray1<'py, f64>,
    lon: PyReadonlyArray1<'py, f64>,
    cyclic_boundary: bool,
    earth_radius: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_array = t.as_array();
    let lat_slice = lat.as_slice()?;
    let lon_slice = lon.as_slice()?;

    let (_nlat, nlon) = (t_array.nrows(), t_array.ncols());
    let t_flat: Vec<f64> = t_array.iter().copied().collect();

    let calculator = SphereLaplacian::new(earth_radius);

    let result_flat = calculator
        .compute_conservative_flat(&t_flat, lat_slice, lon_slice, cyclic_boundary)
        .map_err(|e| PyValueError::new_err(e))?;

    let result_vec2d: Vec<Vec<f64>> = result_flat
        .chunks(nlon)
        .map(|chunk| chunk.to_vec())
        .collect();

    let result_array = PyArray2::from_vec2(py, &result_vec2d)
        .map_err(|e| PyValueError::new_err(format!("创建数组失败: {}", e)))?;

    Ok(result_array)
}