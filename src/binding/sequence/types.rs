use numpy::PyReadonlyArray2;
use pyo3::{
    Bound, PyAny,
    types::{PyAnyMethods, PyList},
};

use crate::traits::CoordSequence;
use crate::{err::TrajDistError, traits::AsCoord};

use super::numpy::TrajectoryRef;

/// 持有对底层 Rust 切片的引用和点的索引，实现真正的零拷贝访问
#[derive(Debug, Clone, Copy)]
pub struct PointRef<'a> {
    data: &'a [f64],
    idx: usize,
}

impl<'a> PointRef<'a> {
    pub fn new(data: &'a [f64], idx: usize) -> Self {
        Self { data, idx }
    }
}

impl<'a> AsCoord for PointRef<'a> {
    #[inline(always)]
    fn x(&self) -> f64 {
        self.data[self.idx * 2]
    }

    #[inline(always)]
    fn y(&self) -> f64 {
        self.data[self.idx * 2 + 1]
    }
}

#[derive(Debug)]
pub enum PyTrajectoryType<'a> {
    /// 使用 TrajectoryRef 实现零拷贝访问 NumPy 数组
    Numpy(TrajectoryRef<'a>),

    /// 从Python List拷贝过来的数据
    Owned(Vec<[f64; 2]>),
}

impl<'a> TryFrom<&Bound<'a, PyAny>> for PyTrajectoryType<'a> {
    type Error = TrajDistError;

    fn try_from(seq: &Bound<'a, PyAny>) -> Result<Self, Self::Error> {
        // 优先尝试转换为 numpy 数组
        if let Ok(readonly_array) = seq.extract::<PyReadonlyArray2<'a, f64>>() {
            // 使用 TrajectoryRef 封装，实现零拷贝访问
            let traj_ref = TrajectoryRef::new(readonly_array)?;
            Ok(Self::Numpy(traj_ref))
        }
        // 其次尝试转换为 Python List
        else if let Ok(loc_list) = seq.downcast::<PyList>() {
            let owned_vec = loc_list.extract::<Vec<[f64; 2]>>()
                .map_err(|_| TrajDistError::DataConvertionError(
                    "Failed to extract List[List[float]] into Vec<[f64; 2]>. Check list structure and element types.".to_string()
                ))?;
            Ok(Self::Owned(owned_vec))
        } else {
            Err(TrajDistError::DataConvertionError(format!(
                "Input must be a 2D numpy.ndarray of float64 or a List[List[float]], not a {:?}",
                seq.get_type()
            )))
        }
    }
}

impl<'a> CoordSequence for PyTrajectoryType<'a> {
    type Coord = PointRef<'a>;

    fn len(&self) -> usize {
        match self {
            PyTrajectoryType::Numpy(traj_ref) => traj_ref.len(),
            PyTrajectoryType::Owned(vec) => vec.len(),
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Self::Coord {
        match self {
            PyTrajectoryType::Numpy(traj_ref) => traj_ref.get(idx),
            PyTrajectoryType::Owned(vec) => {
                // 直接从 Vec 中获取切片引用
                let data = unsafe {
                    std::slice::from_raw_parts(vec.as_ptr() as *const f64, vec.len() * 2)
                };
                PointRef::new(data, idx)
            }
        }
    }
}
