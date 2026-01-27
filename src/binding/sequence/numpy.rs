use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use std::marker::PhantomData;

use crate::binding::sequence::types::PointRef;
use crate::err::TrajDistError;
use crate::traits::CoordSequence;

/// 持有对 NumPy 数组的底层数据指针，实现零拷贝访问整个轨迹
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryRef<'a> {
    data_ptr: *const f64,
    len: usize,
    _phantom: PhantomData<&'a f64>,
}

impl<'a> TrajectoryRef<'a> {
    pub fn new(array: PyReadonlyArray2<'a, f64>) -> Result<Self, TrajDistError> {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(TrajDistError::DataConvertionError(format!(
                "Numpy array must have a shape of (N, 2), but got {:?}",
                shape
            )));
        }
        // 检查数组是否是连续的
        let view = array.as_array();
        if view.is_standard_layout() {
            Ok(Self {
                data_ptr: view.as_ptr(),
                len: shape[0],
                _phantom: PhantomData,
            })
        } else {
            Err(TrajDistError::DataConvertionError(
                "Numpy array must be contiguous (C-order)".to_string(),
            ))
        }
    }

    /// 获取底层数据切片（使用指针创建）
    #[inline(always)]
    fn get_data(&self) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len * 2) }
    }
}

impl<'a> CoordSequence for TrajectoryRef<'a> {
    type Coord = PointRef<'a>;

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Self::Coord {
        PointRef::new(self.get_data(), idx)
    }
}
