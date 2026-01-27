use pyo3::{PyErr, exceptions::PyValueError};

use crate::err::TrajDistError;

impl From<TrajDistError> for PyErr {
    fn from(err: TrajDistError) -> PyErr {
        match err {
            TrajDistError::InvalidCoordinate(n) => {
                PyValueError::new_err(format!("坐标必须包含两个值（经度、纬度），但收到 {} 个", n))
            }
            TrajDistError::InvalidParams(s) => {
                PyValueError::new_err(format!("InvalidParams: {}", s))
            }
            TrajDistError::DataConvertionError(s) => {
                PyValueError::new_err(format!("DataConvertionError: {}", s))
            }
            TrajDistError::InvalidSizeOfListArray => {
                PyValueError::new_err("ListArray<i64>的长度必须为1")
            }
            TrajDistError::InvalidSeqType => PyValueError::new_err("SeqType错误"),
            TrajDistError::InvalidConverter => PyValueError::new_err("converter异常"),
            TrajDistError::OutofIndex(e) => PyValueError::new_err(format!("out of index: {}", e)),
        }
    }
}
