use pyo3::{PyErr, exceptions::PyValueError};

use crate::err::TrajDistError;

impl From<TrajDistError> for PyErr {
    fn from(err: TrajDistError) -> PyErr {
        match err {
            TrajDistError::InvalidCoordinate(n) => PyValueError::new_err(format!(
                "Coordinates must contain two values (longitude, latitude), but received {}",
                n
            )),
            TrajDistError::InvalidParams(s) => {
                PyValueError::new_err(format!("InvalidParams: {}", s))
            }
            TrajDistError::DataConvertionError(s) => {
                PyValueError::new_err(format!("DataConvertionError: {}", s))
            }
            TrajDistError::InvalidSizeOfListArray => {
                PyValueError::new_err("ListArray<i64> must have length 1")
            }
            TrajDistError::InvalidSeqType => PyValueError::new_err("Invalid SeqType"),
            TrajDistError::InvalidConverter => PyValueError::new_err("Invalid converter"),
            TrajDistError::OutofIndex(e) => PyValueError::new_err(format!("out of index: {}", e)),
        }
    }
}
