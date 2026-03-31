//! # Error Types Module
//!
//! This module defines error types used throughout the traj-dist-rs library.
//!
//! ## Error Types
//!
//! - `TrajDistError`: Main error enum for all trajectory distance calculation errors
//!
//! ## Error Categories
//!
//! - **InvalidParams**: Invalid parameters passed to functions
//! - **InvalidCoordinate**: Coordinate array with wrong length
//! - **DataConvertionError**: Error during data type conversion
//! - **OutofIndex**: Index out of bounds access
//! - **InvalidSizeOfListArray**: Invalid ListArray size for arrow data
//! - **InvalidSeqType**: Invalid sequence type
//! - **InvalidConverter**: Invalid converter configuration

// Copyright 2024 All authors of TrajDL
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Main error type for trajectory distance calculations
#[derive(Debug, thiserror::Error)]
pub enum TrajDistError {
    #[error("ListArray<i64> must have length 1")]
    InvalidSizeOfListArray,

    #[error("InvalidParams")]
    InvalidParams(String),

    #[error("Coordinates must be an array of length 2, but received {0} elements")]
    InvalidCoordinate(usize),

    #[error("DataConvertionError: {0}")]
    DataConvertionError(String),

    #[error("Invalid SeqType")]
    InvalidSeqType,

    #[error("Invalid converter")]
    InvalidConverter,

    #[error("Out of bounds")]
    OutofIndex(String),
}
