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

#[derive(Debug, thiserror::Error)]
pub enum TrajDistError {
    #[error("ListArray<i64>的长度必须为1")]
    InvalidSizeOfListArray,

    #[error("InvalidParams")]
    InvalidParams(String),

    #[error("坐标必须是长度为2的数组，但收到了 {0} 个元素")]
    InvalidCoordinate(usize),

    #[error("DataConvertionError: {0}")]
    DataConvertionError(String),

    #[error("SeqType错误")]
    InvalidSeqType,

    #[error("converter异常")]
    InvalidConverter,

    #[error("越界")]
    OutofIndex(String),
}
