use core::ffi::c_uint;

use zstd_sys::ZDICT_params_t;

use crate::{ptr_void, CompressionLevel, WriteBuf};

/// Represents a possible error from dictionary part of the zstd library.
pub type ErrorCode = usize;

/// Wrapper result around most zstd dictionary functions.
///
/// Either a success code (usually number of bytes written), or an error code.
pub type SafeResult = Result<usize, ErrorCode>;

/// Returns true if code represents error.
fn is_error(code: usize) -> bool {
    // Safety: Just FFI
    unsafe { zstd_sys::ZDICT_isError(code) != 0 }
}

/// Parse the result code
///
/// Returns the number of bytes written if the code represents success,
/// or the error message code otherwise.
fn parse_code(code: usize) -> SafeResult {
    if !is_error(code) {
        Ok(code)
    } else {
        Err(code)
    }
}

pub struct Parameters {
    pub compression_level: CompressionLevel,
    pub notification_level: c_uint,
    pub dict_id: c_uint,
}

impl Parameters {
    fn as_zdict_struct(&self) -> ZDICT_params_t {
        ZDICT_params_t {
            compressionLevel: self.compression_level,
            notificationLevel: self.notification_level,
            dictID: self.dict_id,
        }
    }
}

impl Default for Parameters {
    fn default() -> Parameters {
        Parameters {
            compression_level: 0,
            notification_level: 0,
            dict_id: 0,
        }
    }
}

/// Wraps the `ZDICT_trainFromBuffer()` function.
pub fn train_from_buffer<C: WriteBuf + ?Sized>(
    dict_buffer: &mut C,
    samples_buffer: &[u8],
    samples_sizes: &[usize],
) -> SafeResult {
    assert_eq!(samples_buffer.len(), samples_sizes.iter().sum());

    unsafe {
        dict_buffer.write_from(|buffer, capacity| {
            parse_code(zstd_sys::ZDICT_trainFromBuffer(
                buffer,
                capacity,
                ptr_void(samples_buffer),
                samples_sizes.as_ptr(),
                samples_sizes.len() as u32,
            ))
        })
    }
}

/// Wraps the `ZDICT_finalizeDictionary()` function.
pub fn finalize_dictionary<C: WriteBuf + ?Sized>(
    dict_buffer: &mut C,
    dict_content: &[u8],
    samples_buffer: &[u8],
    samples_sizes: &[usize],
    parameters: &Parameters,
) -> SafeResult {
    assert_eq!(samples_buffer.len(), samples_sizes.iter().sum());

    unsafe {
        dict_buffer.write_from(|buffer, capacity| {
            parse_code(zstd_sys::ZDICT_finalizeDictionary(
                buffer,
                capacity,
                ptr_void(dict_content),
                dict_content.len(),
                ptr_void(samples_buffer),
                samples_sizes.as_ptr(),
                samples_sizes.len() as u32,
                parameters.as_zdict_struct(),
            ))
        })
    }
}
