//! Cosine search without a resident FP32 copy of an INT8 corpus.
//! Row scales cancel in cosine similarity when each scale is positive.
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Avx2,
}

impl Kernel {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Self::Avx2;
        }
        Self::Scalar
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
        }
    }
}

pub enum Values {
    Float(Vec<f32>),
    /// Signed two's-complement codes, kept as bytes to read directly from disk.
    Int8(Vec<u8>),
}

pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Values,
}

impl Matrix {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 16];
        reader.read_exact(&mut header).map_err(|e| e.to_string())?;
        if &header[..4] != b"VEC1" {
            return Err("invalid vector file magic".into());
        }
        let word =
            |offset| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap()) as usize;
        let (kind, rows, columns) = (word(4), word(8), word(12));
        if rows == 0 || columns == 0 || kind > 1 {
            return Err("invalid vector shape or dtype".into());
        }
        let count = rows.checked_mul(columns).ok_or("vector shape overflow")?;
        let width = if kind == 0 { 4 } else { 1 };
        let payload = count.checked_mul(width).ok_or("payload size overflow")?;
        if (payload as u64)
            .checked_add(16)
            .ok_or("file length overflow")?
            != length
        {
            return Err("vector file length mismatch".into());
        }
        let values = if kind == 0 {
            let mut values = vec![0f32; count];
            // Every bit pattern is valid for f32. Read into its allocation to avoid
            // a second full corpus allocation, then reject non-finite values.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), payload)
            };
            reader.read_exact(bytes).map_err(|e| e.to_string())?;
            if cfg!(target_endian = "big") {
                for value in &mut values {
                    *value = f32::from_bits(value.to_bits().swap_bytes());
                }
            }
            if !values.iter().all(|v| v.is_finite()) {
                return Err("non-finite vector".into());
            }
            Values::Float(values)
        } else {
            let mut values = vec![0u8; count];
            reader.read_exact(&mut values).map_err(|e| e.to_string())?;
            Values::Int8(values)
        };
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    pub fn payload_bytes(&self) -> usize {
        match &self.values {
            Values::Float(v) => v.len() * 4,
            Values::Int8(v) => v.len(),
        }
    }

    pub fn float_row(&self, row: usize) -> Result<&[f32], String> {
        if row >= self.rows {
            return Err("query row out of bounds".into());
        }
        match &self.values {
            Values::Float(values) => Ok(&values[row * self.columns..(row + 1) * self.columns]),
            _ => Err("queries must be FP32".into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub id: usize,
    pub score: f32,
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.total_cmp(&other.score).is_eq()
    }
}
impl Eq for Match {}
impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Match {
    // The heap's maximum is the worst retained match: low score, then high ID.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

pub struct Index {
    matrix: Matrix,
    inverse_norms: Vec<f32>,
}

impl Index {
    pub fn matrix(&self) -> &Matrix {
        &self.matrix
    }
    pub fn new(matrix: Matrix) -> Result<Self, String> {
        if matrix.rows == 0 || matrix.columns == 0 {
            return Err("empty matrix".into());
        }
        let count = matrix
            .rows
            .checked_mul(matrix.columns)
            .ok_or("matrix shape overflow")?;
        let length = match &matrix.values {
            Values::Float(v) => v.len(),
            Values::Int8(v) => v.len(),
        };
        if length != count {
            return Err("matrix shape mismatch".into());
        }
        let mut inverse_norms = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            let start = row * matrix.columns;
            let end = start + matrix.columns;
            let squared: f64 = match &matrix.values {
                Values::Float(v) => v[start..end].iter().map(|&x| f64::from(x).powi(2)).sum(),
                Values::Int8(v) => v[start..end]
                    .iter()
                    .map(|&x| f64::from(x as i8).powi(2))
                    .sum(),
            };
            if !squared.is_finite() || squared == 0.0 {
                return Err("non-finite or zero corpus vector".into());
            }
            inverse_norms.push((1.0 / squared.sqrt()) as f32);
        }
        Ok(Self {
            matrix,
            inverse_norms,
        })
    }

    pub fn corpus_bytes(&self) -> usize {
        self.matrix.payload_bytes() + self.inverse_norms.len() * 4
    }

    pub fn search(&self, query: &[f32], k: usize, kernel: Kernel) -> Result<Vec<Match>, String> {
        if query.len() != self.matrix.columns || !query.iter().all(|x| x.is_finite()) {
            return Err("invalid query vector".into());
        }
        if kernel == Kernel::Avx2 && Kernel::detect() != Kernel::Avx2 {
            return Err("AVX2 unavailable".into());
        }
        let norm = query
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return Err("zero query vector".into());
        }
        let inverse_query = (1.0 / norm) as f32;
        let k = k.min(self.matrix.rows);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut heap = BinaryHeap::<Match>::with_capacity(k);
        for row in 0..self.matrix.rows {
            let start = row * self.matrix.columns;
            let end = start + self.matrix.columns;
            let dot = match &self.matrix.values {
                Values::Float(v) => dot_float(&v[start..end], query, kernel),
                Values::Int8(v) => dot_int8(&v[start..end], query, kernel),
            };
            let score = dot * self.inverse_norms[row] * inverse_query;
            if !score.is_finite() {
                return Err("non-finite cosine score".into());
            }
            let score = if score == 0.0 {
                0.0
            } else {
                score.clamp(-1.0, 1.0)
            };
            let item = Match { id: row, score };
            if heap.len() < k {
                heap.push(item);
            } else {
                let worst = heap.peek().unwrap();
                if score > worst.score || (score == worst.score && row < worst.id) {
                    *heap.peek_mut().unwrap() = item;
                }
            }
        }
        let mut matches = heap.into_vec();
        matches.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        Ok(matches)
    }
}

fn dot_float(a: &[f32], b: &[f32], kernel: Kernel) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if kernel == Kernel::Avx2 {
        return unsafe { avx2_float(a, b) };
    }
    let _ = kernel;
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn dot_int8(a: &[u8], b: &[f32], kernel: Kernel) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if kernel == Kernel::Avx2 {
        return unsafe { avx2_int8(a, b) };
    }
    let _ = kernel;
    a.iter().zip(b).map(|(&a, b)| f32::from(a as i8) * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_float(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut accumulator = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        let (x, y) = unsafe {
            (
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            )
        };
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(x, y));
        i += 8;
    }
    let mut lanes = [0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    lanes.into_iter().sum::<f32>() + a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_int8(a: &[u8], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut accumulator = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= a.len() {
        let bytes = unsafe { _mm_loadl_epi64(a.as_ptr().add(i).cast::<__m128i>()) };
        let x = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
        let y = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(x, y));
        i += 8;
    }
    let mut lanes = [0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
    lanes.into_iter().sum::<f32>()
        + a[i..]
            .iter()
            .zip(&b[i..])
            .map(|(&x, y)| f32::from(x as i8) * y)
            .sum::<f32>()
}
