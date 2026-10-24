use inference_lab::vectors::{Index, Kernel, Matrix, Values};

#[test]
fn signed_codes_tails_and_avx2_match_f64_reference() {
    for dimensions in [1, 7, 8, 9, 31, 64, 640] {
        let codes: Vec<u8> = (0..dimensions * 17)
            .map(|i| ((i * 73 + 11) % 256) as u8)
            .collect();
        let query: Vec<f32> = (0..dimensions)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) / 13.0)
            .collect();
        let index = Index::new(Matrix {
            rows: 17,
            columns: dimensions,
            values: Values::Int8(codes.clone()),
        })
        .unwrap();
        let qnorm = query
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        for kernel in [Kernel::Scalar, Kernel::detect()] {
            let actual = index.search(&query, 17, kernel).unwrap();
            for item in &actual {
                let row = &codes[item.id * dimensions..(item.id + 1) * dimensions];
                let norm = row
                    .iter()
                    .map(|&x| f64::from(x as i8).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let dot = row
                    .iter()
                    .zip(&query)
                    .map(|(&x, &y)| f64::from(x as i8) * f64::from(y))
                    .sum::<f64>();
                assert!((f64::from(item.score) - dot / norm / qnorm).abs() < 2e-6);
            }
        }
    }
}

#[test]
fn float_and_positive_scaled_codes_have_same_cosine_and_tie_order() {
    let codes: Vec<u8> = vec![1, 2, 3, 1, 2, 3, 255, 254, 253, 0, 1, 0];
    let floats: Vec<f32> = codes
        .iter()
        .enumerate()
        .map(|(i, &x)| f32::from(x as i8) * (i / 3 + 1) as f32)
        .collect();
    let quant = Index::new(Matrix {
        rows: 4,
        columns: 3,
        values: Values::Int8(codes),
    })
    .unwrap();
    let float = Index::new(Matrix {
        rows: 4,
        columns: 3,
        values: Values::Float(floats),
    })
    .unwrap();
    let a = quant.search(&[1.0, 2.0, 3.0], 2, Kernel::detect()).unwrap();
    let b = float.search(&[1.0, 2.0, 3.0], 2, Kernel::detect()).unwrap();
    assert_eq!(a.iter().map(|x| x.id).collect::<Vec<_>>(), vec![0, 1]);
    assert_eq!(b.iter().map(|x| x.id).collect::<Vec<_>>(), vec![0, 1]);
    for (a, b) in a.iter().zip(b) {
        assert!((a.score - b.score).abs() < 1e-6);
    }
}

#[test]
fn invalid_vectors_and_shape_are_rejected() {
    assert!(
        Index::new(Matrix {
            rows: 1,
            columns: 2,
            values: Values::Int8(vec![0, 0])
        })
        .is_err()
    );
    assert!(
        Index::new(Matrix {
            rows: 1,
            columns: 2,
            values: Values::Float(vec![f32::NAN, 1.0])
        })
        .is_err()
    );
    assert!(
        Index::new(Matrix {
            rows: 2,
            columns: 2,
            values: Values::Float(vec![1.0])
        })
        .is_err()
    );
    let index = Index::new(Matrix {
        rows: 1,
        columns: 2,
        values: Values::Float(vec![1.0, 0.0]),
    })
    .unwrap();
    assert!(index.search(&[0.0, 0.0], 1, Kernel::Scalar).is_err());
    assert!(index.search(&[1.0], 1, Kernel::Scalar).is_err());
    assert!(
        index
            .search(&[1.0, f32::INFINITY], 1, Kernel::Scalar)
            .is_err()
    );
    assert!(
        index
            .search(&[1.0, 0.0], 0, Kernel::Scalar)
            .unwrap()
            .is_empty()
    );
}
