use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;
use nalgebra::{DMatrix, DVector};

fn legendre_row(n: usize, eps: f64) -> Vec<f64> {
    let mut p = vec![0.0; n + 1];
    p[0] = 1.0;
    if n >= 1 {
        p[1] = eps;
    }
    for i in 1..n {
        let i_f = i as f64;
        p[i + 1] = ((2.0 * i_f + 1.0) * eps * p[i] - i_f * p[i - 1]) / (i_f + 1.0);
    }
    p
}

fn load_data(filename: &str) -> io::Result<(Vec<f64>, Vec<f64>)> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut eps = Vec::new();
    let mut sigma = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        if let (Some(e_str), Some(s_str)) = (parts.next(), parts.next()) {
            if let (Ok(e), Ok(s)) = (e_str.parse::<f64>(), s_str.parse::<f64>()) {
                eps.push(e);
                sigma.push(s);
            }
        }
    }
    Ok((eps, sigma))
}

fn build_matrix_a(eps: &[f64], n: usize) -> DMatrix<f64> {
    let m = eps.len();
    let mut a = DMatrix::zeros(m, n + 1);
    for i in 0..m {
        let row = legendre_row(n, eps[i]);
        for j in 0..=n {
            a[(i, j)] = row[j];
        }
    }
    a
}

fn cond2_rect(a: &DMatrix<f64>) -> f64 {
    let svd = a.clone().svd(false, false);
    let s = svd.singular_values;
    let s_max = s[0];
    let s_min = s[s.len() - 1];
    if s_min == 0.0 { f64::INFINITY } else { s_max / s_min }
}

fn cond2_square(ata: &DMatrix<f64>) -> f64 {
    let svd = ata.clone().svd(false, false);
    let s = svd.singular_values;
    let s_max = s[0];
    let s_min = s[s.len() - 1];
    if s_min == 0.0 { f64::INFINITY } else { s_max / s_min }
}

fn calc_nrmse(a: &DMatrix<f64>, c: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let m = b.len() as f64;
    let pred = a * c;
    let mut sum_sq = 0.0;
    let mut max_b: f64 = f64::MIN;

    for i in 0..b.len() {
        let diff = pred[i] - b[i];
        sum_sq += diff * diff;
        if b[i] > max_b {
            max_b = b[i];
        }
    }

    (1.0 / max_b) * (sum_sq / m).sqrt()
}

fn solve_nu(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<(DVector<f64>, f64)> {
    let start = Instant::now();
    let at = a.transpose();
    let ata = &at * a;
    let atb = &at * b;

    let chol = ata.cholesky()?;
    let c = chol.solve(&atb);
    let time = start.elapsed().as_secs_f64();
    Some((c, time))
}

fn solve_givens(a_orig: &DMatrix<f64>, b_orig: &DVector<f64>) -> (DVector<f64>, f64) {
    let start = Instant::now();
    let mut a = a_orig.clone();
    let mut b = b_orig.clone();
    let (m, cols) = a.shape();

    let mut p: Vec<usize> = (0..cols).collect();

    for j in 0..cols {
        let mut max_col = j;
        let mut max_norm = 0.0;
        for c in j..cols {
            let mut norm = 0.0;
            for r in j..m {
                norm += a[(r, c)] * a[(r, c)];
            }
            if norm > max_norm {
                max_norm = norm;
                max_col = c;
            }
        }

        if max_col != j {
            a.swap_columns(j, max_col);
            p.swap(j, max_col);
        }

        for i in (j + 1)..m {
            if a[(i, j)] != 0.0 {
                let x = a[(j, j)];
                let y = a[(i, j)];
                let r = (x * x + y * y).sqrt();
                let c_rot = x / r;
                let s_rot = y / r;

                for k in j..cols {
                    let tau1 = a[(j, k)];
                    let tau2 = a[(i, k)];
                    a[(j, k)] = c_rot * tau1 + s_rot * tau2;
                    a[(i, k)] = -s_rot * tau1 + c_rot * tau2;
                }

                let b_j = b[j];
                let b_i = b[i];
                b[j] = c_rot * b_j + s_rot * b_i;
                b[i] = -s_rot * b_j + c_rot * b_i;
            }
        }
    }

    let mut c_perm = DVector::zeros(cols);
    for i in (0..cols).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..cols {
            sum += a[(i, k)] * c_perm[k];
        }
        c_perm[i] = (b[i] - sum) / a[(i, i)];
    }

    let mut c = DVector::zeros(cols);
    for i in 0..cols {
        c[p[i]] = c_perm[i];
    }

    let time = start.elapsed().as_secs_f64();
    (c, time)
}

fn test_matrix_qr_givens() {
    println!("\nQR 3x2 ");
    let mut r = DMatrix::<f64>::from_row_slice(3, 2, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let mut q = DMatrix::<f64>::identity(3, 3);
    let pairs = [(1, 0, 0), (2, 0, 0), (2, 1, 1)];

    for &(i, j, col) in &pairs {
        let x: f64 = r[(j, col)];
        let y: f64 = r[(i, col)];
        if y != 0.0 {
            let rad = (x * x + y * y).sqrt();
            let c = x / rad;
            let s = y / rad;

            for k in 0..2 {
                let tau1 = r[(j, k)];
                let tau2 = r[(i, k)];
                r[(j, k)] = c * tau1 + s * tau2;
                r[(i, k)] = -s * tau1 + c * tau2;
            }

            for k in 0..3 {
                let tau1 = q[(k, j)];
                let tau2 = q[(k, i)];
                q[(k, j)] = c * tau1 + s * tau2;
                q[(k, i)] = -s * tau1 + c * tau2;
            }
        }
    }

    for val in r.iter_mut() {
        if val.abs() < 1e-14 {
            *val = 0.0;
        }
    }

    println!("Q:\n{:.4}", q);
    println!("R:\n{:.4}", r);
    println!("Q * R:\n{:.1}", &q * &r);
}

fn main() {
    test_matrix_qr_givens();

    let filename = "data_5.txt";
    let (eps, sigma) = match load_data(filename) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return;
        }
    };

    let b = DVector::from_vec(sigma);

    println!("\n{:-<105}", "");
    println!(
        "{:<3} | {:<12} | {:<12} | {:<10} | {:<12} | {:<12} | {:<10}",
        "N", "cond2(A^T*A)", "NRMSE", "Время", "cond2(A)", "NRMSE(QR)", "Время(QR)"
    );
    println!("{:-<105}", "");

    for n in 0..=10 {
        let a = build_matrix_a(&eps, n);
        let ata = a.transpose() * &a;

        let cond_ata = cond2_square(&ata);
        let cond_a = cond2_rect(&a);

        let (nrmse_nu_str, time_nu_str) = match solve_nu(&a, &b) {
            Some((c_nu, time_nu)) => {
                let nrmse = calc_nrmse(&a, &c_nu, &b);
                (format!("{:<12.4e}", nrmse), format!("{:<10.5}", time_nu))
            }
            None => ("CholeskyFail".to_string(), "---       ".to_string()),
        };

        let (c_qr, time_qr) = solve_givens(&a, &b);
        let nrmse_qr = calc_nrmse(&a, &c_qr, &b);

        println!(
            "{:<3} | {:<12.2e} | {} | {} | {:<12.2e} | {:<12.4e} | {:<10.5}",
            n, cond_ata, nrmse_nu_str, time_nu_str, cond_a, nrmse_qr, time_qr
        );
    }
    println!("{:-<105}", "");
}