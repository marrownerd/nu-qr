use nalgebra::{DMatrix, DVector};

pub struct CustomSVD {
    pub u: DMatrix<f64>,
    pub s: DVector<f64>,
    pub v: DMatrix<f64>,
}

impl CustomSVD {
    pub fn new(a: &DMatrix<f64>) -> Self {
        let (m, n) = a.shape();
        let mut b = a.clone();
        let mut v = DMatrix::<f64>::identity(n, n);

        let max_sweeps = 100;
        let tol = 1e-15;

        for _ in 0..max_sweeps {
            let mut max_corr: f64 = 0.0;

            for j in 0..n {
                for k in (j + 1)..n {
                    let mut alpha = 0.0;
                    let mut beta = 0.0;
                    let mut gamma = 0.0;

                    for i in 0..m {
                        alpha += b[(i, j)] * b[(i, j)];
                        beta += b[(i, k)] * b[(i, k)];
                        gamma += b[(i, j)] * b[(i, k)];
                    }

                    if alpha * beta > 0.0 {
                        max_corr = max_corr.max(gamma.abs() / (alpha * beta).sqrt());
                    }

                    if gamma.abs() > tol * (alpha * beta).sqrt() {
                        let zeta = (beta - alpha) / (2.0 * gamma);
                        let t = if zeta >= 0.0 {
                            1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                        } else {
                            -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                        };

                        let c = 1.0 / (1.0 + t * t).sqrt();
                        let s = c * t;

                        for i in 0..m {
                            let t1 = b[(i, j)];
                            let t2 = b[(i, k)];
                            b[(i, j)] = c * t1 - s * t2;
                            b[(i, k)] = s * t1 + c * t2;
                        }

                        for i in 0..n {
                            let v1 = v[(i, j)];
                            let v2 = v[(i, k)];
                            v[(i, j)] = c * v1 - s * v2;
                            v[(i, k)] = s * v1 + c * v2;
                        }
                    }
                }
            }

            if max_corr < tol {
                break;
            }
        }

        let mut s = DVector::zeros(n);
        let mut u = DMatrix::zeros(m, n);

        for j in 0..n {
            let mut norm = 0.0;
            for i in 0..m {
                norm += b[(i, j)] * b[(i, j)];
            }
            let sigma = norm.sqrt();
            s[j] = sigma;

            if sigma > 1e-14 {
                for i in 0..m {
                    u[(i, j)] = b[(i, j)] / sigma;
                }
            }
        }

        for i in 0..n {
            let mut max_idx = i;
            for j in (i + 1)..n {
                if s[j] > s[max_idx] {
                    max_idx = j;
                }
            }
            if max_idx != i {
                s.swap_rows(i, max_idx);
                u.swap_columns(i, max_idx);
                v.swap_columns(i, max_idx);
            }
        }

        CustomSVD { u, s, v }
    }

    pub fn solve(&self, b: &DVector<f64>) -> DVector<f64> {
        let n = self.s.len();
        let ut_b = self.u.transpose() * b;

        let mut scaled = DVector::zeros(n);
        for i in 0..n {
            if self.s[i] > 1e-12 {
                scaled[i] = ut_b[i] / self.s[i];
            }
        }

        &self.v * scaled
    }
}