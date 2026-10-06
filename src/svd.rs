use std::cmp;
use std::ops::Mul;
use num::{Float, Signed};

use ApproxEq;
use Matrix;
use EigenDecomposition;

pub struct SVD<T> {
  u: Matrix<T>,
  s: Matrix<T>,
  v: Matrix<T>,
}

impl<T: Float + Signed + ApproxEq<T>> SVD<T> {
  pub fn new(a: &Matrix<T>) -> SVD<T> {
    if a.rows() < a.cols() {
      let svd = SVD::new(&a.t());
      return SVD {
        u: svd.v,
        s: svd.s.t(),
        v: svd.u,
      };
    }

    let ata = a.t().mul(a);
    let edc = EigenDecomposition::new(&ata);
    let v = edc.get_v().clone();
    let eigs = edc.get_real_eigenvalues();
    let singular_values: Vec<T> = eigs.iter().map(|&e| e.sqrt()).collect();

    let s_size = singular_values.len();
    let s = Matrix::block_diag(s_size, s_size, singular_values);
    let s_inv = s.inverse().unwrap();

    let (s_aug, s_inv_aug) = if a.rows() == a.cols() {
      (s, s_inv)
    } else {
      (
        s.cb(&Matrix::zero(a.rows() - a.cols(), s.cols())),
        s_inv.cr(&Matrix::zero(s_inv.rows(), a.rows() - a.cols())),
      )
    };

    let u = a.mul(&v).mul(&s_inv_aug);

    SVD {
      u,
      s: s_aug,
      v,
    }
  }

  pub fn get_u(&self) -> &Matrix<T> {
    &self.u
  }

  pub fn get_s(&self) -> &Matrix<T> {
    &self.s
  }

  pub fn get_v(&self) -> &Matrix<T> {
    &self.v
  }

  pub fn rank(&self) -> usize {
    let eps: T = num::cast(2.0f64.powf(-52.0)).unwrap();
    let max_dim: T = num::cast(cmp::max(self.u.rows(), self.v.rows())).unwrap();
    let tol = max_dim * self.s.get(0, 0) * eps;
    let mut r = 0;
    for i in 0..self.s.rows() {
      if self.s.get(i, i) > tol {
        r += 1;
      }
    }
    r
  }
}

#[test]
fn svd_test() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_m_over_n() {
  let a = m!(1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 10.0, 11.0, 12.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_n_over_m() {
  let a = m!(1.0, 2.0, 3.0, 4.0; 5.0, 6.0, 7.0, 8.0; 9.0, 10.0, 11.0, 12.0);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}

#[test]
fn svd_test_matrix_from_laba() {
  println!("\nQR 3x2 ");
  let mut a = DMatrix::<f64>::from_row_slice(3, 2, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
  ]);
  let svd = SVD::new(&a);
  let u = svd.get_u();
  let s = svd.get_s();
  let v = svd.get_v();
  assert!((u * s * v.t()).approx_eq(&a));
}