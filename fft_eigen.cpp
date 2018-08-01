
#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <cmath>

using Eigen::Map;           // Map vectors rather than copying
using Eigen::VectorXd;      // variable size vector, double precision, real values
using Eigen::VectorXcd;     // variable size vector, double precision, complex values
using Eigen::MatrixXd;
using Eigen::MatrixXcd;

// via the depends attribute we tell Rcpp to create hooks for
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
VectorXd compound_sum_fft(Map<VectorXd> f, const double & p, const double & r) {
    
    Eigen::FFT<double> fft;
    
    VectorXcd f_hat(f.rows());
    
    fft.fwd(f_hat, f);
    
    f_hat = (1.0 - p) / (1.0 - p * f_hat.array());
    f_hat = f_hat.array().pow(r);
    
    fft.inv(f, f_hat);
    
    return f;
}


// [[Rcpp::export]]
void compound_sum_mvfft(Map<MatrixXd> f, const std::vector<double> & p, const std::vector<double> & r) {
    
    Eigen::FFT<double> fft;
    VectorXcd f_hat(f.rows());
    
    for (int k = 0; k < f.cols(); ++k ) {
        fft.fwd(f_hat, f.col(k));
        f_hat = (1.0 - p[k]) / (1.0 - p[k] * f_hat.array());
        f_hat = f_hat.array().pow(r[k]);
        f.col(k) = fft.inv(f_hat);
    }
}
  