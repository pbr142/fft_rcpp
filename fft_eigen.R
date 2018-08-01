

library(Rcpp)
library(RcppEigen)
library(microbenchmark)

# sample data
n_fft <- 2^16
h_fft <- 0.01

points <- c(0, ((1:n_fft) - 0.05) * h_fft)
f <- diff(plnorm(q = points))


lambda <- 7.534
size <- 3.5634
p <- lambda / (lambda + size)


# test for correctness

fun <- function(f, p, r) {
    f_hat <- fft(f)
    f_hat <- ((1.0 - p) / (1.0 - p * f_hat))^r
    Re(fft(f_hat, inverse = TRUE)) / length(f)
}

sourceCpp('fft_eigen.cpp')

res_r <- fun(f, p, size)
res_cpp <- compound_sum_fft(f, p, size)

summary(abs(res_r - res_cpp))

microbenchmark(fun(f, p, size), compound_sum_fft(f, p, size))


# test mvfft for correctness

set.seed(1234L)
n_uom <- 15L

lambda <- rlnorm(n_uom)
size <- rlnorm(n_uom)
p <- lambda / (lambda + size)

meanlog <- rnorm(n_uom)
sdlog <- rlnorm(n_uom)

f <- sapply(1:n_uom, function(i) diff(plnorm(q = points, meanlog = meanlog[i], sdlog = sdlog[i])))


# test for correctness

fun <- function(f, p, r) {
    f_hat <- fft(f)
    f_hat <- ((1.0 - p) / (1.0 - p * f_hat))^r
    Re(fft(f_hat, inverse = TRUE)) / length(f)
}


fun_mv <- function(f, p, r) {
    f_hat <- mvfft(f)
    for (i in seq_along(p)) {
        f_hat[,i] <- ((1.0 - p[i]) / (1.0 - p[i] * f_hat[,i]))^r[i]
        
    }
    Re(mvfft(f_hat, inverse = TRUE)) / nrow(f)
}

fun_uni <- function(f, p, r) {
    n_uom <- length(p)
    sapply(1:n_uom, function(i) fun(f[,i], p[i], size[i]))
}

fun_cpp_uni <- function(f, p, r) {
    n_uom <- length(p)
    sapply(1:n_uom, function(i) compound_sum_fft(f[,i], p[i], size[i]))
}

res_r <- fun_mv(f, p, size)

res_r_uni <-
summary(abs(res_r - res_r_uni))

res_cpp <- compound_sum_mvfft(f, p, size)

max(abs(res_r - res_cpp))
max(abs(f - res_cpp))

microbenchmark(fun_mv(f, p, size), fun_uni(f, p, size),
               compound_sum_mvfft(f, p, size),
               fun_cpp_uni(f, p, size))


