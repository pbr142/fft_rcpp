
library(microbenchmark)
sourceCpp('C:/R/src/fft_eigen.cpp')


# sample data
n_fft <- 2^20
h_fft <- 0.01
x <- c(0, ((1:(n_fft)) - 0.5) * h_fft)

f <- diff(plnorm(q = x))

lambda <- 8.84874
size <- 6.8476
p <- lambda / (lambda + size)


compound_sum_r <- function(f, p, r) {
  f_hat <- fft(z = f)
  g_hat <- ((1 - p) / (1 - p * f_hat))^size
  Re(fft(z = g_hat, inverse = TRUE)) / length(g_hat)
}


microbenchmark(compound_sum_r(f = f, p = p, r = size), compound_sum_fft(f = f, p = p, r = size))




