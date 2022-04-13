#ifndef __EIGEN_HPP__
#define __EIGEN_HPP__

#include "types.hpp"

std::pair<double, VectorXd> power_iteration(const Matrix& mat, unsigned num_iter=5000, double eps=1e-16);

std::pair<VectorXd, Matrix> get_first_eigenvalues(const Matrix& mat, unsigned num, unsigned num_iter=5000, double epsilon=1e-16);

#endif /* __EIGEN_HPP__ */