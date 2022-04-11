#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <Eigen/Sparse>
#include <Eigen/Dense>

using Eigen::MatrixXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

typedef Eigen::VectorXd VectorXd;
typedef Eigen::VectorXi VectorXi;

typedef unsigned int Label;

#endif /* __TYPES_HPP__ */