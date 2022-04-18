#include "eigen.hpp"

std::pair<double, VectorXd> power_iteration(const Matrix &A, unsigned int num_iter, double eps)
{
    VectorXd prev_eigvec = VectorXd::Zero(A.cols());
    VectorXd eigvec =  VectorXd::Random(A.cols());
    eigvec.normalize();
    double eigval = 0;

    for (unsigned int i = 0; i < num_iter && (eigvec - prev_eigvec).norm() > eps; i++) {
        prev_eigvec = eigvec;
        eigvec = A * prev_eigvec;
        eigval = prev_eigvec.dot(eigvec);
        eigvec.normalize();
    }
    
    return std::make_pair(eigval, eigvec);
}

std::pair<VectorXd, Matrix> get_first_eigenvalues(const Matrix &X, unsigned int num, unsigned int num_iter, double eps)
{
    Matrix A(X);
    VectorXd eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (unsigned int i = 0; i < num; i++) {
        std::pair<double, VectorXd> pi_res = power_iteration(A, num_iter, eps);

        eigvalues(i) = pi_res.first;
        eigvectors.col(i) = pi_res.second;

        /* Deflation */
        A -= eigvalues(i) * eigvectors.col(i) * eigvectors.col(i).transpose();
    }

    return std::make_pair(eigvalues, eigvectors);
}