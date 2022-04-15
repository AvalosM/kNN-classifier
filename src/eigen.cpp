#include "eigen.hpp"

std::pair<double, VectorXd> power_iteration(const Matrix &A, unsigned int num_iter, double eps)
{
    VectorXd eigvec = VectorXd::Random(A.cols());
    eigvec.normalize();
    double eigval = 0;

    for (unsigned int i = 0; i < num_iter; i++) {
        VectorXd Ab = A * eigvec;

        /* Rayleigh quotient */
        eigval = eigvec.dot(Ab) / eigvec.norm();

        /* Check remainder against epsilon */
        if ((Ab - eigval * eigvec).norm() < eps) break;

        eigvec = Ab;
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
        double eigval = pi_res.first;
        VectorXd eigvec = pi_res.second;

        eigvalues(i) = eigval;
        eigvectors.col(i) = eigvec;

        /* Deflation */
        A = A - eigval * eigvec * eigvec.transpose();
    }

    return std::make_pair(eigvalues, eigvectors);
}