#include "pca.hpp"
#include "Eigen/Eigenvalues"

PCA::PCA(unsigned int alpha) : alpha_(alpha)
{
}

void mean_center(Matrix &data)
{
    /* Center data
     * 
     * X = [data(0) - mean_vector] where mean vector is the column-wise empirical mean
     *     [         ...         ]
     *     [data(n) - mean_vector]
     */
    data.rowwise() -= data.colwise().mean();
}

void PCA::fit(Matrix data)
{
    /* Calculate covariance matrix
     * 
     * X = Mean centered data
     *
     * Mx = X^t * X / n - 1
     */
    mean_center(data);
    Matrix Mx = data.transpose() * data / (data.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Matrix> solver(Mx, Eigen::DecompositionOptions::ComputeEigenvectors);
    
    PC_ = solver.eigenvectors().rowwise().reverse();
    PC_values_ = solver.eigenvalues().colwise().reverse();
}

Matrix PCA::transform(Matrix X)
{
    mean_center(X);
    return X * PC_.leftCols(alpha_);
}