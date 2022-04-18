#include "pca.hpp"

PCA::PCA(unsigned int alpha) : alpha_(alpha)
{
}

void PCA::fit(Matrix data)
{
    /* Calculate covariance matrix
     * 
     * X = [data(0) - mean_vector]
     *     [         ...         ]
     *     [data(n) - mean_vector]
     * 
     * Mx = X^t * X / n - 1
     */
    data.rowwise() -= (data.colwise().mean());
    Matrix Mx = data.transpose() * data / (data.rows() - 1);

    /* Get first alpha eigenvalue/eigenvector pairs */
    std::pair<VectorXd, Matrix> res = get_first_eigenvalues(Mx, alpha_, 2000, 1e-7);
    PC_values_ = res.first;
    PC_ = res.second;
}

Matrix PCA::transform(Matrix X)
{
    return X * PC_.block(0,0,PC_.rows(), alpha_);
}