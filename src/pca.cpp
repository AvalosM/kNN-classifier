#include "pca.hpp"

PCA::PCA(unsigned int alpha) : alpha_(alpha)
{
}

void PCA::fit(Matrix data)
{
    /* Calculate median value for each pixel
     *
     *     1xn             nxm         1x1                         1xm
     *
     * [1, ..., 1] * [i00, ..., i0m] * 1/n = [(i00 + ... + in0)/n, ..., (i0m + ... + inm)/n]
     *               [..., ..., ...]
     *               [in0, ..., inm]
     */
    VectorXd mean_vector = data.colwise().mean();

    /* Calculate covariance matrix
     * 
     * X = [data(0) - mean_vector]
     *     [         ...         ]
     *     [data(n) - mean_vector]
     * 
     * Mx = X^t * X / n - 1
     */
    data.rowwise() -= mean_vector.transpose();
    Matrix Mx = data.transpose() * data / (data.rows() - 1);

    /* Get first alpha eigenvalue/eigenvector pairs */
    V_ = get_first_eigenvalues(Mx, alpha_, 5000, 1e-8).second;
}

Matrix PCA::transform(Matrix X)
{
    return X * V_.block(0,0,V_.rows(), alpha_);
}