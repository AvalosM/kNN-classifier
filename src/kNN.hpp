#ifndef __KNN_HPP__
#define __KNN_HPP__

#include "types.hpp"

class KNNClassifier {
public:
    KNNClassifier(unsigned int k_neighbors, unsigned int n_labels);

    /**
     * @brief Fit classifier with labeled data
     * 
     * @param X Matrix(nxm) where each row corresponds with a data vector of doubles
     * @param y Vector(nx1) labels for the corresponding row of X
     */
    void fit(Matrix X, VectorXi y);

    /**
     * @brief Set the number of neighbors to consider during prediction
     * 
     * Useful to avoid fitting multiple classifiers during testing
     * 
     * @param k_neighbors
     */
    inline void setneighbors(unsigned int k_neighbors) { k_neighbors_ = k_neighbors; };

    /**
     * @brief Predict a label for each row of X
     * 
     * @param X Matrix(nxm) where each row corresponds with a data vector to be labeled
     * @return Vector with the predicted label for each row
     */
    Eigen::VectorXi predict(Matrix X);

private:
    /**
     * @brief Predict a label for a single data vector
     * 
     * @param data_vector to predict a label for
     */
    Label predict_vector(const VectorXd &data_vector) const;
    
    unsigned int k_neighbors_;
    unsigned int n_labels_;
    /* Base */
    Matrix training_dataset_;
    VectorXi labels_;
};

#endif /* __KNN_HPP__ */