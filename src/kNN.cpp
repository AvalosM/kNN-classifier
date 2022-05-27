#include <algorithm>
#include <numeric>
#include "kNN.hpp"

KNNClassifier::KNNClassifier(unsigned int k_neighbors, unsigned int n_labels) : k_neighbors_(k_neighbors), n_labels_(n_labels) 
{
}

void KNNClassifier::fit(Matrix X, VectorXi y)
{
   this->training_dataset_ = X;
   this->labels_ = y;
}

VectorXi KNNClassifier::predict(Matrix X)
{
    VectorXi predicted_labels = VectorXi(X.rows());
    /* Calculate label prediction for each row of X */
    for (unsigned int i = 0; i < X.rows(); i++) {
        predicted_labels(i) = this->predict_vector(X.row(i));
    }
    return predicted_labels;
}

Label KNNClassifier::predict_vector(const VectorXd &data_vector) const
{
    /* Calculate euclidean distance to training vectors */
    VectorXd distances = (training_dataset_.rowwise() - data_vector.transpose()).rowwise().norm();
    
    VectorXi index_list = VectorXi(training_dataset_.rows());
    std::iota(index_list.begin(), index_list.end(), 0);
    /* Order indexes by distance to data vector */
    std::partial_sort(index_list.begin(), 
                      index_list.begin() + k_neighbors_,
                      index_list.end(),
                      [&distances](unsigned int a, unsigned int b){ return distances(a) < distances(b); });
    
    VectorXi label_occurrences = VectorXi::Zero(n_labels_);
    Label most_occurring = 0;
    /* Count label occurrences in k nearest neighbors and find the most occurring */
    for (unsigned int i = 0; i < k_neighbors_; i++) {
        Label curr_label = labels_(index_list(i));
        most_occurring = (++label_occurrences(curr_label) > label_occurrences(most_occurring)) ? curr_label : most_occurring;
    }
    
    return most_occurring;
}