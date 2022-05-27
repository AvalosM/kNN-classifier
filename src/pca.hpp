#ifndef __PCA_HPP__
#define __PCA_HPP__

#include "types.hpp"

class PCA {
public:
    PCA(unsigned int alpha);

    void fit(Matrix data);

    Matrix transform(Matrix X);

    inline void setalpha(unsigned int alpha) { alpha_ = alpha; };

    inline VectorXd pc_values() { return PC_values_; };

private:
    unsigned int alpha_;
    Matrix PC_;
    VectorXd PC_values_;
};

#endif /* __PCA_HPP__ */