#ifndef __PCA_HPP__
#define __PCA_HPP__

#include "types.hpp"
#include "eigen.hpp"

class PCA {
public:
    PCA(unsigned int alpha);

    void fit(Matrix data);

    inline void setalpha(unsigned int alpha) { alpha_ = alpha; };

    Matrix transform(Matrix X);
private:
    unsigned int alpha_;
    Matrix V_;
};

#endif /* __PCA_HPP__ */