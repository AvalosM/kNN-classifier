#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "kNN.hpp"
#include "pca.hpp"

namespace py=pybind11;

PYBIND11_MODULE(kNN, m) {
    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<unsigned int, unsigned int>())
        .def("fit", &KNNClassifier::fit)
        .def("setneighbors", &KNNClassifier::setneighbors)
        .def("predict", &KNNClassifier::predict);
    py::class_<PCA>(m, "PCA")
        .def(py::init<unsigned int>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def("setalpha", &PCA::setalpha)
        .def("pc_values", &PCA::pc_values);
}