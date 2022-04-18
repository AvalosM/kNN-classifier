#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "kNN.hpp"
#include "pca.hpp"
#include "eigen.hpp"

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
    m.def(
        "power_iteration", &power_iteration,
        "Function that calculates the eigenvalue/eigenvector pair with max |eigval|)",
        py::arg("X"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );
    m.def(
        "get_first_eigenvalues", &get_first_eigenvalues,
        "Function that calculates first num eigenvalue/eigenvector pairs",
        py::arg("X"),
        py::arg("num"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );

}