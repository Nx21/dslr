#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>
#include "Matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matrix, m) {
    m.doc() = "Matrix library Python bindings";

    py::class_<Matrix>(m, "Matrix")
        // Constructors
        .def(py::init<>())
        .def(py::init<size_t, size_t, double>(),
             py::arg("rows"), py::arg("cols"), py::arg("value") = 0.0)
        .def(py::init<const std::vector<std::vector<double>>&>(),
             py::arg("data"))

        // Getters
        .def("getRows", &Matrix::getRows)
        .def("getCols", &Matrix::getCols)

        // Element access
        .def("__getitem__", [](const Matrix& m, std::pair<size_t, size_t> idx) {
            return m(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& m, std::pair<size_t, size_t> idx, double val) {
            m(idx.first, idx.second) = val;
        })

        // Arithmetic operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(py::self / double())
        .def(double() * py::self)

        // Functions
        .def("transpose",  &Matrix::transpose)
        .def("dot",        &Matrix::dot,       py::arg("other"))
        .def("fill",       &Matrix::fill,      py::arg("value"))
        .def("randomize",  &Matrix::randomize, py::arg("min") = -1.0, py::arg("max") = 1.0)
        .def("print",      &Matrix::print)

        // Static factory methods
        .def_static("zeros",    &Matrix::zeros,    py::arg("rows"), py::arg("cols"))
        .def_static("ones",     &Matrix::ones,     py::arg("rows"), py::arg("cols"))
        .def_static("identity", &Matrix::identity, py::arg("size"))

        // Vector operations
        .def("toVector",    &Matrix::toVector)
        .def_static("fromVector", &Matrix::fromVector,
             py::arg("vec"), py::arg("asColumn") = true)

        // Property operations
        .def("getRow", &Matrix::getRow, py::arg("row"))
        .def("getCol", &Matrix::getCol, py::arg("col"))
        .def("setRow", &Matrix::setRow, py::arg("row"), py::arg("values"))
        .def("setCol", &Matrix::setCol, py::arg("col"), py::arg("values"))

        // String representation
        .def("__repr__", [](const Matrix& m) {
            std::ostringstream oss;
            oss << m;
            return oss.str();
        });
}