#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Stats.hpp"

namespace py = pybind11;

PYBIND11_MODULE(stats, m) {
    m.doc() = "Stats library Python bindings";

    m.def("mean", &Stats::mean, py::arg("data"));
    m.def("median", &Stats::median, py::arg("data"));
    m.def("mode", &Stats::mode, py::arg("data"));
    m.def("stdDev", &Stats::stdDev, py::arg("data"));
    m.def("variance", &Stats::variance, py::arg("data"));
    m.def("min", &Stats::min, py::arg("data"));
    m.def("max", &Stats::max, py::arg("data"));
    m.def("range", &Stats::range, py::arg("data"));
    m.def("quartile", &Stats::quartile, py::arg("data"), py::arg("q"));
    m.def("percentile", &Stats::percentile, py::arg("data"), py::arg("p"));
    m.def("covariance", &Stats::covariance, py::arg("x"), py::arg("y"));
    m.def("correlation", &Stats::correlation, py::arg("x"), py::arg("y"));
    m.def("normalize", &Stats::normalize, py::arg("data"));
    m.def("standardize", &Stats::standardize, py::arg("data"));
}
