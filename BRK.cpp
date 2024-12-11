#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

double BRK(py::array_t<double> q1, py::array_t<double> q2, py::array_t<double> n)
{
    py::buffer_info qBuf1 = q1.request(), qBuf2 = q2.request(), nBuf = n.request();
    
    double *Q1 = static_cast<double *>(qBuf1.ptr);
    double *Q2 = static_cast<double *>(qBuf2.ptr);
    double *N = static_cast<double *>(nBuf.ptr);

    // Do the BRK function

    //Return the GB energy
}

PYBIND11_MODULE(BRK, m)
{
    m.doc() = "BRK function binding";

    m.def("BRK", &BRK, "The BRK function for determining GB Energy from GB crystallography",
    py::arg("q1"), py::arg("q2"), py::arg("n"));
}