#include <iomanip>
#include <stdexcept>
#define ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_LAPACK
// #define ARMA_DONT_USE_BLAS
#include "BoundSolve.h"
#include "Cholesky.h"
#include "Gaussian.h"
#include "Python.h"
#include <array>
#include <memory>
#include <type_traits>

namespace hcr_multiplexed_imaging::py {

struct BufferDeleter {
    void operator()(Py_buffer *o) const noexcept {PyBuffer_Release(o);}
};

auto get_buffer(PyObject *o) {
    auto v = std::make_unique<Py_buffer>();
    if (0 == PyObject_GetBuffer(o, v.get(), PyBUF_F_CONTIGUOUS | PyBUF_WRITABLE | PyBUF_FORMAT))
        return std::unique_ptr<Py_buffer, BufferDeleter>(v.release());
    throw std::runtime_error("Invalid input array");
}

template <class T>
std::string format() {
    if constexpr(std::is_same_v<T, float>) return "f";
    if constexpr(std::is_same_v<T, double>) return "d";
    if constexpr(std::is_same_v<T, unsigned long long>) return "Q";
}

template <class T> 
struct Type {using type = T;};

template <class F>
auto dispatch_dtype(PyObject *o, F &&f) {
    std::string const s = get_buffer(o)->format;
    if (s == "f") return f(Type<float>());
    if (s == "d") return f(Type<double>());
    throw std::runtime_error("unsupported type");
}

template <class T>
auto get_array(PyObject *o, int dim) {
    auto v = get_buffer(o);
    if (v->ndim != dim) throw std::runtime_error("Expected ndim == " + std::to_string(dim) + " but got " + std::to_string(v->ndim));
    if (v->format != format<T>()) throw std::runtime_error("Incorrect dtype");
    return v;
}

template <int N, class T>
auto array(PyObject* o) {
    auto v = get_array<T>(o, N);
    if constexpr(N == 1) return la::Col<T>(reinterpret_cast<T *>(v->buf), v->shape[0], false, false);
    if constexpr(N == 2) return la::Mat<T>(reinterpret_cast<T *>(v->buf), v->shape[0], v->shape[1], false, false);
    if constexpr(N == 3) return la::Cube<T>(reinterpret_cast<T *>(v->buf), v->shape[0], v->shape[1], v->shape[2], false, false);
}

template <class F>
auto call_noexcept(F &&f) -> decltype(f()) {
    try {
        return f();
    } catch (std::exception const &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

template <class F>
auto dispatch_ndim(PyObject *o, F &&f) {
    auto const ndim = get_buffer(o)->ndim;
    if (ndim == 2) return f(std::integral_constant<int, 2>());
    if (ndim == 3) return f(std::integral_constant<int, 3>());
    throw std::runtime_error("unsupported type");
}

static PyObject* gmm(PyObject* self, PyObject* args, PyObject* kws) noexcept {
    static std::array<char const *, 14> names = {"image", "m", "grid", "pairs", "weights",
                                    "gradient", "iters", "min", "max", 
                                    "tolerance", "warm_start", "regularize", "truncate", nullptr};
    PyObject* py_image, *py_m, *py_grid, *py_pairs, *py_gradient, *py_weights;
    unsigned long long iters;
    double min, max, tolerance, regularize, truncate;
    int warm_start;
    if (!PyArg_ParseTupleAndKeywords(args, kws, "OOOOOO" "kdd" "dpdd", const_cast<char **>(names.data()),
        &py_image, &py_m, &py_grid, &py_pairs, &py_weights, &py_gradient,
        &iters, &min, &max,
        &tolerance, &warm_start, &regularize, &truncate)) return nullptr;
    if (!PyTuple_Check(py_grid)) return nullptr;

    return call_noexcept([&] {
        auto const pairs = la::conv_to<la::umat>::from(array<2, unsigned long long>(py_pairs));
        ScalarBound const bound(min, max);
        GaussianOptions const ops(AlternatingOptions(iters, tolerance, warm_start), regularize, truncate);
        return dispatch_dtype(py_image, [&](auto t) {
            using T = typename decltype(t)::type;
            auto const m = array<3, T>(py_m);
            std::vector<la::Col<T>> grid;
            std::size_t n = PyTuple_Size(py_grid);
            for (std::size_t i = 0; i != n; ++i)
                grid.emplace_back(array<1, T>(PyTuple_GetItem(py_grid, i)));

            return dispatch_ndim(py_image, [&](auto const N) {
                auto const image = array<decltype(N)::value, T>(py_image);
                auto const [objective, weights, grad] = gmm_squared_loss<T>(image, m, grid, pairs, py_gradient != Py_None, ops, bound);
                if (py_gradient != Py_None) array<3, T>(py_gradient) = grad;
                array<1, T>(py_weights) = weights;
                return PyFloat_FromDouble(objective);
            });
        });
    });
}

static PyObject* cholesky_nnls(PyObject* self, PyObject* args, PyObject* kws) noexcept {
    static std::array<char const *, 4> const names = {"X", "A", "B", nullptr};
    PyObject *X, *A, *B;

    if (!PyArg_ParseTupleAndKeywords(args, kws, "OOO", const_cast<char **>(names.data()),
        &X, &A, &B)) return nullptr;

    return call_noexcept([&] {return dispatch_dtype(A, [&](auto t) {
        using T = typename decltype(t)::type;
        auto x = array<2, T>(X);
        auto const a = array<2, T>(A);
        auto const b = array<2, T>(B);
        if (x.n_rows != b.n_rows) throw std::runtime_error("bad");
        if (x.n_cols != b.n_cols) throw std::runtime_error("bad");
        if (a.n_rows != a.n_cols) throw std::runtime_error("bad");
        if (a.n_rows != b.n_rows) throw std::runtime_error("bad");

        for (la::uword i = 0; i != x.n_cols; ++ i) {
            try {
                x.col(i) = hcr_multiplexed_imaging::cholesky_nnls(a, la::Col<T>(b.col(i)));
            } catch (...) {
                // print(x, a, b.col(i));
                std::cout << std::setprecision(25);
                a.raw_print("A");
                b.col(i).eval().raw_print("B");
                // std::cout << std::setprecision(16);
                // for (auto x : a) std::cout << x << " ";
                // std::cout << std::endl;
                // for (auto x : b) std::cout << x << " ";
                // std::cout << std::endl;
                throw;
            }
        }
        Py_RETURN_NONE;
    });});
}

static PyObject* bound_solve(PyObject* self, PyObject* args, PyObject* kws) noexcept {
    static std::array<char const *, 10> const names = {"X", "A", "B", "min", "max", "iters", "tolerance", "regularize", "warm_start", nullptr};
    PyObject *X, *A, *B;
    unsigned long long iters;
    double min, max, tolerance, regularize;
    int warm_start;

    if (!PyArg_ParseTupleAndKeywords(args, kws, "OOOddkddp", const_cast<char **>(names.data()),
        &X, &A, &B, &min, &max, &iters, &tolerance, &regularize, &warm_start)) return nullptr;
    return call_noexcept([&] {return dispatch_dtype(A, [&](auto t) {
        using T = typename decltype(t)::type;
        ScalarBound bound(min, max, regularize);
        AlternatingOptions alt(iters, tolerance, warm_start);
        auto x = array<2, T>(X);
        AlternatingResult<T> result = bound_solve(x, array<2, T>(A), array<2, T>(B), bound, alt);
        return PyFloat_FromDouble(result.objective);
    });});
}

static PyMethodDef Methods[] = {
    {"bound_solve", (PyCFunction)(void(*)(void)) hcr_multiplexed_imaging::py::bound_solve, METH_VARARGS | METH_KEYWORDS, "Solve bounded least-squares problem"},
    {"cholesky_nnls", (PyCFunction)(void(*)(void)) hcr_multiplexed_imaging::py::cholesky_nnls, METH_VARARGS | METH_KEYWORDS, "Solve non-negative least-squares problem using Cholesky method"},
    {"gmm", (PyCFunction)(void(*)(void)) hcr_multiplexed_imaging::py::gmm, METH_VARARGS | METH_KEYWORDS, "Solve Gaussian mixture model objective and gradient"},
    {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};


static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "core", 
    "C++ extension module for spectral unmixing and dot detection",
    -1,    
    Methods
};

}


PyMODINIT_FUNC PyInit_cpp(void) {
    return PyModule_Create(&hcr_multiplexed_imaging::py::module);
}
