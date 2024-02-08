#pragma once
#include "Common.h"

namespace hcr_imaging {
    
/******************************************************************************************/

struct ScalarBound {
    double minimum, maximum, regularization;

    ScalarBound(double min=0, double max=inf<double>(), double reg=0) : minimum(min), maximum(max), regularization(reg) {
        if (maximum < minimum) throw std::invalid_argument("minimum must not be greater than maximum.");
    }

    template <class T>
    auto operator()(T const &value, Ignore={}, Ignore={}) const {
        return std::clamp<T>(value, minimum, maximum);
    }

    template <class B>
    auto regularize(B const &b) const {return regularization * arma::accu(b);}
};

/******************************************************************************************/

template <class T>
struct VectorBound {
    la::Mat<T> bounds; // (2, N)
    la::Col<T> regularization;

    VectorBound() = default;
    VectorBound(la::Mat<T> b, la::Col<T> reg) : bounds(std::move(b)), regularization(std::move(reg)) {
        if (bounds.n_rows != 2) throw std::invalid_argument("Number of rows must be 2");
        HCR_IMAGING_ALL_EQUAL("Inconsistent dimension", bounds.n_cols, regularization.n_rows);
    }

    auto operator()(T const &value, uint i, Ignore={}) const {
        return std::clamp(value, bounds(0, i), bounds(1, i));
    }

    template <class B>
    auto regularize(B const &b) const {return regularization % b;}
};


/******************************************************************************************/

struct AlternatingOptions {
    std::size_t iters;
    double tolerance;
    bool warm_start;

    AlternatingOptions(std::size_t n=5000, double tol=1e-8, bool warm=false) :
        iters(n), tolerance(tol), warm_start(warm) {
            if (iters == 0) throw std::invalid_argument("number of iterations should be greater than 0");
            if (tolerance <= 0) throw std::invalid_argument("tolerance must be positive.");
        }


    bool operator()(Ignore, Ignore, double obj0, double obj) const {
        // print(obj0, obj, options.tolerance, (obj0 - obj) / options.tolerance);
        return !(obj0 - obj > sq(tolerance) * std::abs(obj));
    }
};


/******************************************************************************************/

template <class T>
struct AlternatingResult {
    uint unconverged = 0, iters = 0;
    T objective = 0;
};

/******************************************************************************************/

template <class A>
struct ClampSolver {
    using value_type = typename std::decay_t<A>::elem_type;
    la::Col<value_type> u, m, a_diag;
    A a;
    AlternatingOptions options;

    ClampSolver(A a0, AlternatingOptions const &ops) : a_diag(a0.diag()), a(static_cast<A &&>(a0)), options{ops} {}

    // solve $\min_x x^T A x - 2 b^T x$ subject to clamp(x)
    // if D is the user domain, clamp(k, x0) should return \min_{x \in D} (x - x0)^2
    template <class X, class B, class F>
    std::tuple<value_type, uint, bool> operator()(X &&x, B const &b, F const &clamp) {
        HCR_IMAGING_ALL_EQUAL("Inconsistent dimensions", a.n_rows, a.n_cols, b.n_rows, x.n_rows);
        if (options.warm_start && x.n_rows == u.n_rows) x = u; // use previous guess

        m = b - a.t() * x - clamp.regularize(b);
        real objective = -(arma::dot(m, x) + arma::dot(b, x)); // x^T A x - 2 x^T b
        uint z = 0;
        bool conv = false;
        for (; z != options.iters; ++z) {
            u = x;
            auto obj = objective;
            for (uint k = 0; k != a.n_cols; ++k) { // O(n^2)
                if (a_diag[k] != 0) { // else x(k) will just be left at 0
                    value_type const t = clamp(x[k] + m[k] / a_diag[k], k); // the latter is the analytic unconstrained argmin
                    if (t != x[k]) m += (x[k] - t) * a.col(k); // old - new (O(n))
                    x[k] = t;
                } else if (x[k] != 0) { // fix up residual if x(k) not already at 0 (e.g. from initialization)
                    m += x[k] * a.col(k); // (O(n))
                    x[k] = 0;
                }
            }
            objective = -(arma::dot(m, x) + arma::dot(b, x));
            if ((conv = options(u, x, obj, objective))) break;
        }
        return {objective, z, !conv};
    }
};

/******************************************************************************************/

template <class A, class C>
struct LogNormalSolver : ClampSolver<A> {
    using base = ClampSolver<A>;
    using base::a, base::a_diag, base::options, base::convergence, base::m, base::u;
    using value_type = typename base::value_type;
    C c;
    la::Col<value_type> d, n, c_diag;

    LogNormalSolver(A a0, C c0, AlternatingOptions const &ops) : base(static_cast<A &&>(a0), ops), c(static_cast<C &&>(c0)), c_diag(c0.diag()) {}

    // solve $\min_x (e^x)^T A e^x - 2 b^T e^x + x^T C x - 2 d^T x$
    template <class X, class B, class D>
    std::tuple<value_type, uint, bool> operator()(X &&x, B const &b, D const &d) {
        HCR_IMAGING_ALL_EQUAL("Inconsistent dimensions", a.n_rows, a.n_cols, b.n_rows, x.n_rows);
        if (options.warm_start) x = u; // use previous guess

        m = b - a.t() * arma::exp(x);
        n = d - c.t() * x;

        uint z = 0;
        for (; z == 0 || (z < options.iters && convergence(u, x)); ++z) {
            u = x;
            for (uint k = 0; k != a.n_cols; ++k) {
                value_type mk = m[k], nk = n[k], xk, ek;
                value_type x0 = mk > 0 ? std::log(mk / a_diag[k]) : 0, e0 = std::exp(x0);

                // Newton solve in 1 dimension
                do {
                    xk = x0 + (e0 * (b - a * e0) - c * x0) / (c + e0 * (2 * a * e0 - b));
                    ek = std::exp(xk);
                    // scalar updates to residuals
                    mk += (ek - e0) * a_diag[k];
                    nk += (xk - x0) * c_diag[k];

                    x0 = xk;
                    e0 = ek;
                } while (xk - xk > 1e-6);

                // vector updates to stored residuals
                m += (ek - std::exp(x[k])) * a.col(k);
                n += (xk - x[k]) * c.col(k);
            }
        }
        m = arma::exp(x);
        return {arma::dot(a.t() * m, m) - 2 * arma::dot(b, m)
              + arma::dot(c.t() * x, x) - 2 * arma::dot(d, x), z != options.iters};
    }
};

/******************************************************************************************/

/*
 * NNLS modified to take A^T * A, A^T * B instead of A, B, and B is la::matrix instead of vector
 * The solution x is modified in place
 * Returns number of unconverged la::columns of B and the objective (x^T A x - 2 x^T b) summed over la::columns of B
 * For a least squares problem, the error is just the objective + b^T b
 * The residual norm^2 is added to in place if provided
 * For least squares, the residual norm is x^T AA x - 2 BA x, so you must add ||B||^2 if you want the true error
 * Returns the number of unconverged points and the total residual squared norm
 * Description: sequential Coordinate-wise algorithm for non-negative least square regression A x = b, s^t. x >= 0
 * Modified from: https://github.com/linxihui/Misc/blob/master/Practice/NMF/nnls.cpp
 * Reference: http://cmp.felk.cvut.cz/ftp/articles/franc/Franc-TR-2005-06.pdf
 * "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem" Franc 2005
 */
template <class F, class A, class T>
AlternatingResult<T> bound_solve(la::Mat<T> &x, A const &a, la::Mat<T> const &b, F const &bound, AlternatingOptions const &ops, la::Col<T> *norm2=nullptr) {
	HCR_IMAGING_ALL_EQUAL("Inconsistent dimensions", b.n_cols, x.n_cols);
    ClampSolver<A const &> solve(a, ops);
    if (norm2 && norm2->n_rows != b.n_cols) {norm2->set_size(b.n_cols); norm2->zeros();}

    AlternatingResult<T> out{0, 0, 0};

    // For each la::column of B
    for (uint i = 0; i != b.n_cols; ++i) {
        auto [err, iters, unconv] = solve(x.col(i), b.col(i), bound);// [&](auto x, auto k) {return bound(x, k, i);});
        if (norm2) (*norm2)(i) += err;
        out.objective += err;
        if (unconv) ++out.unconverged;
        out.iters += iters;
    }
    return out;
}

// Solve diag(v) x = b obeying the constraints that C x <= c
// x should initially be a satisfying solution
template <class T>
void diagonal_bound_solve(la::Col<T> &x, la::Col<T> const &v, la::Col<T> const &b, la::Mat<T> const &C, la::Col<T> const &c) {
    la::Col<T> slack = c - C * x;
    la::Col<T> norm2 = la::square(v % x - b);
    la::uvec order;
    for (uint t = 0; t != 100; ++t) {
        order = la::sort_index(norm2, "descend");
        // for (uint m = 0; m != order.n_rows; ++m) order(m) = m;
        bool done = true;
        for (auto i : order) {
            // Solve for optimal shift disregarding constraints
            T shift = b(i) / v(i) - x(i);
            // Solve for max shift that is allowed in this direction
            for (la::uword j = 0; j != c.n_rows; ++j)
                if (!(C(j, i) * shift <= slack(j))) shift = slack(j) / C(j, i);

            if (shift != 0) {
                slack += -shift * C.col(i);
                x(i) += shift;
                done = false;
                norm2(i) = sq(v(i) * x(i) - b(i));
            }
        }
        if (done) break;
    }
}

// template <class T>
// void diagonal_bound_solve(Mat<T> &x, la::Col<T> const &v, la::Mat<T> const &B, la::Mat<T> const &)

/******************************************************************************************/

// // 2 bounds, each bound either non-existent, 0, or finite. gives 3 * 3 - 1 = 8 possibilities
// template <class A, class T>
// AlternatingResult<T> bound_solve(Mat<T> &x, A const &a, la::Mat<T> const &b, Bounds const &bound, AlternatingOptions const &ops, la::Col<T> *norm2=nullptr) {
//     HCR_IMAGING_REQUIRE(ops.minimum, <=, ops.maximum);
//     // Optimization for unconstrained case
//     if (ops.minimum == real(minf) && ops.maximum == real(inf)) {
//         if constexpr(la::is_sparse<A>) arma::spsolve(x, a, b);
//         else arma::solve(x, a, b);
//         return {0, 0, sq(arma::norm(a * x - b))};
//     }
//     return bound_solve(x, a, b, [min=T(ops.minimum), max=T(ops.maximum)](Ignore, Ignore, T const &x) {
//         return std::clamp(x, min, max);
//     }, ops, norm2);
// }

/******************************************************************************************/

// Non-negative least squares - just solve A^T A x = A^T b instead
template <class A, class B, class F>
auto bound_least_squares(la::Mat<A> const &a, la::Mat<B> const &b, F const &bound, AlternatingOptions const &ops={}) {
	using V = std::common_type_t<A, B>;
    la::Mat<V> x(a.n_cols, b.n_cols, arma::fill::zeros);
    auto res = bound_solve(x, la::Mat<V>(a.t() * a), la::Mat<V>(a.t() * std::move(b)), bound, ops);
    // auto err = arma::norm(b - a * x, "fro");
    res.objective += la::accu(b % b);
    return std::make_pair(std::move(x), std::move(res));
}

/******************************************************************************************/

// Cichocki & Phan: Algorithms for Nonnegative la::matrix and tensor factorization (2008)
template <class T>
auto hals_nmf(la::Mat<T> const &y, std::size_t m, AlternatingOptions const &ops) {
    std::size_t iter;
    la::Mat<T> W, V, P, Q, A, B;

    for (iter = 0; iter != ops.iters; ++iter) {
        W = y.t() * A;
        V = A.t() * A;
        for (std::size_t j = 0; j != m; ++j) {
            B.col(j) += W.col(j) - B * V.col(j);
            // clamp
        }
        P = y.t() * B;
        Q = B.t() * B;
        for (std::size_t j = 0; j != m; ++j) {
            A.col(j) += W.col(j) - A * V.col(j);
            // clamp
            A.col(j) /= arma::norm(A.col(j));
        }
    }
    la::Col<T> w = arma::norm(B);
    B.each_col() /= w;
    return std::make_pair(A, B);
}

/******************************************************************************************/

}
