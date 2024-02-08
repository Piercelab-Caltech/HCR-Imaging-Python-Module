#pragma once
#include "BoundSolve.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

#define HCR_IMAGING_IF(cond) std::enable_if_t<cond, int> = 0

namespace hcr_imaging {

static constexpr real Pi = 3.141592653589793;

/******************************************************************************************/

struct GaussianOptions {
    AlternatingOptions solver;
    real regularize;
    real truncate;

    GaussianOptions(AlternatingOptions const &ops, real reg=0, real trunc=5) : solver(ops), regularize(reg), truncate(trunc) {
        HCR_IMAGING_REQUIRE(truncate, >, 0);
    }
};

/******************************************************************************************/

/// Positions in a 1D array that have non-negligible contributions from a Gaussian(mu, sigma)
template <class T, class X>
auto gaussian_domain(X const &x, T const &mu, T const &sigma, T const &truncate) {
    HCR_IMAGING_REQUIRE(x.size(), >, 0);
    if (x.size() == 1) return arma::span(0, 0);
    auto const rbox = 1 / (x(1) - x(0));
    auto const lim = x.size() - 1;
    T const min = floor((mu - truncate * sigma) * rbox);
    T const max = ceil((mu + truncate * sigma) * rbox);
    return arma::span(std::clamp<T>(min, 0, lim), std::clamp<T>(max, 0, lim));
}

/******************************************************************************************/

template <class F, class T>
auto fold(F const &f, T &&t) {return t;}

template <class F, class T, class U, class ...Ts>
auto fold(F const &f, T &&t, U &&u, Ts &&...ts) {
    return fold(f, f(std::forward<T>(t), std::forward<U>(u)), std::forward<Ts>(ts)...);
}

template <class V>
auto product(V const &v) {
    return std::accumulate(v.begin(), v.end(), typename V::value_type(1), std::multiplies<>());
}

/******************************************************************************************/

/// Positions in an ND array that have non-negligible contributions from a Gaussian(mu, sigma)
template <class H, class F, std::size_t ...Is, class ...Ts>
auto eval_gaussian_base(H const &h, F const &f, std::index_sequence<Is...>, Ts const &...ts) {
    return fold(std::multiplies<>(), h(ts...), f[Is](ts)...);
}

/// Positions in an ND array that have non-negligible contributions from a Gaussian(mu, sigma)
template <class T, uint N, uint, class G, class M, class F, class H, class X, class ...Ts, HCR_IMAGING_IF(N == sizeof...(Ts))>
T eval_gaussian(G const &, M const &, F const &f, H const &h, X const &, Ts const &...ts) {
    return eval_gaussian_base(h, f, std::make_index_sequence<sizeof...(Ts)>(), ts...);
}

template <class T, uint N, uint D, class G, class M, class F, class H, class X, class ...Ts, HCR_IMAGING_IF(N != sizeof...(Ts))>
T eval_gaussian(G &g, M const &m, F const &f, H const &h, X const &x, Ts const &...ts) {
    constexpr uint I = N - 1 - sizeof...(Ts);
    T const s2 = pow(m(1, I), -2), s3 = pow(m(1, I), -3);
    T const mm = -m(0, I), ms1 = mm + m(1, I), ms2 = mm - m(1, I);
    auto const &xi = x[I];
    T sum = 0;
    for (std::size_t t = 0; t != xi.n_rows; ++t) {
        auto const pdf = eval_gaussian<T, N, D>(g, m, f, h, x, t, ts...);
        sum += pdf;
        if constexpr(D != 0) {
            g(0, I) += pdf * s2 * (xi(t) + mm); // \partial x
            g(1, I) += pdf * s3 * (xi(t) + ms1) * (xi(t) + ms2); // \partial \sigma
        }
    }
    return sum;
}

/******************************************************************************************/

/// Gradient and value of one Gaussian on a grid with a constant image
template <class T, uint N, uint D, class G, class H, class M, class X, std::size_t ...Is>
auto one_gaussian(G &grad, M const &m, H const &h, X const &grid, T const &truncate, std::index_sequence<Is...>) {
    // h - Tensor(x, y ...)
    // m - Matrix(2, # dimensions) of mean and sigma
    // grid - Matrix(# points, # dimensions)
    // Calculate \sum_{x,y,...} Normal(m)(x, y, ...) I(x, y, ...)
    // as well as its derivative w.r.t. m
    std::array<arma::span, N> domains = {gaussian_domain<T>(grid[Is], m(0, Is), m(1, Is), truncate)...}; // the grid slices
    std::array<decltype(grid[0](domains[0])), N> views = {grid[Is](domains[Is])...}; // the local grid
    std::array<la::Col<T>, N> f; // the xyz Gaussian evaluated on respective xyz axes in the grid
    for (std::size_t d = 0; d != N; ++d) {
        f[d] = views[d] - m(0, d);
        f[d] = arma::exp((-T(0.5) / sq(m(1, d))) * (f[d] % f[d])) / sqrt(2 * T(Pi) * sq(m(1, d)));
    }
    auto out = eval_gaussian<T, N, D>(grad, m, std::move(f), h(domains[Is]...), std::move(views));
    HCR_IMAGING_ASSERT(is_finite(out));
    return out;
}

/// Gradient and value of one Gaussian on a grid with a constant image
template <class T, uint N, uint D, class G, class H, class M, class X>
auto one_gaussian(G &grad, M const &m, H const &h, X const &grid, T truncate) {
    return one_gaussian<T, N, D>(grad, m, h, grid, truncate, std::make_index_sequence<la::depth<H>>());
}

/******************************************************************************************/

/// Gradient and value of all two Gaussian products on a grid
// grad: (2, # dimensions = N)
// m: 2, # dimensions
template <class T, uint N, uint D, class G, class M, class X>
auto squared_gaussian(G &grad, M const &m, X const &grid, T const &truncate) {
    std::array<T, N> factors;
    HCR_IMAGING_REQUIRE(grid.size(), ==, N);
    HCR_IMAGING_REQUIRE(grad.n_cols, ==, N);
    HCR_IMAGING_REQUIRE(m.n_cols, ==, N);
    for (std::size_t d = 0; d != grid.size(); ++d) {
        T const mm = -m(0, d), s = m(1, d);
        T const c = 1 / (2 * Pi * sq(s));
        T const a = -1 / sq(s);
        T const s2 = 2 * pow(s, -2), s3 = 2 * pow(s, -3);
        T const m1 = mm - s, m2 = mm + s;

        factors[d] = 0;
        for (auto const x : grid[d](gaussian_domain<T>(grid[d], m(0, d), m(1, d) / sqrt(T(2)), truncate))) {
            auto const pdf = c * std::exp(a * sq(x + mm));
            factors[d] += pdf;
            if constexpr(D != 0) {
                grad(0, d) += pdf * s2 * (x + mm); // \partial x
                grad(1, d) += pdf * s3 * (x + m1) * (x + m2); // \partial \sigma
            }
        }
    }
    T out = product(factors);
    if (out == 0) grad.zeros();
    else for (std::size_t d = 0; d != grid.size(); ++d) grad.col(d) *= out / factors[d];
    return out;
}

/******************************************************************************************/

/// Gradient and value of selected two Gaussian products on a grid
template <class T, uint N, uint D, class G, class M, class X>
auto two_gaussian(G &grad, M const &mi, M const &mj, X const &grid, T const &truncate) {
    std::array<T, N> factors;
    for (std::size_t d = 0; d != N; ++d) {
        T const si = mi(1, d), sj = mj(1, d), ui = mi(0, d), uj = mj(0, d);
        T const vi = sq(si), vj = sq(sj);
        T const c = std::exp(-sq(ui - uj) / 2 / (vi + vj)) / (2 * Pi * si * sj);
        T const s = sqrt((vi * vj) / (vi + vj));
        T const a = -0.5 / sq(s);
        T const mu = (vi * uj + vj * ui) / (vi + vj);

        T const ui1 = -ui - si, ui2 = -ui + si;
        T const uj1 = -uj - sj, uj2 = -uj + sj;
        T const si2 = pow(si, -2), si3 = pow(si, -3);
        T const sj2 = pow(sj, -2), sj3 = pow(sj, -3);

        factors[d] = 0;

        for (auto const x : grid[d](gaussian_domain<T>(grid[d], mu, s, truncate))) {
            auto const pdf = c * std::exp(a * sq(x - mu));
            factors[d] += pdf;

            if constexpr(D != 0) {
                grad(0, d)     += pdf * (x - ui) * si2;
                grad(0, d + N) += pdf * (x - uj) * sj2;

                grad(1, d)     += pdf * (x + ui1) * (x + ui2) * si3;
                grad(1, d + N) += pdf * (x + uj1) * (x + uj2) * sj3;
            }
        }
    }
    T out = product(factors);
    if (out == 0) grad.zeros();
    else if (D) for (std::size_t d = 0; d != grid.size(); ++d) {
        grad.col(d) *= out / factors[d];
        grad.col(d + N) *= out / factors[d];
    }
    return out;
}

/******************************************************************************************/

template <class T, uint N, uint D, class M, class G1, class G2, class ...Ts>
la::SpMat<T> gmm_operator(la::uword z, T regularize, M const &m, G1 &diag_grad, G2 &off_grad, la::umat const &pairs, Ts const &...ts) {
    HCR_IMAGING_REQUIRE(pairs.n_rows, ==, 2);
    auto const np = pairs.n_cols;
    la::umat locs(2, z + 2 * np);
    for (std::size_t n = 0; n != z; ++n) // diagonal indices
        locs(0, n) = locs(1, n) = n;

    if (np) {
        locs.cols(z, z + np-1) = pairs;
        locs.tail_cols(np).row(1) = pairs.row(0);
        locs.tail_cols(np).row(0) = pairs.row(1);
    }

    la::Col<T> vals(locs.n_cols);
    for (std::size_t i = 0; i != z; ++i) // diagonal
        vals(i) = regularize + squared_gaussian<T, N, D>(diag_grad.slice(i), m.slice(i), ts...);
    for (std::size_t n = 0; n != np; ++n) // symmetric offdiagonal
        vals(z + n + np) = vals(z + n) = two_gaussian<T, N, D>(off_grad.slice(n), m.slice(pairs(0, n)), m.slice(pairs(1, n)), ts...);

    return la::SpMat<T>(std::move(locs), std::move(vals), z, z);
}

/******************************************************************************************/

template <class T, uint N, uint D, class M, class G, class ...Ts>
la::Col<T> gmm_rhs(G &grad, M const &m, Ts const &...ts) {
    la::Col<T> b(m.n_slices);
    for (std::size_t i = 0; i != m.n_slices; ++i)
        b(i) = one_gaussian<T, N, D>(grad.slice(i), m.slice(i), ts...);
    HCR_IMAGING_ASSERT(b.is_finite());
    return b;
}

/******************************************************************************************/

/// Returns solution, solved weight vector, gradient vector for position and sigma
/// m: the positions and sigmas (2, dimension, N)
/// returns the objective value, the solved weights, and the gradient (2, dimension, N)
template <bool dXS, class T, class H, class M, class X>
auto gmm_gradient(H const &image, M const &m, X const &grid, la::umat const &pairs, GaussianOptions const &ops, ScalarBound const &bound) {
    // H(...), M(2, dimension, N)
    auto const z = m.n_slices, npairs = pairs.n_cols;
    constexpr auto N = la::depth<H>; // Number of spatial dimensions
    constexpr auto D = 2 * uint(dXS); // 2 if gradient else 0

    // Calculate right hand side and its gradient (if the gaussians are G and the image B, then G.T B)
    la::Cube<T> grad(D, N, z, arma::fill::zeros);
    auto const b = gmm_rhs<T, N, D>(grad, m, image, grid, ops.truncate);

    // Calculate sparse A operator (if the gaussians are G, then G.T G)
    la::Cube<T> diag_grad(D, N, z, arma::fill::zeros); // (x/s, dimension, point)
    la::Cube<T> off_grad(D, 2 * N, npairs, arma::fill::zeros); // (x/s, dimension and left/right index, point)
    auto const A = gmm_operator<T, N, D>(z, ops.regularize, m, diag_grad, off_grad, pairs, grid, ops.truncate);

    // Solve for weights
    la::Col<T> x(b.n_rows, arma::fill::zeros);
    auto conv = bound_solve(x, A, b, bound, ops.solver);
    HCR_IMAGING_REQUIRE(conv.unconverged, ==, 0, "NNLS did not converge");

    // Get objective = x^T A x - 2 x^T b -- (b^T b is constant and not included)
    HCR_IMAGING_ASSERT(all_of(x, is_finite));
    T const objective = dot(x, A * x) - 2 * dot(x, b);

    if (D != 0) {
        // Add the gradient stuff coming from A into the gradient
        for (std::size_t i = 0; i != z; ++i) {
            grad.slice(i) *= -2 * x(i);
            grad.slice(i) += sq(x(i)) * diag_grad.slice(i);
        }
        HCR_IMAGING_ASSERT(all_of(grad, is_finite));

        for (std::size_t n = 0; n != npairs; ++n) {
            auto const xx = 2 * x(pairs(0, n)) * x(pairs(1, n));
            grad.slice(pairs(0, n)) += xx * off_grad.slice(n).head_cols(N);
            grad.slice(pairs(1, n)) += xx * off_grad.slice(n).tail_cols(N);
        }
        HCR_IMAGING_ASSERT(all_of(grad, is_finite));
    }

    return std::make_tuple(objective, std::move(x), std::move(grad));
}

/******************************************************************************************/

template <class T, class H, class M, class X>
auto gmm_squared_loss(H const &image, M const &m, X const &grid, la::umat const &pairs,
                      bool grad, GaussianOptions const &ops, ScalarBound const &bound={}) {
    if (grad) return gmm_gradient<true, T>(image, m, grid, pairs, ops, bound);
    else return gmm_gradient<false, T>(image, m, grid, pairs, ops, bound);
}

/******************************************************************************************/



}
