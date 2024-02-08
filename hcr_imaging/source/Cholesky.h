#pragma once
#include "Common.h"
#include <stdexcept>
#include <fstream>

namespace hcr_imaging {

/******************************************************************************************/

template <class O, class X>
void sub_square(O &&o, X const &x) {o -= la::square(x);}

template <class O, class X>
void add_square(O &&o, X const &x) {o += la::square(x);}

template <class O, class T, class X>
void set_scale(O &&o, T a, X x) {o = a * x;}

template <class O, class T, class X>
void add_scale(O &&o, T a, X x) {o += a * x;}

template <class O, class T, class X, class Y>
void axpy(O &&o, T a, X const &x, Y const &y) {o = a * x + y;}

template <class O, class M, class X>
void matvec_negative(O &&o, M const &m, X const &c) {o = -(m * c);}

template <class O, class M, class X>
void matvec(O &&o, M const &m, X const &c) {o = m * c;}

template <class O, class X, class Y>
void add_outer(O &&o, X const &x, Y const &y) {o += x * y.t();}

template <class X, class Y>
real inner(X const &x, Y const &y) {return la::dot(x, y);}

template <class O, class X, class Y>
void multiply(O &&o, X const &x, Y const &y) {o = x % y;}

template <class O, class X, class Y>
void divide(O &&o, X const &x, Y const &y) {o = x / y;}

template <class T>
void pivot_impl(la::Mat<T> &A, uint i, uint j) {A.swap_rows(i, j); A.swap_cols(i, j);}

template <class T>
void pivot_impl(la::SpMat<T> &A, uint i, uint j) {A.swap_rows(i, j); A.swap_cols(i, j);}

template <class T>
void pivot_impl(la::Col<T> &a, uint i, uint j) {a.swap_rows(i, j);}

/******************************************************************************************/

template <class T, class U, class F, std::size_t ...Is>
void for_each_zip_impl(T &&t, U &&u, F &&f, std::index_sequence<Is...>) {
    (f(std::get<Is>(std::forward<T>(t)), std::get<Is>(std::forward<U>(u))), ...);
}

template <class T, class U, class F>
void for_each_zip(T &&t, U &&u, F &&f) {
    static_assert(std::tuple_size_v<std::decay_t<T>> == std::tuple_size_v<std::decay_t<U>>);
    for_each_zip_impl(std::forward<T>(t), std::forward<U>(u), std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<std::decay_t<T>>>());
}

/******************************************************************************************/

template <class M>
struct CholeskyState {
    using T = typename M::elem_type;
    M R;
    la::Col<T> y;
    la::uword m = 0;

    M U;
    la::Col<T> den, sol, inv, num, diag, gain;
    la::Col<T> s, r, off;
    la::uvec order;

    CholeskyState() = default;

    static M eye(la::uword n) {
        if constexpr(la::is_sparse<M>) return la::speye(n, n);
        else return M(n, n, la::fill::eye);
    }

    CholeskyState(M R0, la::Col<T> y0) : R(std::move(R0)), y(std::move(y0)),
        U(eye(R.n_rows)),
        den(R.diag()),
        sol(R.n_rows, la::fill::zeros),
        inv(R.n_rows, la::fill::zeros),
        num(y),
        diag(num / den),
        gain(num % diag),
        s(R.n_rows, la::fill::none),
        r(R.n_rows, la::fill::none),
        off(R.n_rows, la::fill::none),
        order(la::regspace<la::uvec>(0, R.n_rows-1)) {}


    auto fields()       {return std::tie(R, y, U, den, sol, inv, num, diag, gain, s, r, off, order);}
    auto fields() const {return std::tie(R, y, U, den, sol, inv, num, diag, gain, s, r, off, order);}

    void pivot(uint i, uint j) {
        if (i != j) std::apply([i, j](auto &...ts) {(pivot_impl(ts, i, j), ...);}, fields());
    }

    CholeskyState active() const {
        CholeskyState o;
        o.m = m;
        for_each_zip(o.fields(), fields(), [A=span(0, m)](auto &o, auto const &f) {
            if constexpr(la::depth<decltype(o)> == 1) o = f(A);
            else o = f(A, A);
        });
        return o;
    }

    auto split() const {return std::make_pair(span(0, m), span(m, R.n_rows));}

    T objective() const {
        if (m == 0) return 0;
        auto A = span(0, m);
        return la::dot(y(A), sol(A));
    }

    void calculate_gains() {
        if (m == R.n_rows) return;
        auto const [A, B] = split();
        divide(diag(B), num(B), den(B));
        multiply(gain(B), num(B), diag(B));
    }

    // Augment with state m. O(A B) if growth else O(A)
    void augment(bool growth) {
        ++m;
        auto const [A, B] = split();
        set_scale(r(A), 1 / -std::sqrt(den(m-1)), U(A, m-1)); // O(A)
        add_scale(sol(A), inner(r(A), y(A)), r(A)); // O(A)
        add_square(inv(A), r(A)); // O(A)
        if (growth && m != R.n_rows) {
            matvec_negative(s(B), R(A, B).t(), r(A)); // O(A B), less if sparse
            add_scale(num(B), inner(y(A), r(A)), s(B)); // O(B)
            sub_square(den(B), s(B)); // O(B)
            add_outer(U(A, B), r(A), s(B)); // O(A B)
        }
    }

    void shed(bool growth) {
        auto const [A, B] = split();
        set_scale(r(A), 1 / -std::sqrt(den[m-1]), U(A, m-1)); // O(A);
        add_scale(sol(A), -inner(r(A), y(A)), r(A)); // O(A)
        sub_square(inv(A), r(A)); // O(A)
        if (growth && m != R.n_rows) {
            matvec(s(B), R(A, B).t(), r(A)); // O(A B), less if sparse
            add_scale(num(B), inner(y(A), r(A)), s(B)); // O(B)
            add_square(den(B), s(B)); // O(B)
            add_outer(U(A, B), r(A), s(B)); // O(A B)
        }
        --m;
    }

    // Calculate the new solution for the interior if point i is added
    auto offdiag(la::uword i) {
        auto const A = span(0, m);
        axpy(off(A), diag(i), U(A, i), sol(A));
        return off(A);
    }

    // Calculate the loss if point i is removed
    T loss(la::uword i) const {return sq(sol(i)) / inv(i);}
};

/******************************************************************************************/

template <class T>
void remove_negative_solutions(CholeskyState<T> &chol, bool growth) {
    uint const m0 = chol.m;
    for (la::uword t = 0; t != m0; ++t) {
        uint k;
        real least = inf<real>();
        for (la::uword i = 0; i != chol.m; ++i) {
            if (chol.sol(i) < 0) {
                auto loss = chol.loss(i);
                if (loss < least) {
                    least = loss;
                    k = i;
                }
            }
        }
        if (least == inf<real>()) return;
        chol.pivot(chol.m-1, k);
        chol.shed(growth);
    }
    throw std::runtime_error("failure in remove_negative_solutions()");
}

/******************************************************************************************/

template <class M, class F=NoOp>
CholeskyState<M> cholesky_pivot_nnls(M R, la::Col<typename M::elem_type> y, F &&callback={}) {
    CholeskyState<M> chol(std::move(R), std::move(y));
    using T = typename M::elem_type;
    // while (chol.m < chol.R.n_rows) {
    for (uint iter = 0; iter <= 10000; ++iter) {
        if (chol.m == chol.R.n_rows) break;
        if (iter == 10000) throw std::runtime_error("yikes");
        chol.calculate_gains();
        callback(chol);
        // if (i) check(ct, chol, 1e-8, true);
        try {
            la::uvec const order = la::sort_index(chol.gain(span(chol.m, chol.R.n_rows)), "descend");
        } catch (...) {
            print(chol.m, chol.R.n_rows);
            print(chol.gain.n_rows);
            print(chol.gain);
        }
        la::uvec const order = la::sort_index(chol.gain(span(chol.m, chol.R.n_rows)), "descend");

        uint j;
        T best = 0;
        T const current = chol.objective();

        for (auto o : order) {
            if (chol.gain(chol.m + o) <= best) {
                break;
            } else if (chol.diag(chol.m + o) <= 0) {
                continue;
            } else if (chol.m && chol.offdiag(chol.m + o).min() < 0) {
                auto copy = chol.active();
                copy.pivot(chol.m + o, copy.m);
                copy.augment(true);
                remove_negative_solutions(copy, false);
                if (current - copy.objective() > best) {
                    best = current - copy.objective();
                    j = chol.m + o;
                }
            } else {
                j = chol.m + o;
                best = chol.gain(chol.m + o);
            }
        }
        if (best == 0) break;

        chol.pivot(chol.m, j);
        chol.augment(true);

        remove_negative_solutions(chol, true);
    }
    return chol;
}

/******************************************************************************************/

template <class T, class M>
la::Col<T> cholesky_nnls(M A, la::Col<T> b) {
    la::Col<T> x(b.n_rows, la::fill::zeros);
    if (b.min() > 0) {
        auto chol = cholesky_pivot_nnls(std::move(A), std::move(b));
        span s(0, chol.m);
        x(chol.order(s)) = chol.sol(s);
    }
    return x;
}

/******************************************************************************************/

}