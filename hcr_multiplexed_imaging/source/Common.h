#pragma once
#include <armadillo>
#include <stdexcept>
#include <limits>
#include <sstream>
#include <iostream>
#define HCR_MULTIPLEXED_IMAGING_ASSERT(condition, ...) //if (!condition) throw ::unmix::Error(__VA_ARGS__)
#define HCR_MULTIPLEXED_IMAGING_REQUIRE(a, op, b, ...) //HCR_MULTIPLEXED_IMAGING_ASSERT(a op b, a, b, __VA_ARGS__)
#define HCR_MULTIPLEXED_IMAGING_ALL_EQUAL(msg, ...)

namespace hcr_multiplexed_imaging {

template <class ...Ts>
auto print(Ts const &...ts) {
    std::cout << std::boolalpha;
    ((std::cout << ts << "\n"), ...);
    std::cout << std::endl;
}

/******************************************************************************************/

namespace la {

using namespace arma;

template <class T>
struct Depth;

template <class T> struct Depth<Col<T>> {static constexpr unsigned int value = 1;};
template <class T> struct Depth<Mat<T>> {static constexpr unsigned int value = 2;};
template <class T> struct Depth<SpMat<T>> {static constexpr unsigned int value = 2;};
template <class T> struct Depth<Cube<T>> {static constexpr unsigned int value = 3;};

template <class T>
static constexpr unsigned int depth = Depth<std::decay_t<T>>::value;

template <class T> 
struct IsSparse : std::false_type {};

template <class T> 
struct IsSparse<SpMat<T>> : std::true_type {};

template <class T>
static constexpr bool is_sparse = IsSparse<T>::value;

}

struct span {
    la::uword start, stop;
    span(la::uword b, la::uword e) : start(b), stop(e) {}
    operator la::span() const {return la::span(start, stop-1);}
};

/******************************************************************************************/

struct Ignore {
    template <class ...Ts>
    constexpr Ignore(Ts const &...) {}
};

struct NoOp {
    template <class ...Ts>
    void operator()(Ts const &...) const {}
};

template <class T>
auto inf() {return std::numeric_limits<T>::infinity();}

template <class T>
auto sq(T const &t) {return t * t;}

using real = double;
using uint = unsigned int;

/******************************************************************************************/

}