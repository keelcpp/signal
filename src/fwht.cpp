#include "keel/signal.h"

#include "keel/array.h"

namespace keel::signal {

namespace {

constexpr bool is_pow2(usize n) noexcept {
    return n != 0 && (n & (n - 1)) == 0;
}

constexpr i32 log2_pow2(usize n) noexcept {
    i32 k = 0;
    while ((usize{1} << k) < n) {
        ++k;
    }
    return k;
}

constexpr usize bit_reverse(usize i, i32 bits) noexcept {
    usize r = 0;
    for (i32 b = 0; b < bits; ++b) {
        r = (r << 1) | (i & usize{1});
        i >>= 1;
    }
    return r;
}

void hadamard_butterfly(f64* col, usize n) noexcept {
    for (usize h = 1; h < n; h <<= 1) {
        for (usize i = 0; i < n; i += (h << 1)) {
            for (usize j = 0; j < h; ++j) {
                const f64 a = col[i + j];
                const f64 b = col[i + j + h];
                col[i + j] = a + b;
                col[i + j + h] = a - b;
            }
        }
    }
}

void permute_dyadic(const f64* in, f64* out, usize n, i32 log2n) noexcept {
    for (usize i = 0; i < n; ++i) {
        out[i] = in[bit_reverse(i, log2n)];
    }
}

void permute_sequency(const f64* in, f64* out, usize n, i32 log2n) noexcept {
    for (usize i = 0; i < n; ++i) {
        const usize g = i ^ (i >> 1);
        out[i] = in[bit_reverse(g, log2n)];
    }
}

void unpermute_sequency(const f64* in, f64* out, usize n, i32 log2n) noexcept {
    for (usize i = 0; i < n; ++i) {
        const usize g = i ^ (i >> 1);
        out[bit_reverse(g, log2n)] = in[i];
    }
}

SignalStatus fwht_impl(ArrayRef<f64> x, MutArrayRef<f64> out, FwhtOrdering ordering, bool forward,
                       Allocator* alloc) {
    const usize n = out.size();
    if (!is_pow2(n)) {
        return SignalStatus::kInvalidArgument;
    }
    const i32 log2n = log2_pow2(n);
    const usize to_copy = keel::min(x.size(), n);

    Array<f64> buf(n, 0.0, alloc);
    if (forward) {
        for (usize i = 0; i < to_copy; ++i) {
            buf[i] = x[i];
        }
    } else {
        Array<f64> raw(n, 0.0, alloc);
        for (usize i = 0; i < to_copy; ++i) {
            raw[i] = x[i];
        }
        if (ordering == FwhtOrdering::kHadamard) {
            for (usize i = 0; i < n; ++i) {
                buf[i] = raw[i];
            }
        } else if (ordering == FwhtOrdering::kDyadic) {
            permute_dyadic(raw.data(), buf.data(), n, log2n);
        } else {
            unpermute_sequency(raw.data(), buf.data(), n, log2n);
        }
    }

    hadamard_butterfly(buf.data(), n);

    if (forward) {
        const f64 inv = 1.0 / as<f64>(n);
        for (usize i = 0; i < n; ++i) {
            buf[i] *= inv;
        }
        if (ordering == FwhtOrdering::kHadamard) {
            for (usize i = 0; i < n; ++i) {
                out[i] = buf[i];
            }
        } else if (ordering == FwhtOrdering::kDyadic) {
            permute_dyadic(buf.data(), out.data(), n, log2n);
        } else {
            permute_sequency(buf.data(), out.data(), n, log2n);
        }
    } else {
        for (usize i = 0; i < n; ++i) {
            out[i] = buf[i];
        }
    }
    return SignalStatus::kOk;
}

} // namespace

SignalStatus fwht(ArrayRef<f64> x, MutArrayRef<f64> out, FwhtOrdering ordering, Allocator* alloc) {
    return fwht_impl(x, out, ordering, true, alloc);
}

SignalStatus ifwht(ArrayRef<f64> x, MutArrayRef<f64> out, FwhtOrdering ordering, Allocator* alloc) {
    return fwht_impl(x, out, ordering, false, alloc);
}

} // namespace keel::signal
