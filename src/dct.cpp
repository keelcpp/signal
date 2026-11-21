#include "keel/signal.h"

#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

void dct2_direct(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept {
    const usize n = x.size();
    const f64 w0 = keel::sqrt(1.0 / as<f64>(n));
    const f64 wk = keel::sqrt(2.0 / as<f64>(n));
    const f64 pi_over_2n = kPi / (2.0 * as<f64>(n));
    for (usize k = 0; k < n; ++k) {
        f64 acc = 0.0;
        const f64 ang_k = pi_over_2n * as<f64>(k);
        for (usize i = 0; i < n; ++i) {
            acc += x[i] * keel::cos(ang_k * as<f64>(2 * i + 1));
        }
        out[k] = (k == 0 ? w0 : wk) * acc;
    }
}

void idct2_direct(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept {
    const usize n = x.size();
    const f64 w0 = keel::sqrt(1.0 / as<f64>(n));
    const f64 wk = keel::sqrt(2.0 / as<f64>(n));
    const f64 pi_over_2n = kPi / (2.0 * as<f64>(n));
    for (usize i = 0; i < n; ++i) {
        f64 acc = w0 * x[0];
        const f64 ang_n = pi_over_2n * as<f64>(2 * i + 1);
        for (usize k = 1; k < n; ++k) {
            acc += wk * x[k] * keel::cos(ang_n * as<f64>(k));
        }
        out[i] = acc;
    }
}

void dct2_fft(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = x.size();
    const usize m = 2 * n;
    Array<f64> y(m, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        y[i] = x[i];
        y[m - 1 - i] = x[i];
    }
    Array<c64> spec(n + 1, c64(0.0, 0.0), alloc);
    keel::fft::rfft_1d(y.as_ref(), spec.as_mut_ref());

    const f64 w0 = keel::sqrt(1.0 / as<f64>(n));
    const f64 wk = keel::sqrt(2.0 / as<f64>(n));
    const f64 pi_over_2n = kPi / (2.0 * as<f64>(n));
    for (usize k = 0; k < n; ++k) {
        const f64 phase = -pi_over_2n * as<f64>(k);
        const c64 rot(keel::cos(phase), keel::sin(phase));
        const f64 v = (spec[k] * rot).real() * 0.5;
        out[k] = (k == 0 ? w0 : wk) * v;
    }
}

void idct2_fft(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = x.size();
    const usize m = 2 * n;
    const f64 w0 = keel::sqrt(1.0 / as<f64>(n));
    const f64 wk = keel::sqrt(2.0 / as<f64>(n));
    const f64 pi_over_2n = kPi / (2.0 * as<f64>(n));

    Array<c64> half(n + 1, c64(0.0, 0.0), alloc);
    for (usize k = 0; k < n; ++k) {
        const f64 vk = ((k == 0) ? (x[0] / w0) : (x[k] / wk)) * 2.0;
        const f64 phase = pi_over_2n * as<f64>(k);
        half[k] = vk * c64(keel::cos(phase), keel::sin(phase));
    }
    half[n] = c64(0.0, 0.0);

    Array<f64> y(m, 0.0, alloc);
    keel::fft::irfft_1d(half.as_ref(), y.as_mut_ref(), m);
    for (usize i = 0; i < n; ++i) {
        out[i] = y[i];
    }
}

void dct1_direct(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept {
    const usize n = x.size();
    if (n == 1) {
        out[0] = x[0];
        return;
    }
    const f64 scale = keel::sqrt(2.0 / as<f64>(n - 1));
    const f64 s0 = kInvSqrt2;
    const f64 pi_over_nm1 = kPi / as<f64>(n - 1);
    auto beta = [&](usize i) noexcept -> f64 { return (i == 0 || i == n - 1) ? s0 : 1.0; };
    for (usize k = 0; k < n; ++k) {
        f64 acc = 0.0;
        for (usize i = 0; i < n; ++i) {
            acc += beta(i) * x[i] * keel::cos(pi_over_nm1 * as<f64>(i) * as<f64>(k));
        }
        out[k] = scale * beta(k) * acc;
    }
}

void dct4_direct(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept {
    const usize n = x.size();
    const f64 scale = keel::sqrt(2.0 / as<f64>(n));
    const f64 pi_over_n = kPi / as<f64>(n);
    for (usize k = 0; k < n; ++k) {
        f64 acc = 0.0;
        const f64 kk = as<f64>(k) + 0.5;
        for (usize i = 0; i < n; ++i) {
            acc += x[i] * keel::cos(pi_over_n * (as<f64>(i) + 0.5) * kk);
        }
        out[k] = scale * acc;
    }
}

void dct2_impl(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = x.size();
    if (n == 1) {
        out[0] = x[0];
        return;
    }
    if (n >= 16) {
        dct2_fft(x, out, alloc);
    } else {
        dct2_direct(x, out);
    }
}

void idct2_impl(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = x.size();
    if (n == 1) {
        out[0] = x[0];
        return;
    }
    if (n >= 16) {
        idct2_fft(x, out, alloc);
    } else {
        idct2_direct(x, out);
    }
}

} // namespace

SignalStatus dct(ArrayRef<f64> x, MutArrayRef<f64> out, i32 type, Allocator* alloc) {
    if (type < 1 || type > 4 || out.size() < x.size()) {
        return SignalStatus::kInvalidArgument;
    }
    if (x.empty()) {
        return SignalStatus::kOk;
    }
    // Support in-place calls safely.
    Array<f64> tmp(alloc);
    ArrayRef<f64> src = x;
    if (x.data() == out.data()) {
        tmp = Array<f64>::from(x, alloc);
        src = tmp.as_ref();
    }
    if (type == 1) {
        dct1_direct(src, out);
    } else if (type == 2) {
        dct2_impl(src, out, alloc);
    } else if (type == 3) {
        idct2_impl(src, out, alloc);
    } else {
        dct4_direct(src, out);
    }
    return SignalStatus::kOk;
}

SignalStatus idct(ArrayRef<f64> x, MutArrayRef<f64> out, i32 type, Allocator* alloc) {
    if (type < 1 || type > 4 || out.size() < x.size()) {
        return SignalStatus::kInvalidArgument;
    }
    if (x.empty()) {
        return SignalStatus::kOk;
    }
    Array<f64> tmp(alloc);
    ArrayRef<f64> src = x;
    if (x.data() == out.data()) {
        tmp = Array<f64>::from(x, alloc);
        src = tmp.as_ref();
    }
    if (type == 1) {
        dct1_direct(src, out);
    } else if (type == 2) {
        idct2_impl(src, out, alloc);
    } else if (type == 3) {
        dct2_impl(src, out, alloc);
    } else {
        dct4_direct(src, out);
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
