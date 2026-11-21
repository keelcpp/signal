#include "keel/signal.h"

#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

SignalStatus hilbert(ArrayRef<f64> x, MutArrayRef<c64> out, Allocator* alloc) {
    const usize n = x.size();
    if (out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }
    (void)alloc;

    MutArrayRef<c64> buf = out.slice(0, n);
    keel::fft::fft_1d_real(x, buf);

    // Analytic spectrum mask: double positive frequencies, keep DC (and Nyquist when n even),
    // zero negative frequencies.
    for (usize i = 1; i < (n + 1) / 2; ++i) {
        buf[i] *= 2.0;
    }
    for (usize i = n / 2 + 1; i < n; ++i) {
        buf[i] = c64(0.0, 0.0);
    }

    keel::fft::ifft_1d_inplace(buf);
    return SignalStatus::kOk;
}

SignalStatus envelope_analytic(ArrayRef<f64> x, MutArrayRef<f64> upper, MutArrayRef<f64> lower,
                               Allocator* alloc) {
    const usize n = x.size();
    if (upper.size() < n || (!lower.empty() && lower.size() < n)) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }

    f64 mean = 0.0;
    for (usize i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= as<f64>(n);

    Array<f64> xc(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        xc[i] = x[i] - mean;
    }
    Array<c64> h(n, c64(0.0, 0.0), alloc);
    const SignalStatus st = hilbert(xc.as_ref(), h.as_mut_ref(), alloc);
    if (st != SignalStatus::kOk) {
        return st;
    }
    for (usize i = 0; i < n; ++i) {
        const f64 a = keel::abs(h[i]);
        upper[i] = mean + a;
        if (!lower.empty()) {
            lower[i] = mean - a;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus envelope_fir(ArrayRef<f64> x, usize n_taps, MutArrayRef<f64> upper,
                          MutArrayRef<f64> lower, Allocator* alloc) {
    const usize n = x.size();
    if (n_taps == 0 || upper.size() < n || (!lower.empty() && lower.size() < n)) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }

    f64 mean = 0.0;
    for (usize i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= as<f64>(n);

    Array<f64> xc(n, 0.0, alloc);
    for (usize i = 0; i < n; ++i) {
        xc[i] = x[i] - mean;
    }

    Array<f64> kw(n_taps, 0.0, alloc);
    kaiser(kw.as_mut_ref(), 8.0);

    Array<c64> filt(n_taps, c64(0.0, 0.0), alloc);
    f64 norm_re = 0.0;
    for (usize k = 0; k < n_taps; ++k) {
        const f64 t = 0.5 * (as<f64>(k) - 0.5 * (as<f64>(n_taps) - 1.0));
        const f64 s = (keel::abs(t) < 1e-15) ? 1.0 : (keel::sin(kPi * t) / (kPi * t));
        const f64 w = kw[k];
        const c64 h(s * w * keel::cos(kPi * t), s * w * keel::sin(kPi * t));
        filt[k] = h;
        norm_re += h.real();
    }
    const f64 inv_norm = 1.0 / norm_re;
    for (usize k = 0; k < n_taps; ++k) {
        filt[k] *= inv_norm;
    }

    const usize pad = n_taps / 2;
    for (usize i = 0; i < n; ++i) {
        c64 acc(0.0, 0.0);
        for (usize k = 0; k < n_taps; ++k) {
            const i64 src = as<i64>(i) + as<i64>(pad) - as<i64>(k);
            if (src >= 0 && as<usize>(src) < n) {
                acc += filt[k] * xc[as<usize>(src)];
            }
        }
        const f64 a = keel::abs(acc);
        upper[i] = mean + a;
        if (!lower.empty()) {
            lower[i] = mean - a;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus envelope_rms(ArrayRef<f64> x, usize window_len, MutArrayRef<f64> upper,
                          MutArrayRef<f64> lower) noexcept {
    const usize n = x.size();
    if (window_len == 0 || upper.size() < n || (!lower.empty() && lower.size() < n)) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }

    f64 mean = 0.0;
    for (usize i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= as<f64>(n);

    const i64 left = as<i64>(window_len / 2);
    const i64 right = as<i64>(window_len - 1) - left;
    for (usize i = 0; i < n; ++i) {
        const i64 lo = keel::max(i64{0}, as<i64>(i) - left);
        const i64 hi = keel::min(as<i64>(n) - 1, as<i64>(i) + right);
        f64 s = 0.0;
        i64 cnt = 0;
        for (i64 j = lo; j <= hi; ++j) {
            const f64 d = x[as<usize>(j)] - mean;
            s += d * d;
            ++cnt;
        }
        const f64 a = keel::sqrt(s / as<f64>(cnt));
        upper[i] = mean + a;
        if (!lower.empty()) {
            lower[i] = mean - a;
        }
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
