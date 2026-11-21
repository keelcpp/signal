#include "keel/signal.h"

namespace keel::signal {

SignalStatus conv_1d(ArrayRef<f64> a, ArrayRef<f64> b, MutArrayRef<f64> out) noexcept {
    const usize na = a.size();
    const usize nb = b.size();
    if (na == 0 || nb == 0) {
        return SignalStatus::kOk;
    }
    const usize nc = na + nb - 1;
    if (out.size() < nc) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize k = 0; k < nc; ++k) {
        out[k] = 0.0;
    }
    for (usize i = 0; i < na; ++i) {
        const f64 ai = a[i];
        for (usize j = 0; j < nb; ++j) {
            out[i + j] += ai * b[j];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus conv_2d(ArrayRef<f64> a, usize m, usize n, ArrayRef<f64> b, usize p, usize q,
                     MutArrayRef<f64> out) noexcept {
    if (m == 0 || n == 0 || p == 0 || q == 0) {
        return SignalStatus::kOk;
    }
    const usize out_r = m + p - 1;
    const usize out_c = n + q - 1;
    const usize total = out_r * out_c;
    if (a.size() < m * n || b.size() < p * q || out.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize idx = 0; idx < total; ++idx) {
        out[idx] = 0.0;
    }
    for (usize j = 0; j < q; ++j) {
        for (usize i = 0; i < p; ++i) {
            const f64 bij = b[j * p + i];
            if (bij == 0.0) continue;
            for (usize cc = 0; cc < n; ++cc) {
                const usize out_col = j + cc;
                for (usize rr = 0; rr < m; ++rr) {
                    out[out_col * out_r + (i + rr)] += a[cc * m + rr] * bij;
                }
            }
        }
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
