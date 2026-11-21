#include "keel/signal.h"

namespace keel::signal {

void conv_1d(const f64* a, usize na, const f64* b, usize nb, f64* out) noexcept {
    if (na == 0 || nb == 0 || !a || !b || !out) return;
    const usize nc = na + nb - 1;
    for (usize k = 0; k < nc; ++k) {
        out[k] = 0.0;
    }
    for (usize i = 0; i < na; ++i) {
        const f64 ai = a[i];
        for (usize j = 0; j < nb; ++j) {
            out[i + j] += ai * b[j];
        }
    }
}

void conv_2d(const f64* a, usize m, usize n, const f64* b, usize p, usize q, f64* out) noexcept {
    if (m == 0 || n == 0 || p == 0 || q == 0 || !a || !b || !out) return;
    const usize out_r = m + p - 1;
    const usize out_c = n + q - 1;
    const usize total = out_r * out_c;
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
}

} // namespace keel::signal
