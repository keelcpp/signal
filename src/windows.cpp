#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

[[nodiscard]] f64 bessel_i0(f64 x) noexcept {
    f64 sum = 1.0;
    f64 term = 1.0;
    for (i32 k = 1; k <= 100; ++k) {
        const f64 ratio = x / (2.0 * as<f64>(k));
        term *= ratio * ratio;
        sum += term;
        if (term < 1e-16 * sum) {
            break;
        }
    }
    return sum;
}

} // namespace

void rectwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    for (usize i = 0; i < n; ++i) {
        out[i] = 1.0;
    }
}

void hann(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 denom = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        out[i] = 0.5 * (1.0 - keel::cos(kTau * as<f64>(i) / denom));
    }
}

void hamming(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 denom = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        out[i] = 0.54 - 0.46 * keel::cos(kTau * as<f64>(i) / denom);
    }
}

void blackman(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 denom = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 x = kTau * as<f64>(i) / denom;
        out[i] = 0.42 - 0.5 * keel::cos(x) + 0.08 * keel::cos(2.0 * x);
    }
}

void bartlett(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 half = as<f64>(n - 1) * 0.5;
    for (usize i = 0; i < n; ++i) {
        out[i] = 1.0 - keel::abs((as<f64>(i) - half) / half);
    }
}

void triang(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 mid = (as<f64>(n) - 1.0) * 0.5;
    if ((n & 1u) == 0u) {
        const f64 denom = as<f64>(n);
        const f64 nm1 = as<f64>(n - 1);
        for (usize i = 0; i < n; ++i) {
            out[i] = 1.0 - keel::abs((2.0 * as<f64>(i) - nm1) / denom);
        }
    } else {
        const f64 denom = as<f64>(n + 1);
        for (usize i = 0; i < n; ++i) {
            out[i] = 1.0 - keel::abs(2.0 * (as<f64>(i) - mid)) / denom;
        }
    }
}

void kaiser(MutArrayRef<f64> out, f64 beta) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 denom = bessel_i0(beta);
    const f64 nm1 = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 alpha = 2.0 * as<f64>(i) / nm1 - 1.0;
        const f64 inside = 1.0 - alpha * alpha;
        out[i] = bessel_i0(beta * keel::sqrt(inside > 0.0 ? inside : 0.0)) / denom;
    }
}

void tukeywin(MutArrayRef<f64> out, f64 r) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (r < 0.0) r = 0.0;
    if (r > 1.0) r = 1.0;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    if (r == 0.0) {
        for (usize i = 0; i < n; ++i) {
            out[i] = 1.0;
        }
        return;
    }
    const f64 nm1 = as<f64>(n - 1);
    const f64 t_edge = r * nm1 * 0.5;
    for (usize i = 0; i < n; ++i) {
        const f64 x = as<f64>(i);
        if (x < t_edge) {
            out[i] = 0.5 * (1.0 + keel::cos(kPi * (x / t_edge - 1.0)));
        } else if (x > nm1 - t_edge) {
            out[i] = 0.5 * (1.0 + keel::cos(kPi * ((x - nm1) / t_edge + 1.0)));
        } else {
            out[i] = 1.0;
        }
    }
}

void flattopwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    constexpr f64 a0 = 0.21557895;
    constexpr f64 a1 = 0.41663158;
    constexpr f64 a2 = 0.277263158;
    constexpr f64 a3 = 0.083578947;
    constexpr f64 a4 = 0.006947368;
    const f64 nm1 = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 x = kTau * as<f64>(i) / nm1;
        out[i] = a0 - a1 * keel::cos(x) + a2 * keel::cos(2.0 * x) - a3 * keel::cos(3.0 * x) +
                 a4 * keel::cos(4.0 * x);
    }
}

void parzenwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 half = as<f64>(n) * 0.5;
    const f64 mid = (as<f64>(n) - 1.0) * 0.5;
    for (usize i = 0; i < n; ++i) {
        const f64 dist = keel::abs(as<f64>(i) - mid);
        const f64 r = dist / half;
        if (dist <= half * 0.5) {
            out[i] = 1.0 - 6.0 * r * r + 6.0 * r * r * r;
        } else {
            const f64 t = 1.0 - r;
            out[i] = 2.0 * t * t * t;
        }
    }
}

void nuttallwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    constexpr f64 a0 = 0.3635819;
    constexpr f64 a1 = 0.4891775;
    constexpr f64 a2 = 0.1365995;
    constexpr f64 a3 = 0.0106411;
    const f64 nm1 = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 x = kTau * as<f64>(i) / nm1;
        out[i] = a0 - a1 * keel::cos(x) + a2 * keel::cos(2.0 * x) - a3 * keel::cos(3.0 * x);
    }
}

void blackmanharris(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    constexpr f64 a0 = 0.35875;
    constexpr f64 a1 = 0.48829;
    constexpr f64 a2 = 0.14128;
    constexpr f64 a3 = 0.01168;
    const f64 nm1 = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 x = kTau * as<f64>(i) / nm1;
        out[i] = a0 - a1 * keel::cos(x) + a2 * keel::cos(2.0 * x) - a3 * keel::cos(3.0 * x);
    }
}

void bohmanwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 half = (as<f64>(n) - 1.0) * 0.5;
    for (usize i = 0; i < n; ++i) {
        const f64 x = keel::abs((as<f64>(i) - half) / half);
        if (x >= 1.0) {
            out[i] = 0.0;
        } else {
            out[i] = (1.0 - x) * keel::cos(kPi * x) + keel::sin(kPi * x) / kPi;
        }
    }
}

void barthannwin(MutArrayRef<f64> out) noexcept {
    const usize n = out.size();
    if (n == 0) return;
    if (n == 1) {
        out[0] = 1.0;
        return;
    }
    const f64 nm1 = as<f64>(n - 1);
    for (usize i = 0; i < n; ++i) {
        const f64 t = as<f64>(i) / nm1 - 0.5;
        out[i] = 0.62 - 0.48 * keel::abs(t) + 0.38 * keel::cos(kTau * t);
    }
}

SignalStatus gausswin(MutArrayRef<f64> out, f64 alpha) noexcept {
    if (alpha <= 0.0) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n = out.size();
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (n == 1) {
        out[0] = 1.0;
        return SignalStatus::kOk;
    }
    const f64 half = as<f64>(n - 1) * 0.5;
    for (usize i = 0; i < n; ++i) {
        const f64 v = (as<f64>(i) - half) / half;
        out[i] = keel::exp(-0.5 * (alpha * v) * (alpha * v));
    }
    return SignalStatus::kOk;
}

SignalStatus chebwin(MutArrayRef<f64> out, f64 at_db, Allocator* alloc) {
    if (at_db <= 0.0) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n = out.size();
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (n == 1) {
        out[0] = 1.0;
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const i64 order = as<i64>(n - 1);
    const f64 order_f = as<f64>(order);
    const f64 r_lin = keel::pow(10.0, at_db / 20.0);
    const f64 beta = keel::cosh(keel::acosh(r_lin) / order_f);

    auto cheb_t = [order, order_f](f64 x) noexcept -> f64 {
        if (keel::abs(x) <= 1.0) {
            return keel::cos(order_f * keel::acos(x));
        }
        const f64 a = keel::cosh(order_f * keel::acosh(keel::abs(x)));
        return (x < 0.0 && ((order & 1) != 0)) ? -a : a;
    };

    const usize k_max = (n - 1) / 2;
    Array<f64> wp(k_max + 1, scratch.allocator());
    const f64 n_f = as<f64>(n);
    for (usize k = 0; k <= k_max; ++k) {
        const f64 ck = beta * keel::cos(kPi * as<f64>(k) / n_f);
        wp[k] = cheb_t(ck);
    }

    const f64 n0 = 0.5 * as<f64>(n - 1);
    const f64 two_pi_over_n = kTau / n_f;
    for (usize idx = 0; idx < n; ++idx) {
        f64 s = wp[0];
        const f64 phase_n = as<f64>(idx) - n0;
        for (usize k = 1; k <= k_max; ++k) {
            s += 2.0 * wp[k] * keel::cos(two_pi_over_n * as<f64>(k) * phase_n);
        }
        out[idx] = s / n_f;
    }

    f64 peak = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 a = keel::abs(out[i]);
        if (a > peak) peak = a;
    }
    if (peak > 0.0) {
        const f64 inv = 1.0 / peak;
        for (usize i = 0; i < n; ++i) {
            out[i] *= inv;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus taylorwin(MutArrayRef<f64> out, i32 nbar, f64 sll_db, Allocator* alloc) {
    if (sll_db >= 0.0 || nbar < 2) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n = out.size();
    if (n == 0) {
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 r = keel::pow(10.0, -sll_db / 20.0);
    const f64 a = keel::log(r + keel::sqrt(r * r - 1.0)) / kPi;
    const f64 nbar_f = as<f64>(nbar);
    const f64 sigma2 = (nbar_f * nbar_f) / (a * a + (nbar_f - 0.5) * (nbar_f - 0.5));

    Array<f64> fm(as<usize>(nbar - 1), 0.0, scratch.allocator());
    for (i32 m = 1; m < nbar; ++m) {
        const f64 m2 = as<f64>(m) * as<f64>(m);
        f64 num = 1.0;
        f64 den = 1.0;
        for (i32 k = 1; k < nbar; ++k) {
            const f64 kf = as<f64>(k) - 0.5;
            const f64 p = sigma2 * (a * a + kf * kf);
            num *= 1.0 - m2 / p;
        }
        for (i32 k = 1; k < nbar; ++k) {
            if (k == m) continue;
            const f64 k2 = as<f64>(k) * as<f64>(k);
            den *= 1.0 - m2 / k2;
        }
        const f64 sign = ((m & 1) != 0) ? 1.0 : -1.0;
        fm[as<usize>(m - 1)] = sign * 0.5 * num / den;
    }

    const f64 nm1 = as<f64>(n - 1);
    const f64 n_f = as<f64>(n);
    for (usize i = 0; i < n; ++i) {
        const f64 xn = as<f64>(i) - nm1 * 0.5;
        f64 w = 1.0;
        for (i32 m = 1; m < nbar; ++m) {
            w += 2.0 * fm[as<usize>(m - 1)] *
                 keel::cos(kTau * as<f64>(m) * xn / n_f);
        }
        out[i] = w;
    }
    return SignalStatus::kOk;
}

SignalStatus dpss(usize n, f64 nw, MutArrayRef<f64> tapers_col_major, MutArrayRef<f64> ratios,
                  Allocator* alloc) {
    usize k = ratios.size();
    if (n == 0 || k == 0) {
        return SignalStatus::kOk;
    }
    if (k > n) k = n;
    if (tapers_col_major.size() < n * k) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 w = nw / as<f64>(n);
    const f64 cos2piw = keel::cos(kTau * w);

    Array<f64> d(n, 0.0, scratch.allocator());
    Array<f64> e(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        const f64 h = 0.5 * (as<f64>(n - 1) - 2.0 * as<f64>(i));
        d[i] = h * h * cos2piw;
        if (i + 1 < n) {
            e[i] = 0.5 * as<f64>(i + 1) * as<f64>(n - 1 - i);
        }
    }

    Array<f64> z(n * n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        z[i + i * n] = 1.0;
    }

    // Symmetric tridiagonal implicit QL with Wilkinson shift (tql2).
    constexpr f64 eps = 2.2204460492503131e-16;
    for (usize l = 0; l < n; ++l) {
        for (i32 iter = 0; iter < 100; ++iter) {
            usize m = l;
            for (; m + 1 < n; ++m) {
                const f64 dd = keel::abs(d[m]) + keel::abs(d[m + 1]);
                if (keel::abs(e[m]) <= eps * dd) break;
            }
            if (m == l) break;
            f64 g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            f64 r = keel::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
            f64 s = 1.0;
            f64 c = 1.0;
            f64 p = 0.0;
            bool underflow = false;
            for (usize ii = m; ii-- > l;) {
                const f64 f = s * e[ii];
                const f64 b = c * e[ii];
                r = keel::hypot(f, g);
                e[ii + 1] = r;
                if (r == 0.0) {
                    d[ii + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[ii + 1] - p;
                r = (d[ii] - g) * s + 2.0 * c * b;
                p = s * r;
                d[ii + 1] = g + p;
                g = c * r - b;
                f64* z0 = z.data() + ii * n;
                f64* z1 = z.data() + (ii + 1) * n;
                for (usize row = 0; row < n; ++row) {
                    const f64 v0 = z0[row];
                    const f64 v1 = z1[row];
                    z1[row] = s * v0 + c * v1;
                    z0[row] = c * v0 - s * v1;
                }
            }
            if (underflow) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    Array<usize> order(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) order[i] = i;
    keel::sort(order.data(), n, [&](usize a, usize b) noexcept { return d[a] > d[b]; });

    Array<f64> sinc_lag(n, scratch.allocator());
    sinc_lag[0] = 2.0 * w;
    for (usize lag = 1; lag < n; ++lag) {
        sinc_lag[lag] = keel::sin(kTau * w * as<f64>(lag)) / (kPi * as<f64>(lag));
    }

    for (usize j = 0; j < k; ++j) {
        const f64* col = z.data() + order[j] * n;
        f64 pol = 0.0;
        if ((j & 1u) == 0u) {
            for (usize idx = 0; idx < n; ++idx) pol += col[idx];
        } else {
            for (usize idx = 0; idx < n; ++idx) {
                pol += (as<f64>(n - 1) - 2.0 * as<f64>(idx)) * col[idx];
            }
        }
        const f64 sign = (pol < 0.0) ? -1.0 : 1.0;
        MutArrayRef<f64> dst_col = tapers_col_major.slice(j * n, n);
        for (usize idx = 0; idx < n; ++idx) {
            dst_col[idx] = sign * col[idx];
        }

        f64 lam = 0.0;
        for (usize idx = 0; idx < n; ++idx) {
            lam += dst_col[idx] * dst_col[idx] * sinc_lag[0];
            for (usize m = idx + 1; m < n; ++m) {
                lam += 2.0 * dst_col[idx] * dst_col[m] * sinc_lag[m - idx];
            }
        }
        ratios[j] = keel::clamp(lam, 0.0, 1.0);
    }

    return SignalStatus::kOk;
}

} // namespace keel::signal
