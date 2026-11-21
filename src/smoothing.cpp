#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

inline bool is_nan_scalar(f64 v) noexcept { return keel::is_nan(v); }
inline bool is_nan_scalar(c64 v) noexcept { return keel::is_nan(v.real()) || keel::is_nan(v.imag()); }
inline f64 norm_scalar(f64 v) noexcept { return v * v; }
inline f64 norm_scalar(c64 v) noexcept { return keel::norm(v); }
inline f64 conj_scalar(f64 v) noexcept { return v; }
inline c64 conj_scalar(c64 v) noexcept { return keel::conj(v); }

template <typename Scalar>
Array<Scalar> fillgaps_fwd_pass(ArrayRef<Scalar> col, usize maxlen, usize order, Allocator* alloc) {
    const usize n = col.size();
    Array<Scalar> v = Array<Scalar>::from(col, alloc);
    usize i_first = n;
    for (usize i = 0; i < n; ++i) {
        if (!is_nan_scalar(v[i])) {
            i_first = i;
            break;
        }
    }
    if (i_first == n) {
        return v;
    }

    usize k = i_first + 1;
    while (k < n) {
        if (!is_nan_scalar(v[k])) {
            ++k;
            continue;
        }
        usize g_end = k;
        while (g_end + 1 < n && is_nan_scalar(v[g_end + 1])) {
            ++g_end;
        }
        const usize gap = g_end - k + 1;
        const usize s_start =
            (maxlen > 0 && (k - i_first) > maxlen) ? (k - maxlen) : i_first;
        const usize seg_len = k - s_start;

        Scalar mu{};
        for (usize i = s_start; i < k; ++i) {
            mu += v[i];
        }
        mu /= as<f64>(seg_len);

        Array<Scalar> seg0(seg_len, Scalar{}, alloc);
        f64 e0 = 0.0;
        for (usize i = 0; i < seg_len; ++i) {
            seg0[i] = v[s_start + i] - mu;
            e0 += norm_scalar(seg0[i]);
        }
        e0 /= as<f64>(seg_len);

        if (seg_len <= 1 || !(e0 > 0.0)) {
            for (usize m = 0; m < gap; ++m) {
                v[k + m] = mu;
            }
            k = g_end + 1;
            continue;
        }

        const usize max_p = (order > 0) ? keel::min(order, seg_len - 1) : (seg_len - 1);
        Array<Scalar> ef = seg0.clone(alloc);
        Array<Scalar> eb = seg0.clone(alloc);
        Array<Scalar> a(max_p + 1, Scalar{}, alloc);
        a[0] = Scalar(1.0);
        f64 err = e0;

        usize best_p = max_p;
        f64 best_aic = Limits<f64>::infinity();
        Array<Scalar> best_a(alloc);

        for (usize m = 1; m <= max_p; ++m) {
            Scalar num{};
            f64 den = 0.0;
            for (usize idx = m; idx < seg_len; ++idx) {
                num += ef[idx] * conj_scalar(eb[idx - 1]);
                den += norm_scalar(ef[idx]) + norm_scalar(eb[idx - 1]);
            }
            const Scalar km = (den > 0.0) ? (Scalar(-2.0) * num / den) : Scalar{};

            for (usize idx = seg_len - 1; idx >= m; --idx) {
                const Scalar f = ef[idx];
                const Scalar b = eb[idx - 1];
                ef[idx] = f + km * b;
                eb[idx] = b + conj_scalar(km) * f;
            }

            Array<Scalar> a_next = a.clone(alloc);
            for (usize j = 1; j < m; ++j) {
                a_next[j] = a[j] + km * conj_scalar(a[m - j]);
            }
            a_next[m] = km;
            a = keel::move(a_next);

            err *= keel::max(0.0, 1.0 - norm_scalar(km));

            if (order == 0) {
                const f64 aic_m = as<f64>(seg_len) * keel::log(keel::max(err, 1e-300)) +
                                  2.0 * as<f64>(m);
                if (m == 1 || aic_m < best_aic) {
                    best_aic = aic_m;
                    best_p = m;
                    best_a.assign(a.slice(0, m + 1));
                } else if (m - best_p >= 31) {
                    break;
                }
            }
        }

        if (order > 0) {
            best_p = max_p;
            best_a = keel::move(a);
        }

        Array<Scalar> z(best_p + gap, Scalar{}, alloc);
        for (usize i = 0; i < best_p; ++i) {
            z[i] = seg0[seg_len - best_p + i];
        }
        for (usize m = 0; m < gap; ++m) {
            Scalar acc{};
            for (usize j = 1; j <= best_p; ++j) {
                acc -= best_a[j] * z[best_p + m - j];
            }
            z[best_p + m] = acc;
            v[k + m] = acc + mu;
        }
        k = g_end + 1;
    }
    return v;
}

template <typename Scalar>
SignalStatus fillgaps_impl(ArrayRef<Scalar> x, usize maxlen, usize order, MutArrayRef<Scalar> out,
                           Allocator* alloc) {
    const usize n = x.size();
    if (out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (maxlen > 0 && order > 0 && order >= maxlen) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }

    usize i_first = n;
    usize i_last = n;
    for (usize i = 0; i < n; ++i) {
        if (!is_nan_scalar(x[i])) {
            if (i_first == n) i_first = i;
            i_last = i;
        }
    }
    if (i_first == n) {
        for (usize i = 0; i < n; ++i) out[i] = x[i];
        return SignalStatus::kOk;
    }

    const Array<Scalar> yf = fillgaps_fwd_pass(x, maxlen, order, alloc);
    Array<Scalar> rev(n, Scalar{}, alloc);
    for (usize i = 0; i < n; ++i) {
        rev[i] = x[n - 1 - i];
    }
    const Array<Scalar> yb_rev = fillgaps_fwd_pass(rev.as_ref(), maxlen, order, alloc);
    Array<Scalar> yb(n, Scalar{}, alloc);
    for (usize i = 0; i < n; ++i) {
        yb[i] = yb_rev[n - 1 - i];
    }

    for (usize i = 0; i < n; ++i) out[i] = x[i];
    for (usize i = 0; i < i_first; ++i) out[i] = yb[i];
    for (usize i = i_last + 1; i < n; ++i) out[i] = yf[i];

    usize k = i_first + 1;
    while (k < i_last) {
        if (!is_nan_scalar(x[k])) {
            ++k;
            continue;
        }
        usize g_end = k;
        while (g_end + 1 < i_last && is_nan_scalar(x[g_end + 1])) {
            ++g_end;
        }
        const usize gap = g_end - k + 1;
        for (usize m = 0; m < gap; ++m) {
            const f64 w = as<f64>(m + 1) / as<f64>(gap + 1);
            out[k + m] = (1.0 - w) * yf[k + m] + w * yb[k + m];
        }
        k = g_end + 1;
    }
    return SignalStatus::kOk;
}

SignalStatus gauss_jordan(f64* a, f64* b, i32 n, i32 m) noexcept {
    for (i32 k = 0; k < n; ++k) {
        i32 piv = k;
        f64 max_abs = keel::abs(a[k * n + k]);
        for (i32 r = k + 1; r < n; ++r) {
            const f64 v = keel::abs(a[r * n + k]);
            if (v > max_abs) {
                max_abs = v;
                piv = r;
            }
        }
        if (max_abs < 1e-300) {
            return SignalStatus::kSingular;
        }
        if (piv != k) {
            for (i32 c = 0; c < n; ++c) keel::swap(a[k * n + c], a[piv * n + c]);
            for (i32 c = 0; c < m; ++c) keel::swap(b[k * m + c], b[piv * m + c]);
        }
        const f64 inv = 1.0 / a[k * n + k];
        for (i32 c = 0; c < n; ++c) a[k * n + c] *= inv;
        for (i32 c = 0; c < m; ++c) b[k * m + c] *= inv;
        for (i32 r = 0; r < n; ++r) {
            if (r == k) continue;
            const f64 f = a[r * n + k];
            if (f == 0.0) continue;
            for (i32 c = 0; c < n; ++c) a[r * n + c] -= f * a[k * n + c];
            for (i32 c = 0; c < m; ++c) b[r * m + c] -= f * b[k * m + c];
        }
    }
    return SignalStatus::kOk;
}

Array<f64> build_vandermonde(i32 order, i32 framelen, Allocator* alloc) {
    const usize n = as<usize>(framelen);
    const usize p = as<usize>(order + 1);
    Array<f64> v(n * p, 0.0, alloc);
    const f64 half = as<f64>(framelen / 2);
    for (usize i = 0; i < n; ++i) {
        f64 val = 1.0;
        const f64 x = as<f64>(i) - half;
        for (usize k = 0; k < p; ++k) {
            v[i * p + k] = val;
            val *= x;
        }
    }
    return v;
}

} // namespace

SignalStatus medfilt1(ArrayRef<f64> x, usize k, bool zeropad, MutArrayRef<f64> out,
                      Allocator* alloc) {
    const usize n = x.size();
    if (out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (k == 0) {
        const f64 nan = Limits<f64>::quiet_nan();
        for (usize i = 0; i < n; ++i) {
            out[i] = nan;
        }
        return SignalStatus::kOk;
    }

    Array<f64> src_copy(alloc);
    ArrayRef<f64> src = x;
    if (x.data() == out.data()) {
        src_copy = Array<f64>::from(x, alloc);
        src = src_copy.as_ref();
    }

    const usize left_half = k / 2;
    const usize right_half = k - 1 - left_half;

    Array<f64> win(alloc);
    win.reserve(k);

    auto insert_sorted = [&](f64 val) {
        const usize idx = keel::upper_bound(win.as_ref(), val);
        win.push_back(0.0);
        if (idx + 1 < win.size()) {
            keel::mem_move(win.data() + idx + 1, win.data() + idx,
                           (win.size() - 1 - idx) * sizeof(f64));
        }
        win[idx] = val;
    };

    auto compute_median = [&]() noexcept -> f64 {
        const usize m = win.size();
        if (!zeropad) {
            if (m == 0) return 0.0;
            if ((m & usize{1}) != 0) return win[m / 2];
            return 0.5 * (win[m / 2 - 1] + win[m / 2]);
        }
        const usize nz = k - m;
        const usize n_neg = keel::lower_bound(win.as_ref(), 0.0);
        const usize n_zero_win = keel::upper_bound(win.as_ref(), 0.0) - n_neg;
        const usize zeros_tot = nz + n_zero_win;
        auto val_at = [&](usize r) noexcept -> f64 {
            if (r < n_neg) return win[r];
            if (r < n_neg + zeros_tot) return 0.0;
            return win[r - nz];
        };
        if ((k & usize{1}) != 0) return val_at(k / 2);
        return 0.5 * (val_at(k / 2 - 1) + val_at(k / 2));
    };

    usize hi = keel::min(n, right_half + 1);
    usize lo = 0;
    for (usize i = 0; i < hi; ++i) {
        insert_sorted(src[i]);
    }
    out[0] = compute_median();

    for (usize i = 1; i < n; ++i) {
        const usize new_lo = (i >= left_half) ? (i - left_half) : 0;
        const usize new_hi = keel::min(n, i + right_half + 1);
        for (usize j = hi; j < new_hi; ++j) {
            insert_sorted(src[j]);
        }
        hi = new_hi;
        for (usize j = lo; j < new_lo; ++j) {
            const usize pos = keel::lower_bound(win.as_ref(), src[j]);
            if (pos < win.size() && win[pos] == src[j]) {
                win.erase(pos);
            }
        }
        lo = new_lo;
        out[i] = compute_median();
    }
    return SignalStatus::kOk;
}

SignalStatus fillgaps(ArrayRef<f64> x, usize maxlen, usize order, MutArrayRef<f64> out,
                      Allocator* alloc) {
    return fillgaps_impl(x, maxlen, order, out, alloc);
}

SignalStatus fillgaps_c64(ArrayRef<c64> x, usize maxlen, usize order, MutArrayRef<c64> out,
                          Allocator* alloc) {
    return fillgaps_impl(x, maxlen, order, out, alloc);
}

SignalStatus sgolay(i32 order, i32 framelen, MutArrayRef<f64> b_rowmajor,
                    MutArrayRef<f64> g_rowmajor, ArrayRef<f64> weights, Allocator* alloc) {
    if (framelen <= 0 || (framelen & 1) == 0 || order < 0 || order >= framelen) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n = as<usize>(framelen);
    const usize p = as<usize>(order + 1);
    if (!b_rowmajor.empty() && b_rowmajor.size() < n * n) {
        return SignalStatus::kInvalidArgument;
    }
    if (!g_rowmajor.empty() && g_rowmajor.size() < n * p) {
        return SignalStatus::kInvalidArgument;
    }
    if (!weights.empty()) {
        if (weights.size() != n) {
            return SignalStatus::kInvalidArgument;
        }
        for (usize i = 0; i < n; ++i) {
            if (!(weights[i] > 0.0)) {
                return SignalStatus::kInvalidArgument;
            }
        }
    }

    const Array<f64> v = build_vandermonde(order, framelen, alloc);
    const bool has_w = !weights.empty();

    if (!b_rowmajor.empty()) {
        Array<f64> vtv(p * p, 0.0, alloc);
        Array<f64> x_rhs(p * n, 0.0, alloc);
        for (usize k = 0; k < p; ++k) {
            for (usize i = 0; i < n; ++i) {
                x_rhs[k * n + i] = v[i * p + k] * (has_w ? weights[i] : 1.0);
            }
        }
        for (usize i = 0; i < p; ++i) {
            for (usize j = 0; j < p; ++j) {
                f64 s = 0.0;
                for (usize t = 0; t < n; ++t) {
                    s += v[t * p + i] * (has_w ? weights[t] : 1.0) * v[t * p + j];
                }
                vtv[i * p + j] = s;
            }
        }
        const SignalStatus st = gauss_jordan(vtv.data(), x_rhs.data(), order + 1, framelen);
        if (st != SignalStatus::kOk) {
            return st;
        }
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                f64 s = 0.0;
                for (usize k = 0; k < p; ++k) {
                    s += v[i * p + k] * x_rhs[k * n + j];
                }
                b_rowmajor[i * n + j] = s;
            }
        }
    }

    if (!g_rowmajor.empty()) {
        Array<f64> vtv(p * p, 0.0, alloc);
        for (usize i = 0; i < p; ++i) {
            for (usize j = 0; j < p; ++j) {
                f64 s = 0.0;
                for (usize t = 0; t < n; ++t) {
                    s += v[t * p + i] * v[t * p + j];
                }
                vtv[i * p + j] = s;
            }
        }
        Array<f64> inv(p * p, 0.0, alloc);
        for (usize i = 0; i < p; ++i) {
            inv[i * p + i] = 1.0;
        }
        const SignalStatus st = gauss_jordan(vtv.data(), inv.data(), order + 1, order + 1);
        if (st != SignalStatus::kOk) {
            return st;
        }
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < p; ++j) {
                f64 s = 0.0;
                for (usize k = 0; k < p; ++k) {
                    s += v[i * p + k] * inv[k * p + j];
                }
                g_rowmajor[i * p + j] = s;
            }
        }
    }

    return SignalStatus::kOk;
}

SignalStatus sgolayfilt(ArrayRef<f64> x, i32 order, i32 framelen, MutArrayRef<f64> out,
                        ArrayRef<f64> weights, Allocator* alloc) {
    if (framelen <= 0 || (framelen & 1) == 0 || order < 0 || order >= framelen) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n = x.size();
    const usize fl = as<usize>(framelen);
    if (n < fl || out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> b(fl * fl, 0.0, alloc);
    const SignalStatus st = sgolay(order, framelen, b.as_mut_ref(), {}, weights, alloc);
    if (st != SignalStatus::kOk) {
        return st;
    }

    Array<f64> src_copy(alloc);
    ArrayRef<f64> src = x;
    if (x.data() == out.data()) {
        src_copy = Array<f64>::from(x, alloc);
        src = src_copy.as_ref();
    }

    const usize half = fl / 2;
    const f64* b_center = b.data() + half * fl;
    for (usize i = half; i < n - half; ++i) {
        f64 s = 0.0;
        for (usize k = 0; k < fl; ++k) {
            s += b_center[k] * src[i - half + k];
        }
        out[i] = s;
    }
    for (usize i = 0; i < half; ++i) {
        const f64* b_row = b.data() + i * fl;
        f64 s = 0.0;
        for (usize k = 0; k < fl; ++k) {
            s += b_row[k] * src[k];
        }
        out[i] = s;
    }
    for (usize i = n - half; i < n; ++i) {
        const usize row_idx = fl - 1 - (n - 1 - i);
        const f64* b_row = b.data() + row_idx * fl;
        f64 s = 0.0;
        for (usize k = 0; k < fl; ++k) {
            s += b_row[k] * src[n - fl + k];
        }
        out[i] = s;
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
