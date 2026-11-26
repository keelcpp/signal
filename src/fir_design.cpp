#include "keel/signal.h"

#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

bool solve_spd_or_lu(const f64* a_in, usize n, const f64* b_in, f64* x_out, Allocator* alloc) {
    Array<f64> L(n * n, 0.0, alloc);
    Array<f64> y(n, 0.0, alloc);

    bool chol_ok = true;
    for (usize i = 0; i < n && chol_ok; ++i) {
        for (usize j = 0; j <= i; ++j) {
            f64 s = a_in[i * n + j];
            for (usize k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s <= 1e-14) {
                    chol_ok = false;
                    break;
                }
                L[i * n + j] = sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    if (chol_ok) {
        for (usize i = 0; i < n; ++i) {
            f64 s = b_in[i];
            for (usize k = 0; k < i; ++k) s -= L[i * n + k] * y[k];
            y[i] = s / L[i * n + i];
        }
        for (usize ii = 0; ii < n; ++ii) {
            const usize i = n - 1 - ii;
            f64 s = y[i];
            for (usize k = i + 1; k < n; ++k) s -= L[k * n + i] * x_out[k];
            x_out[i] = s / L[i * n + i];
        }
        return true;
    }

    for (usize i = 0; i < n * n; ++i) L[i] = a_in[i];
    for (usize i = 0; i < n; ++i) y[i] = b_in[i];
    for (usize k = 0; k < n; ++k) {
        usize piv = k;
        f64 max_abs = abs(L[k * n + k]);
        for (usize i = k + 1; i < n; ++i) {
            const f64 v = abs(L[i * n + k]);
            if (v > max_abs) {
                max_abs = v;
                piv = i;
            }
        }
        if (piv != k) {
            for (usize j = k; j < n; ++j) keel::swap(L[k * n + j], L[piv * n + j]);
            keel::swap(y[k], y[piv]);
        }
        f64 diag = L[k * n + k];
        if (abs(diag) < 1e-15) diag = (diag >= 0.0) ? 1e-15 : -1e-15;
        for (usize i = k + 1; i < n; ++i) {
            const f64 factor = L[i * n + k] / diag;
            L[i * n + k] = 0.0;
            for (usize j = k + 1; j < n; ++j) L[i * n + j] -= factor * L[k * n + j];
            y[i] -= factor * y[k];
        }
    }
    for (usize ii = 0; ii < n; ++ii) {
        const usize i = n - 1 - ii;
        f64 s = y[i];
        for (usize j = i + 1; j < n; ++j) s -= L[i * n + j] * x_out[j];
        f64 diag = L[i * n + i];
        if (abs(diag) < 1e-15) diag = (diag >= 0.0) ? 1e-15 : -1e-15;
        x_out[i] = s / diag;
    }
    return true;
}

inline f64 ic_real(f64 alpha, f64 w) noexcept {
    return (abs(alpha) < 1e-15) ? w : (sin(alpha * w) / alpha);
}

inline f64 iclin_real(f64 alpha, f64 w, f64 w0) noexcept {
    if (abs(alpha) < 1e-15) {
        const f64 d = w - w0;
        return 0.5 * d * d;
    }
    return (w - w0) * sin(alpha * w) / alpha + cos(alpha * w) / (alpha * alpha);
}

inline f64 is_real(f64 alpha, f64 w) noexcept {
    return -cos(alpha * w) / alpha;
}

inline f64 islin_real(f64 alpha, f64 w, f64 w0) noexcept {
    return -(w - w0) * cos(alpha * w) / alpha + sin(alpha * w) / (alpha * alpha);
}

inline i32 next_pow2_i32(i32 v) noexcept {
    i32 p = 1;
    while (p < v && p <= (i32{1} << 29)) p <<= 1;
    return p;
}

inline f64 bary_eval(const f64* xk, const f64* yk, const f64* gk, usize m, f64 x) noexcept {
    for (usize k = 0; k < m; ++k) {
        if (abs(x - xk[k]) < 1e-15) return yk[k];
    }
    f64 num = 0.0;
    f64 den = 0.0;
    for (usize k = 0; k < m; ++k) {
        const f64 t = gk[k] / (x - xk[k]);
        num += t * yk[k];
        den += t;
    }
    return num / den;
}

struct PmGridPoint {
    f64 w;
    f64 d;
    f64 weight;
    i32 band;
};

} // namespace

SignalStatus fir1_single(i32 order, f64 wn, bool highpass, MutArrayRef<f64> out_b) noexcept {
    if (order < 1 || !is_finite(wn) || !(wn > 0.0 && wn < 1.0)) return SignalStatus::kInvalidArgument;
    if (highpass && ((order & 1) != 0)) return SignalStatus::kInvalidArgument;
    const usize filt_len = as<usize>(order + 1);
    if (out_b.size() != filt_len) return SignalStatus::kInvalidArgument;

    const f64 wc = kPi * wn;
    const f64 half = as<f64>(order) / 2.0;
    const f64 f0 = highpass ? 1.0 : 0.0;
    f64 gain = 0.0;
    for (usize i = 0; i < filt_len; ++i) {
        const f64 n = as<f64>(i) - half;
        const f64 lp = (abs(n) < 1e-12) ? (wc / kPi) : (sin(wc * n) / (kPi * n));
        const f64 ideal = highpass ? (((abs(n) < 1e-12) ? 1.0 : 0.0) - lp) : lp;
        const f64 win = 0.54 - 0.46 * cos(kTau * as<f64>(i) / as<f64>(order));
        out_b[i] = ideal * win;
        gain += out_b[i] * cos(kPi * f0 * n);
    }
    if (abs(gain) > 1e-15) {
        for (usize i = 0; i < filt_len; ++i) out_b[i] /= gain;
    }
    return SignalStatus::kOk;
}

SignalStatus fir1_multiband(i32 order, ArrayRef<f64> wn, bool first_band_pass, ArrayRef<f64> window,
                            bool scale, Array<f64>& out_b, Allocator* alloc) {
    const usize nw = wn.size();
    if (order < 1 || nw == 0) return SignalStatus::kInvalidArgument;
    for (usize k = 0; k < nw; ++k) {
        if (!(wn[k] > 0.0 && wn[k] < 1.0)) return SignalStatus::kInvalidArgument;
        if (k > 0 && !(wn[k] > wn[k - 1])) return SignalStatus::kInvalidArgument;
    }

    const bool pass_at_nyquist = ((nw % 2 == 1) ? !first_band_pass : first_band_pass);
    if (pass_at_nyquist && (order % 2 == 1) && window.empty()) {
        order += 1;
    }

    const usize filt_len = as<usize>(order + 1);
    if (!window.empty() && window.size() != filt_len) return SignalStatus::kInvalidArgument;

    Array<f64> edges(nw + 2, 0.0, alloc);
    edges[0] = 0.0;
    for (usize k = 0; k < nw; ++k) edges[k + 1] = wn[k];
    edges[nw + 1] = 1.0;

    const f64 half = as<f64>(order) / 2.0;
    out_b.resize(filt_len);
    for (usize i = 0; i < filt_len; ++i) out_b[i] = 0.0;

    for (usize b = 0; b <= nw; ++b) {
        const bool is_pass = (b % 2 == 0) ? first_band_pass : !first_band_pass;
        if (!is_pass) continue;
        const f64 f_lo = edges[b];
        const f64 f_hi = edges[b + 1];
        for (usize i = 0; i < filt_len; ++i) {
            const f64 n = as<f64>(i) - half;
            const f64 sr = (abs(n) < 1e-12) ? f_hi : (sin(kPi * f_hi * n) / (kPi * n));
            const f64 sl = (abs(n) < 1e-12) ? f_lo : (sin(kPi * f_lo * n) / (kPi * n));
            out_b[i] += (sr - sl);
        }
    }

    for (usize i = 0; i < filt_len; ++i) {
        const f64 w =
            window.empty()
                ? ((order == 0) ? 1.0 : (0.54 - 0.46 * cos(kTau * as<f64>(i) / as<f64>(order))))
                : window[i];
        out_b[i] *= w;
    }

    if (scale) {
        f64 f0 = 0.0;
        if (!first_band_pass) {
            f0 = (edges[2] == 1.0) ? 1.0 : (0.5 * (edges[1] + edges[2]));
        }
        f64 gain = 0.0;
        for (usize i = 0; i < filt_len; ++i) {
            gain += out_b[i] * cos(kPi * f0 * (as<f64>(i) - half));
        }
        if (abs(gain) > 1e-15) {
            for (usize i = 0; i < filt_len; ++i) out_b[i] /= gain;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus fir2(i32 order, ArrayRef<f64> f, ArrayRef<f64> m, i32 npt_in, i32 lap_in,
                  ArrayRef<f64> window, Array<f64>& out_b, Allocator* alloc) {
    const usize nf = f.size();
    if (order <= 0 || nf < 2 || m.size() != nf) return SignalStatus::kInvalidArgument;
    if (f[0] != 0.0 || f[nf - 1] != 1.0) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < nf; ++i) {
        if (!is_finite(f[i]) || !is_finite(m[i])) return SignalStatus::kInvalidArgument;
        if (i > 0 && f[i] < f[i - 1]) return SignalStatus::kInvalidArgument;
    }

    if ((order % 2) == 1 && m[nf - 1] != 0.0) order += 1;
    const i32 nn = order + 1;
    if (!window.empty() && as<i32>(window.size()) != nn) return SignalStatus::kInvalidArgument;

    const bool npt_default = (npt_in <= 0);
    i32 npt = npt_default ? 512 : npt_in;
    if (!npt_default) {
        if (2 * npt < nn) return SignalStatus::kInvalidArgument;
    } else if (2 * npt < nn) {
        npt = (nn + 1) / 2;
    }
    npt = next_pow2_i32(npt);
    const i32 lap = (lap_in > 0) ? lap_in : (npt / 25);

    const i32 npt1 = npt + 1;
    Array<c64> h_half(as<usize>(npt1), c64(0.0, 0.0), alloc);
    auto put_h = [&](i32 j1, f64 val) noexcept -> bool {
        if (j1 < 1 || j1 > npt1) return false;
        h_half[as<usize>(j1 - 1)] = c64(val, 0.0);
        return true;
    };

    if (!put_h(1, m[0])) return SignalStatus::kInvalidArgument;
    i32 nb = 1;
    for (usize i = 0; i + 1 < nf; ++i) {
        const f64 df = f[i + 1] - f[i];
        i32 ne = 0;
        if (df == 0.0) {
            nb = as<i32>(ceil(as<f64>(nb) - as<f64>(lap) / 2.0));
            ne = nb + lap;
        } else {
            ne = as<i32>(floor(f[i + 1] * as<f64>(npt1)));
        }
        if (nb < 1 || ne > npt1) return SignalStatus::kInvalidArgument;
        for (i32 j = nb; j <= ne; ++j) {
            const f64 inc = (ne == nb) ? 0.0 : (as<f64>(j - nb) / as<f64>(ne - nb));
            if (!put_h(j, inc * m[i + 1] + (1.0 - inc) * m[i])) {
                return SignalStatus::kInvalidArgument;
            }
        }
        nb = ne + 1;
    }

    const f64 dt = as<f64>(nn - 1) / 2.0;
    for (i32 k = 0; k <= npt; ++k) {
        const f64 phase = -kPi * dt * as<f64>(k) / as<f64>(npt);
        h_half[as<usize>(k)] *= c64(cos(phase), sin(phase));
    }

    const usize n_full = as<usize>(2 * npt);
    Array<c64> spec(n_full, c64(0.0, 0.0), alloc);
    for (i32 k = 0; k <= npt; ++k) spec[as<usize>(k)] = h_half[as<usize>(k)];
    for (i32 k = 1; k < npt; ++k) spec[as<usize>(npt + k)] = conj(h_half[as<usize>(npt - k)]);

    fft::ifft_1d_inplace(spec.as_mut_ref());

    out_b.resize(as<usize>(nn));
    for (i32 k = 0; k < nn; ++k) {
        const f64 w = window.empty()
                          ? ((nn == 1) ? 1.0
                                       : (0.54 - 0.46 * cos(kTau * as<f64>(k) / as<f64>(nn - 1))))
                          : window[as<usize>(k)];
        out_b[as<usize>(k)] = spec[as<usize>(k)].real() * w;
    }
    return SignalStatus::kOk;
}

SignalStatus firls(i32 order, ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> w, FirFtype ftype,
                   MutArrayRef<f64> out_b, Allocator* alloc) {
    const usize fn = f.size();
    if (order < 1 || fn == 0 || a.size() != fn || (fn % 2) != 0 ||
        out_b.size() != as<usize>(order + 1)) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < fn; ++i) {
        if (!is_finite(f[i]) || !is_finite(a[i]) || f[i] < 0.0 || f[i] > 1.0) {
            return SignalStatus::kInvalidArgument;
        }
        if (i > 0 && f[i] < f[i - 1]) return SignalStatus::kInvalidArgument;
    }
    const usize num_bands = fn / 2;
    if (!w.empty()) {
        if (w.size() != num_bands) return SignalStatus::kInvalidArgument;
        for (usize k = 0; k < num_bands; ++k) {
            if (!(w[k] > 0.0) || !is_finite(w[k])) return SignalStatus::kInvalidArgument;
        }
    }

    const bool anti_sym = (ftype != FirFtype::kStandard);
    const bool odd_n = (order % 2) != 0;
    const usize K = (!odd_n && !anti_sym) ? as<usize>(order / 2 + 1)
                  : (!odd_n && anti_sym)  ? as<usize>(order / 2)
                                          : as<usize>((order + 1) / 2);

    Array<f64> alpha(K, 0.0, alloc);
    for (usize i = 0; i < K; ++i) {
        if (!odd_n && !anti_sym) alpha[i] = as<f64>(i);
        else if (!odd_n && anti_sym) alpha[i] = as<f64>(i + 1);
        else alpha[i] = as<f64>(i) + 0.5;
    }

    Array<f64> Q(K * K, 0.0, alloc);
    Array<f64> bvec(K, 0.0, alloc);

    for (usize k = 0; k < num_bands; ++k) {
        const f64 w1 = kPi * f[2 * k];
        const f64 w2 = kPi * f[2 * k + 1];
        if (w2 <= w1) continue;
        const f64 wk = w.empty() ? 1.0 : w[k];
        const f64 d1 = a[2 * k];
        const f64 d2 = a[2 * k + 1];
        const f64 slope = (d2 - d1) / (w2 - w1);

        for (usize i = 0; i < K; ++i) {
            const f64 ai = alpha[i];
            for (usize j = 0; j < K; ++j) {
                const f64 aj = alpha[j];
                const f64 sij = ai + aj;
                const f64 dij = abs(ai - aj);
                const f64 i_plus = ic_real(sij, w2) - ic_real(sij, w1);
                const f64 i_minus = ic_real(dij, w2) - ic_real(dij, w1);
                if (!anti_sym) Q[i * K + j] += wk * (i_minus + i_plus) / kTau;
                else Q[i * K + j] += wk * (i_minus - i_plus) / kTau;
            }
            if (!anti_sym) {
                const f64 i_const = d1 * (ic_real(ai, w2) - ic_real(ai, w1)) / kPi;
                const f64 i_lin = slope * (iclin_real(ai, w2, w1) - iclin_real(ai, w1, w1)) / kPi;
                bvec[i] += wk * (i_const + i_lin);
            } else {
                const f64 i_const = d1 * (is_real(ai, w2) - is_real(ai, w1)) / kPi;
                const f64 i_lin = slope * (islin_real(ai, w2, w1) - islin_real(ai, w1, w1)) / kPi;
                bvec[i] += wk * (i_const + i_lin);
            }
        }
    }

    Array<f64> c(K, 0.0, alloc);
    if (!solve_spd_or_lu(Q.data(), K, bvec.data(), c.data(), alloc)) {
        return SignalStatus::kInvalidArgument;
    }

    if (!odd_n && !anti_sym) {
        const usize M = as<usize>(order / 2);
        out_b[M] = c[0];
        for (usize k = 1; k <= M; ++k) {
            const f64 v = 0.5 * c[k];
            out_b[M - k] = v;
            out_b[M + k] = v;
        }
    } else if (odd_n && !anti_sym) {
        const usize M = as<usize>((order - 1) / 2);
        for (usize k = 0; k <= M; ++k) {
            const f64 v = 0.5 * c[k];
            out_b[M - k] = v;
            out_b[M + 1 + k] = v;
        }
    } else if (!odd_n && anti_sym) {
        const usize M = as<usize>(order / 2);
        out_b[M] = 0.0;
        for (usize k = 0; k < M; ++k) {
            const f64 v = 0.5 * c[k];
            out_b[M - (k + 1)] = v;
            out_b[M + (k + 1)] = -v;
        }
    } else {
        const usize M = as<usize>((order - 1) / 2);
        for (usize k = 0; k <= M; ++k) {
            const f64 v = 0.5 * c[k];
            out_b[M - k] = v;
            out_b[M + 1 + k] = -v;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus firpm(i32 order, ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> w, FirFtype ftype,
                   MutArrayRef<f64> out_b, f64& out_err, Allocator* alloc) {
    const usize fn = f.size();
    if (order < 3 || fn == 0 || a.size() != fn || (fn % 2) != 0 ||
        out_b.size() != as<usize>(order + 1)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize num_bands = fn / 2;
    if (!w.empty() && w.size() != num_bands) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < fn; ++i) {
        if (!is_finite(f[i]) || !is_finite(a[i]) || f[i] < 0.0 || f[i] > 1.0) {
            return SignalStatus::kInvalidArgument;
        }
        if (i > 0 && f[i] < f[i - 1]) return SignalStatus::kInvalidArgument;
    }

    const bool is_diff = (ftype == FirFtype::kDifferentiator);
    const bool anti_sym = (ftype != FirFtype::kStandard);
    enum : i32 { kTypeI = 1, kTypeII = 2, kTypeIII = 3, kTypeIV = 4 };
    const i32 filter_type =
        !anti_sym ? ((order % 2 == 0) ? kTypeI : kTypeII) : ((order % 2 == 0) ? kTypeIII : kTypeIV);
    usize L = 0;
    switch (filter_type) {
        case kTypeI:   L = as<usize>(order) / 2; break;
        case kTypeII:  L = as<usize>(order - 1) / 2; break;
        case kTypeIII: L = as<usize>(order) / 2 - 1; break;
        case kTypeIV:  L = as<usize>(order - 1) / 2; break;
    }
    const usize r = L + 1;
    const usize n_extr = r + 1;
    constexpr i32 kLgrid = 16;

    f64 total_width = 0.0;
    for (usize k = 0; k < num_bands; ++k) total_width += f[2 * k + 1] - f[2 * k];
    if (total_width <= 0.0) return SignalStatus::kInvalidArgument;

    f64 df = 1.0 / (as<f64>(kLgrid) * as<f64>(r));
    if (total_width / df < as<f64>(n_extr + 4 * num_bands)) {
        df = total_width / as<f64>(kLgrid * as<i32>(r));
    }

    Array<PmGridPoint> grid(alloc);
    grid.reserve(as<usize>(total_width / df) + 4 * num_bands);

    for (usize k = 0; k < num_bands; ++k) {
        const f64 f1 = f[2 * k];
        const f64 f2 = f[2 * k + 1];
        const f64 bw = f2 - f1;
        if (bw <= 0.0) continue;
        const f64 a1 = a[2 * k];
        const f64 a2 = a[2 * k + 1];
        const f64 wt = w.empty() ? 1.0 : w[k];
        if (!is_finite(wt) || !(wt > 0.0)) return SignalStatus::kInvalidArgument;

        Array<f64> f_band(alloc);
        f64 f_cur =
            ((filter_type == kTypeIII || filter_type == kTypeIV) && k == 0 && f1 == 0.0) ? df : f1;
        while (f_cur <= f2 + 1e-15) {
            f_band.push_back(f_cur);
            f_cur += df;
        }
        if (!f_band.empty()) f_band.back() = f2;
        else f_band.push_back(f2);

        if ((filter_type == kTypeII || filter_type == kTypeIII) && !f_band.empty() &&
            f_band.back() > 1.0 - df + 1e-15) {
            f_band.pop_back();
        }

        const bool diff_active = is_diff && abs(a2) >= 1e-4;
        for (usize i = 0; i < f_band.size(); ++i) {
            const f64 f_norm = f_band[i];
            const f64 t = (bw > 0.0) ? ((f_norm - f1) / bw) : 0.0;
            PmGridPoint pt{};
            pt.w = kPi * f_norm;
            const f64 d_raw = a1 + t * (a2 - a1);
            f64 q_val = 1.0;
            switch (filter_type) {
                case kTypeI:
                    q_val = 1.0;
                    break;
                case kTypeII:
                    if (pt.w > kPi * 0.99999) continue;
                    q_val = cos(0.5 * pt.w);
                    break;
                case kTypeIII:
                    if (pt.w < kPi * 1e-5 || pt.w > kPi * 0.99999) continue;
                    q_val = sin(pt.w);
                    break;
                case kTypeIV:
                    if (pt.w < kPi * 1e-5) continue;
                    q_val = sin(0.5 * pt.w);
                    break;
            }
            f64 w_eff = wt;
            if (diff_active) {
                if (f_norm < 1e-12) continue;
                w_eff = wt / (0.5 * f_norm);
            }
            if (filter_type != kTypeI) {
                pt.d = d_raw / q_val;
                pt.weight = w_eff * q_val;
            } else {
                pt.d = d_raw;
                pt.weight = w_eff;
            }
            pt.band = as<i32>(k);
            grid.push_back(pt);
        }
    }
    if (grid.size() < n_extr + 1) return SignalStatus::kInvalidArgument;

    Array<usize> extr(n_extr, usize{0}, alloc);
    {
        Array<usize> band_start(num_bands, usize{0}, alloc);
        Array<usize> band_end(num_bands, usize{0}, alloc);
        i32 cur_band = -1;
        for (usize i = 0; i < grid.size(); ++i) {
            const i32 b = grid[i].band;
            if (b != cur_band) {
                band_start[as<usize>(b)] = i;
                if (cur_band >= 0) band_end[as<usize>(cur_band)] = i - 1;
                cur_band = b;
            }
        }
        band_end[as<usize>(cur_band)] = grid.size() - 1;

        Array<usize> per_band(num_bands, usize{0}, alloc);
        usize assigned = 0;
        for (usize k = 0; k < num_bands; ++k) {
            const f64 bw = f[2 * k + 1] - f[2 * k];
            const usize nb = max<usize>(2, as<usize>(round(as<f64>(n_extr) * bw / total_width)));
            per_band[k] = nb;
            assigned += nb;
        }
        while (assigned > n_extr) {
            usize kmax = 0;
            for (usize k = 1; k < num_bands; ++k) {
                if (per_band[k] > per_band[kmax]) kmax = k;
            }
            if (per_band[kmax] <= 2) break;
            per_band[kmax]--;
            assigned--;
        }
        while (assigned < n_extr) {
            usize kmax = 0;
            for (usize k = 1; k < num_bands; ++k) {
                if (per_band[k] > per_band[kmax]) kmax = k;
            }
            per_band[kmax]++;
            assigned++;
        }
        usize out_i = 0;
        for (usize k = 0; k < num_bands; ++k) {
            const usize nb = per_band[k];
            const usize s = band_start[k];
            const usize e = band_end[k];
            for (usize i = 0; i < nb && out_i < n_extr; ++i) {
                extr[out_i++] = (nb == 1) ? s : (s + ((e - s) * i) / (nb - 1));
            }
        }
    }

    Array<f64> xk(n_extr, 0.0, alloc);
    Array<f64> gammak(n_extr, 0.0, alloc);
    Array<f64> ck(n_extr, 0.0, alloc);
    Array<f64> xkp(r, 0.0, alloc);
    Array<f64> ckp(r, 0.0, alloc);
    Array<f64> gkp(r, 0.0, alloc);
    Array<f64> err_grid(grid.size(), 0.0, alloc);
    Array<usize> new_extr(alloc);
    new_extr.reserve(n_extr + 8);
    Array<usize> peaks(alloc);
    peaks.reserve(grid.size() / 2 + 8);

    f64 delta = 0.0;
    constexpr i32 kMaxIter = 50;
    constexpr f64 kConvTol = 1e-9;

    auto compute_poly = [&]() noexcept {
        for (usize k = 0; k < n_extr; ++k) xk[k] = cos(grid[extr[k]].w);
        for (usize k = 0; k < n_extr; ++k) {
            f64 prod = 1.0;
            for (usize j = 0; j < n_extr; ++j) {
                if (j != k) prod *= (xk[k] - xk[j]);
            }
            gammak[k] = 1.0 / prod;
        }
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize k = 0; k < n_extr; ++k) {
            num += gammak[k] * grid[extr[k]].d;
            const f64 sg = (k % 2 == 0) ? 1.0 : -1.0;
            den += sg * gammak[k] / grid[extr[k]].weight;
        }
        delta = num / den;
        for (usize k = 0; k < n_extr; ++k) {
            const f64 sg = (k % 2 == 0) ? 1.0 : -1.0;
            ck[k] = grid[extr[k]].d - sg * delta / grid[extr[k]].weight;
        }
        for (usize k = 0; k < r; ++k) {
            xkp[k] = xk[k];
            ckp[k] = ck[k];
        }
        for (usize k = 0; k < r; ++k) {
            f64 prod = 1.0;
            for (usize j = 0; j < r; ++j) {
                if (j != k) prod *= (xkp[k] - xkp[j]);
            }
            gkp[k] = 1.0 / prod;
        }
    };

    for (i32 iter = 0; iter < kMaxIter; ++iter) {
        compute_poly();

        for (usize i = 0; i < grid.size(); ++i) {
            i32 extr_idx = -1;
            for (usize k = 0; k < n_extr; ++k) {
                if (extr[k] == i) {
                    extr_idx = as<i32>(k);
                    break;
                }
            }
            if (extr_idx >= 0) {
                const f64 sg = (extr_idx % 2 == 0) ? 1.0 : -1.0;
                err_grid[i] = sg * delta;
                continue;
            }
            const f64 xi = cos(grid[i].w);
            const f64 hi = bary_eval(xkp.data(), ckp.data(), gkp.data(), r, xi);
            err_grid[i] = grid[i].weight * (grid[i].d - hi);
        }

        new_extr.clear();
        peaks.clear();
        for (usize i = 0; i < grid.size(); ++i) {
            const f64 abs_e = abs(err_grid[i]);
            const bool same_l = (i > 0) && (grid[i - 1].band == grid[i].band);
            const bool same_r = (i + 1 < grid.size()) && (grid[i + 1].band == grid[i].band);
            const f64 abs_l = same_l ? abs(err_grid[i - 1]) : -1.0;
            const f64 abs_r = same_r ? abs(err_grid[i + 1]) : -1.0;
            const bool is_peak = (!same_l || !same_r) ? (abs_e > 0.0) : ((abs_e > abs_l) && (abs_e >= abs_r));
            if (is_peak) peaks.push_back(i);
        }
        for (usize pi = 0; pi < peaks.size(); ++pi) {
            const usize idx = peaks[pi];
            if (new_extr.empty()) {
                new_extr.push_back(idx);
                continue;
            }
            const f64 e_last = err_grid[new_extr.back()];
            const f64 e_cur = err_grid[idx];
            if ((e_cur >= 0.0) == (e_last >= 0.0)) {
                if (abs(e_cur) > abs(e_last)) new_extr.back() = idx;
            } else {
                new_extr.push_back(idx);
            }
        }
        while (new_extr.size() > n_extr) {
            if (abs(err_grid[new_extr.front()]) < abs(err_grid[new_extr.back()])) {
                new_extr.erase(0);
            } else {
                new_extr.pop_back();
            }
        }
        if (new_extr.size() < n_extr) continue;

        f64 max_e = 0.0;
        for (usize i = 0; i < grid.size(); ++i) {
            const f64 v = abs(err_grid[i]);
            if (v > max_e) max_e = v;
        }
        const f64 d_abs = abs(delta);
        const bool converged = (max_e - d_abs) < kConvTol * (d_abs > 0.0 ? d_abs : 1.0);
        for (usize k = 0; k < n_extr; ++k) extr[k] = new_extr[k];
        if (converged) break;
    }

    compute_poly();

    Array<f64> hs(L + 1, 0.0, alloc);
    for (usize m = 0; m <= L; ++m) {
        const f64 wm = kPi * as<f64>(m) / as<f64>(L);
        hs[m] = bary_eval(xkp.data(), ckp.data(), gkp.data(), r, cos(wm));
    }

    Array<f64> acoef(L + 1, 0.0, alloc);
    for (usize k = 0; k <= L; ++k) {
        f64 sum = 0.5 * hs[0] + 0.5 * hs[L] * ((k % 2 == 0) ? 1.0 : -1.0);
        for (usize m = 1; m < L; ++m) {
            sum += hs[m] * cos(kPi * as<f64>(k) * as<f64>(m) / as<f64>(L));
        }
        const f64 ck_scale = (k == 0 || k == L) ? 2.0 : 1.0;
        acoef[k] = (2.0 / (as<f64>(L) * ck_scale)) * sum;
    }

    for (usize i = 0; i <= as<usize>(order); ++i) out_b[i] = 0.0;

    if (filter_type == kTypeI) {
        out_b[L] = acoef[0];
        for (usize k = 1; k <= L; ++k) {
            const f64 v = 0.5 * acoef[k];
            out_b[L - k] = v;
            out_b[L + k] = v;
        }
    } else if (filter_type == kTypeII) {
        Array<f64> b(L + 1, 0.0, alloc);
        b[0] = acoef[0] + (L >= 1 ? 0.5 * acoef[1] : 0.0);
        for (usize n = 1; n + 1 <= L; ++n) b[n] = 0.5 * (acoef[n] + acoef[n + 1]);
        if (L >= 1) b[L] = 0.5 * acoef[L];
        for (usize n = 0; n <= L; ++n) {
            const f64 v = 0.5 * b[n];
            out_b[L - n] = v;
            out_b[L + 1 + n] = v;
        }
    } else if (filter_type == kTypeIII) {
        const usize lh = L + 1;
        Array<f64> c(lh + 1, 0.0, alloc);
        c[1] = acoef[0] - (L >= 2 ? 0.5 * acoef[2] : 0.0);
        for (usize n = 2; n <= lh; ++n) {
            const f64 am1 = (n - 1 <= L) ? acoef[n - 1] : 0.0;
            const f64 ap1 = (n + 1 <= L) ? acoef[n + 1] : 0.0;
            c[n] = 0.5 * am1 - 0.5 * ap1;
        }
        out_b[lh] = 0.0;
        for (usize n = 1; n <= lh; ++n) {
            const f64 v = 0.5 * c[n];
            out_b[lh - n] = -v;
            out_b[lh + n] = v;
        }
    } else {
        Array<f64> dvec(L + 2, 0.0, alloc);
        dvec[1] = (L >= 1) ? (acoef[0] - 0.5 * acoef[1]) : acoef[0];
        for (usize n = 2; n <= L; ++n) dvec[n] = 0.5 * (acoef[n - 1] - acoef[n]);
        dvec[L + 1] = (L >= 1) ? (0.5 * acoef[L]) : 0.0;
        for (usize n = 1; n <= L + 1; ++n) {
            const f64 v = 0.5 * dvec[n];
            const usize lo = L + 1 - n;
            out_b[lo] = -v;
            out_b[as<usize>(order) - lo] = v;
        }
    }

    if (is_diff) {
        for (usize i = 0; i <= as<usize>(order); ++i) out_b[i] = -out_b[i];
    }

    out_err = abs(delta);
    return SignalStatus::kOk;
}

} // namespace keel::signal
