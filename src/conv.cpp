#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

constexpr usize conv_next_pow2(usize n) noexcept {
    if (n <= 1) return 1;
    usize p = 1;
    while (p < n) p <<= 1;
    return p;
}

bool should_use_fft_conv(usize na, usize nb) noexcept {
    if (na == 0 || nb == 0) return false;
    const f64 direct = as<f64>(na) * as<f64>(nb);
    const usize nfft = conv_next_pow2(na + nb - 1);
    const f64 fn = as<f64>(nfft < 2 ? 2 : nfft);
    const f64 fft_cost = 6.0 * as<f64>(nfft) * keel::log2(fn);
    return direct > fft_cost;
}

} // namespace

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
    if (a.size() / n < m || b.size() / q < p) {
        return SignalStatus::kInvalidArgument;
    }
    const usize out_r = m + p - 1;
    const usize out_c = n + q - 1;
    if (out.size() / out_c < out_r) {
        return SignalStatus::kInvalidArgument;
    }
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
    return SignalStatus::kOk;
}

SignalStatus conv_1d_auto(ArrayRef<f64> a, ArrayRef<f64> b, MutArrayRef<f64> out,
                          Allocator* alloc) {
    const usize na = a.size();
    const usize nb = b.size();
    if (na == 0 || nb == 0) return SignalStatus::kOk;
    const usize nc = na + nb - 1;
    if (out.size() < nc) return SignalStatus::kInvalidArgument;
    if (!should_use_fft_conv(na, nb)) {
        return conv_1d(a, b, out);
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize nfft = conv_next_pow2(nc);
    const usize nhalf = nfft / 2 + 1;
    Array<f64> ap(nfft, 0.0, scratch.allocator());
    Array<f64> bp(nfft, 0.0, scratch.allocator());
    for (usize i = 0; i < na; ++i) ap[i] = a[i];
    for (usize i = 0; i < nb; ++i) bp[i] = b[i];
    Array<c64> af(nhalf, c64(0.0, 0.0), scratch.allocator());
    Array<c64> bf(nhalf, c64(0.0, 0.0), scratch.allocator());
    keel::fft::rfft_1d(ap.as_ref(), af.as_mut_ref());
    keel::fft::rfft_1d(bp.as_ref(), bf.as_mut_ref());
    for (usize k = 0; k < nhalf; ++k) af[k] *= bf[k];
    keel::fft::irfft_1d(af.as_ref(), ap.as_mut_ref(), nfft);
    for (usize i = 0; i < nc; ++i) out[i] = ap[i];
    return SignalStatus::kOk;
}

SignalStatus cconv(ArrayRef<f64> x, ArrayRef<f64> y, usize n, MutArrayRef<f64> out,
                   Allocator* alloc) {
    const usize nx = x.size();
    const usize ny = y.size();
    if (n == 0) {
        n = (nx == 0 || ny == 0) ? 0 : (nx + ny - 1);
    }
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    if (n == 0) return SignalStatus::kOk;
    for (usize k = 0; k < n; ++k) out[k] = 0.0;

    if (nx == 0 || ny == 0) {
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    if (n >= nx + ny - 1) {
        const usize nfft = conv_next_pow2(n);
        const usize nhalf = nfft / 2 + 1;
        Array<f64> xp(nfft, 0.0, scratch.allocator());
        Array<f64> yp(nfft, 0.0, scratch.allocator());
        for (usize i = 0; i < nx; ++i) xp[i] = x[i];
        for (usize i = 0; i < ny; ++i) yp[i] = y[i];
        Array<c64> xf(nhalf, c64(0.0, 0.0), scratch.allocator());
        Array<c64> yf(nhalf, c64(0.0, 0.0), scratch.allocator());
        keel::fft::rfft_1d(xp.as_ref(), xf.as_mut_ref());
        keel::fft::rfft_1d(yp.as_ref(), yf.as_mut_ref());
        for (usize k = 0; k < nhalf; ++k) xf[k] *= yf[k];
        keel::fft::irfft_1d(xf.as_ref(), xp.as_mut_ref(), nfft);
        for (usize k = 0; k < n; ++k) out[k] = xp[k];
        return SignalStatus::kOk;
    }

    // Fold x and y modulo n so circular convolution matches MATLAB cconv(x, y, n)
    // even when nx > n or ny > n.
    Array<f64> xw(n, 0.0, scratch.allocator());
    Array<f64> yw(n, 0.0, scratch.allocator());
    for (usize i = 0; i < nx; ++i) xw[i % n] += x[i];
    for (usize j = 0; j < ny; ++j) yw[j % n] += y[j];

    if (n >= 64) {
        Array<c64> xf(n, c64(0.0, 0.0), scratch.allocator());
        Array<c64> yf(n, c64(0.0, 0.0), scratch.allocator());
        keel::fft::fft_1d_real(xw.as_ref(), xf.as_mut_ref());
        keel::fft::fft_1d_real(yw.as_ref(), yf.as_mut_ref());
        for (usize k = 0; k < n; ++k) xf[k] *= yf[k];
        keel::fft::ifft_1d_inplace(xf.as_mut_ref());
        for (usize k = 0; k < n; ++k) out[k] = xf[k].real();
        return SignalStatus::kOk;
    }

    for (usize k = 0; k < n; ++k) {
        f64 s = 0.0;
        for (usize i = 0; i < n; ++i) {
            const usize j = (k + n - i) % n;
            s += xw[i] * yw[j];
        }
        out[k] = s;
    }
    return SignalStatus::kOk;
}

SignalStatus deconv(ArrayRef<f64> b, ArrayRef<f64> a, Array<f64>& q_out, Array<f64>& r_out,
                    Allocator* alloc) {
    const usize nb = b.size();
    const usize na = a.size();
    if (na == 0 || a[0] == 0.0) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (na > nb) {
        q_out = Array<f64>(1, 0.0, alloc);
        r_out = Array<f64>(b, alloc);
        return SignalStatus::kOk;
    }
    const usize nq = nb - na + 1;
    q_out = Array<f64>(nq, 0.0, alloc);
    r_out = Array<f64>(b, alloc);
    const f64 a0 = a[0];
    for (usize i = 0; i < nq; ++i) {
        const f64 qi = r_out[i] / a0;
        q_out[i] = qi;
        for (usize j = 0; j < na; ++j) {
            r_out[i + j] -= qi * a[j];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus xcorr(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> r_out,
                   MutArrayRef<f64> lags_out, Allocator* alloc) {
    const usize nx = x.size();
    const usize ny = y.size();
    const usize max_len = keel::max(nx, ny);
    const usize nc = (max_len > 0) ? (2 * max_len - 1) : 0;
    if (r_out.size() < nc || (!lags_out.empty() && lags_out.size() < nc)) {
        return SignalStatus::kInvalidArgument;
    }
    if (nc == 0) return SignalStatus::kOk;

    const i64 max_lag = as<i64>(max_len) - 1;
    ArenaTemp scratch = keel::get_scratch(alloc);
    if (should_use_fft_conv(max_len, max_len)) {
        const usize nfft = conv_next_pow2(nc);
        const usize nhalf = nfft / 2 + 1;
        Array<f64> xp(nfft, 0.0, scratch.allocator());
        Array<f64> yp(nfft, 0.0, scratch.allocator());
        for (usize i = 0; i < nx; ++i) xp[i] = x[i];
        for (usize i = 0; i < ny; ++i) yp[i] = y[i];
        Array<c64> xf(nhalf, c64(0.0, 0.0), scratch.allocator());
        Array<c64> yf(nhalf, c64(0.0, 0.0), scratch.allocator());
        keel::fft::rfft_1d(xp.as_ref(), xf.as_mut_ref());
        keel::fft::rfft_1d(yp.as_ref(), yf.as_mut_ref());
        for (usize k = 0; k < nhalf; ++k) xf[k] *= keel::conj(yf[k]);
        keel::fft::irfft_1d(xf.as_ref(), xp.as_mut_ref(), nfft);
        const usize neg_lags = as<usize>(max_lag);
        for (usize i = 0; i < neg_lags; ++i) r_out[i] = xp[nfft - neg_lags + i];
        for (usize i = 0; i <= neg_lags; ++i) r_out[neg_lags + i] = xp[i];
    } else {
        Array<f64> x_pad(max_len, 0.0, scratch.allocator());
        Array<f64> y_rev(max_len, 0.0, scratch.allocator());
        for (usize i = 0; i < nx; ++i) x_pad[i] = x[i];
        for (usize i = 0; i < max_len; ++i) {
            const usize y_idx = max_len - 1 - i;
            y_rev[i] = (y_idx < ny) ? y[y_idx] : 0.0;
        }
        const SignalStatus st = conv_1d(x_pad.as_ref(), y_rev.as_ref(), r_out.slice(0, nc));
        if (st != SignalStatus::kOk) return st;
    }

    if (!lags_out.empty()) {
        for (usize i = 0; i < nc; ++i) {
            lags_out[i] = as<f64>(as<i64>(i) - max_lag);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus xcorr_c64(ArrayRef<c64> x, ArrayRef<c64> y, MutArrayRef<c64> r_out,
                       MutArrayRef<f64> lags_out, Allocator* alloc) {
    const usize nx = x.size();
    const usize ny = y.size();
    const usize max_len = keel::max(nx, ny);
    const usize nc = (max_len > 0) ? (2 * max_len - 1) : 0;
    if (r_out.size() < nc || (!lags_out.empty() && lags_out.size() < nc)) {
        return SignalStatus::kInvalidArgument;
    }
    if (nc == 0) return SignalStatus::kOk;

    const i64 max_lag = as<i64>(max_len) - 1;
    if (should_use_fft_conv(max_len, max_len)) {
        ArenaTemp scratch = keel::get_scratch(alloc);
        const usize nfft = conv_next_pow2(nc);
        Array<c64> fx(nfft, c64(0.0, 0.0), scratch.allocator());
        Array<c64> fy(nfft, c64(0.0, 0.0), scratch.allocator());
        for (usize i = 0; i < nx; ++i) fx[i] = x[i];
        for (usize i = 0; i < ny; ++i) fy[i] = y[i];
        keel::fft::fft_1d_inplace(fx.as_mut_ref());
        keel::fft::fft_1d_inplace(fy.as_mut_ref());
        for (usize k = 0; k < nfft; ++k) fx[k] *= keel::conj(fy[k]);
        keel::fft::ifft_1d_inplace(fx.as_mut_ref());
        const usize neg_lags = as<usize>(max_lag);
        for (usize i = 0; i < neg_lags; ++i) r_out[i] = fx[nfft - neg_lags + i];
        for (usize i = 0; i <= neg_lags; ++i) r_out[neg_lags + i] = fx[i];
    } else {
        for (i64 m = -max_lag; m <= max_lag; ++m) {
            c64 acc(0.0, 0.0);
            for (usize n = 0; n < max_len; ++n) {
                const i64 idx_x = as<i64>(n) + m;
                if (idx_x >= 0 && as<usize>(idx_x) < nx && n < ny) {
                    acc += x[as<usize>(idx_x)] * keel::conj(y[n]);
                }
            }
            r_out[as<usize>(m + max_lag)] = acc;
        }
    }
    if (!lags_out.empty()) {
        for (usize i = 0; i < nc; ++i) {
            lags_out[i] = as<f64>(as<i64>(i) - max_lag);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus xcov(ArrayRef<f64> x, ArrayRef<f64> y, i32 maxlag, XcorrScale scale,
                  Array<f64>& c_out, Array<f64>& lags_out, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize nx = x.size();
    const usize ny = y.size();
    Array<f64> xc(nx, 0.0, scratch.allocator());
    Array<f64> yc(ny, 0.0, scratch.allocator());
    if (nx > 0) {
        f64 s = 0.0;
        for (usize i = 0; i < nx; ++i) s += x[i];
        const f64 m = s / as<f64>(nx);
        for (usize i = 0; i < nx; ++i) xc[i] = x[i] - m;
    }
    if (ny > 0) {
        f64 s = 0.0;
        for (usize i = 0; i < ny; ++i) s += y[i];
        const f64 m = s / as<f64>(ny);
        for (usize i = 0; i < ny; ++i) yc[i] = y[i] - m;
    }

    const usize n_max = keel::max(nx, ny);
    const usize nc = (n_max > 0) ? (2 * n_max - 1) : 0;
    Array<f64> cfull(nc, 0.0, scratch.allocator());
    Array<f64> lfull(nc, 0.0, scratch.allocator());
    const SignalStatus st =
        xcorr(xc.as_ref(), yc.as_ref(), cfull.as_mut_ref(), lfull.as_mut_ref(), scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    if (scale == XcorrScale::kBiased) {
        const f64 inv = (n_max > 0) ? (1.0 / as<f64>(n_max)) : 0.0;
        for (usize i = 0; i < nc; ++i) cfull[i] *= inv;
    } else if (scale == XcorrScale::kUnbiased) {
        for (usize i = 0; i < nc; ++i) {
            const f64 div = as<f64>(n_max) - keel::abs(lfull[i]);
            cfull[i] = (div > 0.0) ? (cfull[i] / div) : 0.0;
        }
    } else if (scale == XcorrScale::kCoeff) {
        f64 c0x = 0.0;
        f64 c0y = 0.0;
        for (usize i = 0; i < nx; ++i) c0x += xc[i] * xc[i];
        for (usize i = 0; i < ny; ++i) c0y += yc[i] * yc[i];
        const f64 denom = keel::sqrt(c0x * c0y);
        if (denom > 0.0) {
            for (usize i = 0; i < nc; ++i) cfull[i] /= denom;
        }
    }

    const i32 full_max_lag = (n_max > 0) ? (as<i32>(n_max) - 1) : 0;
    const i32 use_maxlag = (maxlag < 0) ? full_max_lag : maxlag;
    if (use_maxlag == full_max_lag && nc > 0) {
        c_out = Array<f64>(cfull.as_ref(), alloc);
        lags_out = Array<f64>(lfull.as_ref(), alloc);
        return SignalStatus::kOk;
    }
    const usize out_len = as<usize>(2 * use_maxlag + 1);
    c_out = Array<f64>(out_len, 0.0, alloc);
    lags_out = Array<f64>(out_len, 0.0, alloc);
    for (i32 m = -use_maxlag; m <= use_maxlag; ++m) {
        const i32 src = full_max_lag + m;
        const usize dst = as<usize>(m + use_maxlag);
        lags_out[dst] = as<f64>(m);
        c_out[dst] = (src >= 0 && as<usize>(src) < nc) ? cfull[as<usize>(src)] : 0.0;
    }
    return SignalStatus::kOk;
}

SignalStatus xcorr2(ArrayRef<f64> a, usize ra, usize ca, ArrayRef<f64> b, usize rb, usize cb,
                    MutArrayRef<f64> out, Allocator* alloc) {
    if (ra == 0 || ca == 0 || rb == 0 || cb == 0) return SignalStatus::kOk;
    if (a.size() / ca < ra || b.size() / cb < rb) {
        return SignalStatus::kInvalidArgument;
    }
    const usize r_out = ra + rb - 1;
    const usize c_out = ca + cb - 1;
    if (out.size() / c_out < r_out) {
        return SignalStatus::kInvalidArgument;
    }
    const usize cells_out = r_out * c_out;

    const f64 cells_direct = as<f64>(ra) * as<f64>(ca) * as<f64>(rb) * as<f64>(cb);
    if (cells_direct > 8192.0 && cells_out >= 64) {
        // 2-D FFT cross-correlation: pad A and flipped B to r_out x c_out, 2-D FFT, multiply, IFFT.
        ArenaTemp scratch = keel::get_scratch(alloc);
        Array<c64> fa(cells_out, c64(0.0, 0.0), scratch.allocator());
        Array<c64> fb(cells_out, c64(0.0, 0.0), scratch.allocator());
        for (usize j = 0; j < ca; ++j) {
            for (usize i = 0; i < ra; ++i) {
                fa[j * r_out + i] = c64(a[j * ra + i], 0.0);
            }
        }
        for (usize j = 0; j < cb; ++j) {
            for (usize i = 0; i < rb; ++i) {
                fb[j * r_out + i] = c64(b[(cb - 1 - j) * rb + (rb - 1 - i)], 0.0);
            }
        }
        const usize dims[2] = {r_out, c_out};
        const keel::fft::FftBatchLayout col_layout =
            keel::fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 0);
        const keel::fft::FftBatchLayout row_layout =
            keel::fft::FftBatchLayout::from_col_major(ArrayRef<usize>(dims, 2), 1);
        Array<c64> tmp(cells_out, c64(0.0, 0.0), scratch.allocator());
        keel::fft::fft_batch(fa.as_ref(), tmp.as_mut_ref(), col_layout);
        keel::fft::fft_batch(tmp.as_ref(), fa.as_mut_ref(), row_layout);
        keel::fft::fft_batch(fb.as_ref(), tmp.as_mut_ref(), col_layout);
        keel::fft::fft_batch(tmp.as_ref(), fb.as_mut_ref(), row_layout);
        for (usize k = 0; k < cells_out; ++k) fa[k] *= fb[k];
        keel::fft::ifft_batch(fa.as_ref(), tmp.as_mut_ref(), row_layout);
        keel::fft::ifft_batch(tmp.as_ref(), fa.as_mut_ref(), col_layout);
        for (usize k = 0; k < cells_out; ++k) out[k] = fa[k].real();
        return SignalStatus::kOk;
    }

    for (usize k = 0; k < cells_out; ++k) out[k] = 0.0;
    for (usize i = 0; i < ra; ++i) {
        for (usize j = 0; j < ca; ++j) {
            const f64 av = a[j * ra + i];
            if (av == 0.0) continue;
            for (usize p = 0; p < rb; ++p) {
                const i64 kr = as<i64>(i) - as<i64>(p) + as<i64>(rb) - 1;
                if (kr < 0 || kr >= as<i64>(r_out)) continue;
                for (usize q = 0; q < cb; ++q) {
                    const i64 kc = as<i64>(j) - as<i64>(q) + as<i64>(cb) - 1;
                    if (kc < 0 || kc >= as<i64>(c_out)) continue;
                    out[as<usize>(kc) * r_out + as<usize>(kr)] += av * b[q * rb + p];
                }
            }
        }
    }
    return SignalStatus::kOk;
}

SignalStatus xcorr2_c64(ArrayRef<c64> a, usize ra, usize ca, ArrayRef<c64> b, usize rb, usize cb,
                        MutArrayRef<c64> out) noexcept {
    if (ra == 0 || ca == 0 || rb == 0 || cb == 0) return SignalStatus::kOk;
    if (a.size() / ca < ra || b.size() / cb < rb) {
        return SignalStatus::kInvalidArgument;
    }
    const usize r_out = ra + rb - 1;
    const usize c_out = ca + cb - 1;
    if (out.size() / c_out < r_out) {
        return SignalStatus::kInvalidArgument;
    }
    const usize cells_out = r_out * c_out;
    for (usize k = 0; k < cells_out; ++k) out[k] = c64(0.0, 0.0);
    for (usize i = 0; i < ra; ++i) {
        for (usize j = 0; j < ca; ++j) {
            const c64 av = a[j * ra + i];
            if (av == c64(0.0, 0.0)) continue;
            for (usize p = 0; p < rb; ++p) {
                const i64 kr = as<i64>(i) - as<i64>(p) + as<i64>(rb) - 1;
                if (kr < 0 || kr >= as<i64>(r_out)) continue;
                for (usize q = 0; q < cb; ++q) {
                    const i64 kc = as<i64>(j) - as<i64>(q) + as<i64>(cb) - 1;
                    if (kc < 0 || kc >= as<i64>(c_out)) continue;
                    out[as<usize>(kc) * r_out + as<usize>(kr)] += av * keel::conj(b[q * rb + p]);
                }
            }
        }
    }
    return SignalStatus::kOk;
}

SignalStatus finddelay(ArrayRef<f64> x, ArrayRef<f64> y, i64 max_lag, i64& out_delay,
                       Allocator* alloc) {
    const usize max_len = keel::max(x.size(), y.size());
    if (max_len == 0) {
        out_delay = 0;
        return SignalStatus::kOk;
    }
    const usize nc = 2 * max_len - 1;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> c(nc, 0.0, scratch.allocator());
    Array<f64> lags(nc, 0.0, scratch.allocator());
    const SignalStatus st = xcorr(x, y, c.as_mut_ref(), lags.as_mut_ref(), scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    i64 best_lag = 0;
    f64 best_mag = -1.0;
    for (usize i = 0; i < nc; ++i) {
        const i64 lag = as<i64>(lags[i]);
        if (max_lag > 0 && keel::abs(lag) > max_lag) continue;
        const f64 mag = keel::abs(c[i]);
        if (keel::is_finite(mag) && mag > best_mag) {
            best_mag = mag;
            best_lag = lag;
        }
    }
    out_delay = -best_lag;
    return SignalStatus::kOk;
}

} // namespace keel::signal
