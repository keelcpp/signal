#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

SignalStatus ar_spectrum(ArrayRef<f64> a, f64 sigma2, usize nfft, MutArrayRef<f64> pxx_out,
                         MutArrayRef<f64> freq_out) noexcept {
    if (nfft < 2) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (pxx_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }
    const bool is_odd = (nfft & usize{1}) != 0;
    const f64 inv_nfft = 1.0 / as<f64>(nfft);
    for (usize k = 0; k < n_out; ++k) {
        const f64 w = kTau * as<f64>(k) * inv_nfft;
        c64 poly_a(1.0, 0.0);
        for (usize i = 0; i < a.size(); ++i) {
            poly_a += a[i] * keel::polar(1.0, -w * as<f64>(i + 1));
        }
        const f64 mag2 = keel::norm(poly_a);
        f64 psd = (mag2 > 0.0) ? (sigma2 / (kTau * mag2)) : 0.0;
        if (k > 0 && (is_odd || k < nfft / 2)) {
            psd *= 2.0;
        }
        pxx_out[k] = psd;
        freq_out[k] = w;
    }
    return SignalStatus::kOk;
}

struct CrossWelchAccum {
    Array<c64> sxy;
    Array<f64> sxx;
    Array<f64> syy;
    f64 win_power = 0.0;
    usize n_segments = 0;
};

SignalStatus cross_welch_impl(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window,
                              usize noverlap, usize nfft, CrossWelchAccum& out, Allocator* alloc) {
    const usize nx = x.size();
    const usize win_len = window.size();
    if (nx != y.size() || win_len == 0 || noverlap >= win_len || nfft < win_len) {
        return SignalStatus::kInvalidArgument;
    }

    f64 win_power = 0.0;
    for (usize i = 0; i < win_len; ++i) {
        win_power += window[i] * window[i];
    }
    if (!keel::is_finite(win_power) || !(win_power > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    const usize n_out = nfft / 2 + 1;
    out.sxx = Array<f64>(n_out, 0.0, alloc);
    out.syy = Array<f64>(n_out, 0.0, alloc);
    out.sxy = Array<c64>(n_out, c64(0.0, 0.0), alloc);
    out.win_power = win_power;
    out.n_segments = 0;

    Array<f64> seg_x(nfft, 0.0, alloc);
    Array<f64> seg_y(nfft, 0.0, alloc);
    Array<c64> spec_x(n_out, c64(0.0, 0.0), alloc);
    Array<c64> spec_y(n_out, c64(0.0, 0.0), alloc);

    const usize step = win_len - noverlap;
    for (usize start = 0; start + win_len <= nx; start += step) {
        for (usize i = 0; i < win_len; ++i) {
            seg_x[i] = x[start + i] * window[i];
            seg_y[i] = y[start + i] * window[i];
        }
        for (usize i = win_len; i < nfft; ++i) {
            seg_x[i] = 0.0;
            seg_y[i] = 0.0;
        }
        keel::fft::rfft_1d(seg_x.as_ref(), spec_x.as_mut_ref());
        keel::fft::rfft_1d(seg_y.as_ref(), spec_y.as_mut_ref());
        for (usize i = 0; i < n_out; ++i) {
            out.sxx[i] += keel::norm(spec_x[i]);
            out.syy[i] += keel::norm(spec_y[i]);
            out.sxy[i] += spec_x[i] * keel::conj(spec_y[i]);
        }
        ++out.n_segments;
    }
    return SignalStatus::kOk;
}

template <typename SampleT>
SignalStatus stft_impl(ArrayRef<SampleT> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                       StftRange range, MutArrayRef<c64> s_colmajor, Allocator* alloc) {
    const usize n = x.size();
    const usize m = window.size();
    if (m == 0 || m > n || noverlap >= m || nfft < m) {
        return SignalStatus::kInvalidArgument;
    }
    const usize hop = m - noverlap;
    const usize k_frames = (n - m) / hop + 1;
    const bool is_one_sided = (range == StftRange::kOneSided);
    const bool is_centered = (range == StftRange::kCentered);
    const usize out_rows = is_one_sided ? (nfft / 2 + 1) : nfft;
    if (s_colmajor.size() / k_frames < out_rows) {
        return SignalStatus::kInvalidArgument;
    }

    const usize c_shift = ((nfft % 2) == 0) ? (nfft / 2 + 1) : ((nfft + 1) / 2);

    Array<c64> frame(nfft, c64(0.0, 0.0), alloc);
    for (usize k = 0; k < k_frames; ++k) {
        const usize start = k * hop;
        for (usize i = 0; i < m; ++i) {
            if constexpr (IsSame<SampleT, c64>) {
                frame[i] = x[start + i] * window[i];
            } else {
                frame[i] = c64(x[start + i] * window[i], 0.0);
            }
        }
        for (usize i = m; i < nfft; ++i) {
            frame[i] = c64(0.0, 0.0);
        }
        keel::fft::fft_1d_inplace(frame.as_mut_ref());
        if (is_centered) {
            for (usize r = 0; r < out_rows; ++r) {
                s_colmajor[k * out_rows + r] = frame[(r + c_shift) % nfft];
            }
        } else {
            for (usize r = 0; r < out_rows; ++r) {
                s_colmajor[k * out_rows + r] = frame[r];
            }
        }
    }
    return SignalStatus::kOk;
}

} // namespace

SignalStatus periodogram(ArrayRef<f64> x, ArrayRef<f64> window, usize nfft, f64 fs,
                         MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out, Allocator* alloc) {
    const usize n = x.size();
    if (n == 0 || nfft == 0 || !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (pxx_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }
    if (!window.empty() && window.size() != n) {
        return SignalStatus::kInvalidArgument;
    }

    const bool use_win = !window.empty();
    const usize copy_n = keel::min(n, nfft);
    Array<f64> buf(nfft, 0.0, alloc);
    f64 win_power = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_win ? window[i] : 1.0;
        win_power += w * w;
        if (i < copy_n) {
            buf[i] = x[i] * w;
        }
    }
    if (!keel::is_finite(win_power) || !(win_power > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    Array<c64> spec(n_out, c64(0.0, 0.0), alloc);
    keel::fft::rfft_1d(buf.as_ref(), spec.as_mut_ref());

    const bool is_odd = (nfft & usize{1}) != 0;
    const f64 scale = 1.0 / (win_power * fs);
    const f64 df = fs / as<f64>(nfft);
    for (usize i = 0; i < n_out; ++i) {
        f64 mag2 = keel::norm(spec[i]);
        if (i > 0 && (is_odd || i < nfft / 2)) {
            mag2 *= 2.0;
        }
        pxx_out[i] = mag2 * scale;
        freq_out[i] = df * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus periodogram_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize nfft, f64 fs,
                             MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out,
                             Allocator* alloc) {
    const usize n = x.size();
    if (n == 0 || nfft == 0 || !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    if (pxx_out.size() < nfft || freq_out.size() < nfft) {
        return SignalStatus::kInvalidArgument;
    }
    if (!window.empty() && window.size() != n) {
        return SignalStatus::kInvalidArgument;
    }

    const bool use_win = !window.empty();
    const usize copy_n = keel::min(n, nfft);
    Array<c64> buf(nfft, c64(0.0, 0.0), alloc);
    f64 win_power = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 w = use_win ? window[i] : 1.0;
        win_power += w * w;
        if (i < copy_n) {
            buf[i] = x[i] * w;
        }
    }
    if (!keel::is_finite(win_power) || !(win_power > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    keel::fft::fft_1d_inplace(buf.as_mut_ref());

    const f64 scale = 1.0 / (win_power * fs);
    for (usize i = 0; i < nfft; ++i) {
        pxx_out[i] = keel::norm(buf[i]) * scale;
        freq_out[i] = fs * as<f64>(i) / as<f64>(nfft);
    }
    return SignalStatus::kOk;
}

SignalStatus pwelch(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap, usize nfft, f64 fs,
                    MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out, Allocator* alloc) {
    const usize nx = x.size();
    const usize win_len = window.size();
    if (nx == 0 || win_len == 0 || noverlap >= win_len || nfft < win_len ||
        !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (pxx_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }

    f64 win_power = 0.0;
    for (usize i = 0; i < win_len; ++i) {
        win_power += window[i] * window[i];
    }
    if (!keel::is_finite(win_power) || !(win_power > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    for (usize i = 0; i < n_out; ++i) {
        pxx_out[i] = 0.0;
    }
    Array<f64> seg(nfft, 0.0, alloc);
    Array<c64> spec(n_out, c64(0.0, 0.0), alloc);

    const bool is_odd = (nfft & usize{1}) != 0;
    const usize step = win_len - noverlap;
    usize n_segments = 0;
    for (usize start = 0; start + win_len <= nx; start += step) {
        for (usize i = 0; i < win_len; ++i) {
            seg[i] = x[start + i] * window[i];
        }
        for (usize i = win_len; i < nfft; ++i) {
            seg[i] = 0.0;
        }
        keel::fft::rfft_1d(seg.as_ref(), spec.as_mut_ref());
        for (usize i = 0; i < n_out; ++i) {
            f64 mag2 = keel::norm(spec[i]);
            if (i > 0 && (is_odd || i < nfft / 2)) {
                mag2 *= 2.0;
            }
            pxx_out[i] += mag2;
        }
        ++n_segments;
    }

    const f64 scale = (n_segments > 0) ? (1.0 / (win_power * fs * as<f64>(n_segments))) : 0.0;
    const f64 df = fs / as<f64>(nfft);
    for (usize i = 0; i < n_out; ++i) {
        pxx_out[i] *= scale;
        freq_out[i] = df * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus pwelch_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize noverlap, usize nfft, f64 fs,
                        MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out, Allocator* alloc) {
    const usize nx = x.size();
    const usize win_len = window.size();
    if (nx == 0 || win_len == 0 || noverlap >= win_len || nfft < win_len ||
        !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    if (pxx_out.size() < nfft || freq_out.size() < nfft) {
        return SignalStatus::kInvalidArgument;
    }

    f64 win_power = 0.0;
    for (usize i = 0; i < win_len; ++i) {
        win_power += window[i] * window[i];
    }
    if (!keel::is_finite(win_power) || !(win_power > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    for (usize i = 0; i < nfft; ++i) {
        pxx_out[i] = 0.0;
    }
    Array<c64> buf(nfft, c64(0.0, 0.0), alloc);

    const usize step = win_len - noverlap;
    usize n_segments = 0;
    for (usize start = 0; start + win_len <= nx; start += step) {
        for (usize i = 0; i < win_len; ++i) {
            buf[i] = x[start + i] * window[i];
        }
        for (usize i = win_len; i < nfft; ++i) {
            buf[i] = c64(0.0, 0.0);
        }
        keel::fft::fft_1d_inplace(buf.as_mut_ref());
        for (usize i = 0; i < nfft; ++i) {
            pxx_out[i] += keel::norm(buf[i]);
        }
        ++n_segments;
    }

    const f64 scale = (n_segments > 0) ? (1.0 / (win_power * fs * as<f64>(n_segments))) : 0.0;
    for (usize i = 0; i < nfft; ++i) {
        pxx_out[i] *= scale;
        freq_out[i] = fs * as<f64>(i) / as<f64>(nfft);
    }
    return SignalStatus::kOk;
}

SignalStatus cpsd(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window, usize noverlap,
                  usize nfft, f64 fs, MutArrayRef<c64> pxy_out, MutArrayRef<f64> freq_out,
                  Allocator* alloc) {
    if (!keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (pxy_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }
    CrossWelchAccum acc;
    const SignalStatus st = cross_welch_impl(x, y, window, noverlap, nfft, acc, alloc);
    if (st != SignalStatus::kOk) {
        return st;
    }
    const bool is_odd = (nfft & usize{1}) != 0;
    const f64 scale =
        (acc.n_segments > 0) ? (1.0 / (acc.win_power * fs * as<f64>(acc.n_segments))) : 0.0;
    const f64 df = fs / as<f64>(nfft);
    for (usize i = 0; i < n_out; ++i) {
        c64 s = acc.sxy[i] * scale;
        if (i > 0 && (is_odd || i < nfft / 2)) {
            s *= 2.0;
        }
        pxy_out[i] = s;
        freq_out[i] = df * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus mscohere(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window, usize noverlap,
                      usize nfft, f64 fs, MutArrayRef<f64> cxy_out, MutArrayRef<f64> freq_out,
                      Allocator* alloc) {
    if (!keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (cxy_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }
    CrossWelchAccum acc;
    const SignalStatus st = cross_welch_impl(x, y, window, noverlap, nfft, acc, alloc);
    if (st != SignalStatus::kOk) {
        return st;
    }
    const f64 df = fs / as<f64>(nfft);
    for (usize i = 0; i < n_out; ++i) {
        const f64 denom = acc.sxx[i] * acc.syy[i];
        cxy_out[i] = (denom > 0.0) ? (keel::norm(acc.sxy[i]) / denom) : 0.0;
        freq_out[i] = df * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus tfestimate(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window, usize noverlap,
                        usize nfft, f64 fs, MutArrayRef<c64> txy_out, MutArrayRef<f64> freq_out,
                        Allocator* alloc) {
    if (!keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize n_out = nfft / 2 + 1;
    if (txy_out.size() < n_out || freq_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }
    CrossWelchAccum acc;
    const SignalStatus st = cross_welch_impl(x, y, window, noverlap, nfft, acc, alloc);
    if (st != SignalStatus::kOk) {
        return st;
    }
    const f64 df = fs / as<f64>(nfft);
    for (usize i = 0; i < n_out; ++i) {
        const c64 syx = keel::conj(acc.sxy[i]);
        txy_out[i] = (acc.sxx[i] > 0.0) ? (syx / acc.sxx[i]) : c64(0.0, 0.0);
        freq_out[i] = df * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus pyulear(ArrayRef<f64> x, usize order, usize nfft, MutArrayRef<f64> pxx_out,
                     MutArrayRef<f64> freq_out, Allocator* alloc) {
    const usize n = x.size();
    if (order < 1 || order >= n) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> r(order + 1, 0.0, alloc);
    for (usize k = 0; k <= order; ++k) {
        f64 s = 0.0;
        for (usize i = 0; i + k < n; ++i) {
            s += x[i] * x[i + k];
        }
        r[k] = s / as<f64>(n);
    }

    Array<f64> a(order, 0.0, alloc);
    Array<f64> a_prev(order, 0.0, alloc);
    f64 sigma2 = r[0];
    const f64 k1 = (r[0] != 0.0) ? (-r[1] / r[0]) : 0.0;
    a[0] = k1;
    sigma2 = r[0] * (1.0 - k1 * k1);
    for (usize m = 1; m < order; ++m) {
        f64 num = r[m + 1];
        for (usize k = 0; k < m; ++k) {
            num += a[k] * r[m - k];
        }
        const f64 kappa = (sigma2 != 0.0) ? (-num / sigma2) : 0.0;
        for (usize k = 0; k < m; ++k) {
            a_prev[k] = a[k];
        }
        for (usize k = 0; k < m; ++k) {
            a[k] = a_prev[k] + kappa * a_prev[m - 1 - k];
        }
        a[m] = kappa;
        sigma2 *= (1.0 - kappa * kappa);
    }
    return ar_spectrum(a.as_ref(), sigma2, nfft, pxx_out, freq_out);
}

SignalStatus pburg(ArrayRef<f64> x, usize order, usize nfft, MutArrayRef<f64> pxx_out,
                   MutArrayRef<f64> freq_out, Allocator* alloc) {
    const usize n = x.size();
    if (order < 1 || order >= n) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> f = Array<f64>::from(x, alloc);
    Array<f64> b = Array<f64>::from(x, alloc);
    Array<f64> a(order, 0.0, alloc);
    Array<f64> a_prev(order, 0.0, alloc);

    f64 sigma2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        sigma2 += x[i] * x[i];
    }
    sigma2 /= as<f64>(n);

    for (usize m = 0; m < order; ++m) {
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize i = 0; i + 1 < n - m; ++i) {
            num += f[i + 1] * b[i];
            den += f[i + 1] * f[i + 1] + b[i] * b[i];
        }
        const f64 kappa = (den > 0.0) ? (-2.0 * num / den) : 0.0;
        for (usize k = 0; k < m; ++k) {
            a_prev[k] = a[k];
        }
        for (usize k = 0; k < m; ++k) {
            a[k] = a_prev[k] + kappa * a_prev[m - 1 - k];
        }
        a[m] = kappa;
        sigma2 *= (1.0 - kappa * kappa);

        const usize active = n - m - 1;
        for (usize i = 0; i < active; ++i) {
            const f64 fn = f[i + 1] + kappa * b[i];
            const f64 bn = b[i] + kappa * f[i + 1];
            f[i] = fn;
            b[i] = bn;
        }
    }
    return ar_spectrum(a.as_ref(), sigma2, nfft, pxx_out, freq_out);
}

SignalStatus spectrogram(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                         MutArrayRef<c64> s_colmajor, MutArrayRef<f64> freq_out,
                         MutArrayRef<f64> time_out, Allocator* alloc) {
    const usize nx = x.size();
    const usize win_len = window.size();
    if (win_len == 0 || win_len > nx || noverlap >= win_len || nfft < win_len) {
        return SignalStatus::kInvalidArgument;
    }
    const usize step = win_len - noverlap;
    const usize n_seg = (nx - win_len) / step + 1;
    const usize n_freqs = nfft / 2 + 1;
    if (s_colmajor.size() / n_seg < n_freqs || freq_out.size() < n_freqs ||
        time_out.size() < n_seg) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> seg_buf(nfft, 0.0, alloc);
    Array<c64> spec(n_freqs, c64(0.0, 0.0), alloc);
    for (usize seg = 0; seg < n_seg; ++seg) {
        const usize start = seg * step;
        for (usize i = 0; i < win_len; ++i) {
            seg_buf[i] = x[start + i] * window[i];
        }
        for (usize i = win_len; i < nfft; ++i) {
            seg_buf[i] = 0.0;
        }
        keel::fft::rfft_1d(seg_buf.as_ref(), spec.as_mut_ref());
        for (usize i = 0; i < n_freqs; ++i) {
            s_colmajor[seg * n_freqs + i] = spec[i];
        }
        time_out[seg] = as<f64>(start + win_len / 2);
    }
    const f64 dw = kTau / as<f64>(nfft);
    for (usize i = 0; i < n_freqs; ++i) {
        freq_out[i] = dw * as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus stft(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                  StftRange range, MutArrayRef<c64> s_colmajor, Allocator* alloc) {
    return stft_impl(x, window, noverlap, nfft, range, s_colmajor, alloc);
}

SignalStatus stft_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                      StftRange range, MutArrayRef<c64> s_colmajor, Allocator* alloc) {
    return stft_impl(x, window, noverlap, nfft, range, s_colmajor, alloc);
}

SignalStatus istft(ArrayRef<c64> s_colmajor, usize in_rows, usize n_frames, ArrayRef<f64> window,
                   usize noverlap, usize nfft, StftRange range, MutArrayRef<f64> x_out,
                   Allocator* alloc) {
    const usize m = window.size();
    if (m == 0 || noverlap >= m || nfft < m || n_frames == 0 || in_rows == 0) {
        return SignalStatus::kInvalidArgument;
    }
    const bool is_one_sided = (range == StftRange::kOneSided);
    const bool is_centered = (range == StftRange::kCentered);
    if (!is_one_sided && in_rows != nfft) {
        return SignalStatus::kInvalidArgument;
    }
    if (is_one_sided && in_rows != nfft / 2 + 1) {
        return SignalStatus::kInvalidArgument;
    }
    if (s_colmajor.size() / n_frames < in_rows) {
        return SignalStatus::kInvalidArgument;
    }
    const usize hop = m - noverlap;
    const usize n_out = (n_frames - 1) * hop + m;
    if (x_out.size() < n_out) {
        return SignalStatus::kInvalidArgument;
    }

    const usize inv_c_shift = nfft - (((nfft % 2) == 0) ? (nfft / 2 + 1) : ((nfft + 1) / 2));

    for (usize i = 0; i < n_out; ++i) {
        x_out[i] = 0.0;
    }
    Array<f64> wnorm(n_out, 0.0, alloc);
    Array<c64> fcol(nfft, c64(0.0, 0.0), alloc);

    for (usize k = 0; k < n_frames; ++k) {
        if (is_one_sided) {
            for (usize r = 0; r < in_rows; ++r) {
                fcol[r] = s_colmajor[k * in_rows + r];
            }
            for (usize r = 1; r < nfft - in_rows + 1; ++r) {
                fcol[nfft - r] = keel::conj(fcol[r]);
            }
        } else if (is_centered) {
            for (usize r = 0; r < nfft; ++r) {
                fcol[r] = s_colmajor[k * in_rows + ((r + inv_c_shift) % nfft)];
            }
        } else {
            for (usize r = 0; r < nfft; ++r) {
                fcol[r] = s_colmajor[k * in_rows + r];
            }
        }
        keel::fft::ifft_1d_inplace(fcol.as_mut_ref());
        const usize start = k * hop;
        for (usize i = 0; i < m; ++i) {
            x_out[start + i] += window[i] * fcol[i].real();
            wnorm[start + i] += window[i] * window[i];
        }
    }

    for (usize i = 0; i < n_out; ++i) {
        if (wnorm[i] > 0.0) {
            x_out[i] /= wnorm[i];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus iscola(ArrayRef<f64> window, usize noverlap, bool wola, ColaResult& out,
                    Allocator* alloc) {
    const usize m = window.size();
    if (m == 0 || noverlap >= m) {
        return SignalStatus::kInvalidArgument;
    }
    const usize hop = m - noverlap;
    const usize n_ovw = (m + hop - 1) / hop;
    const usize k_frames = n_ovw + 2;
    const usize l_tot = (k_frames - 1) * hop + m;

    Array<f64> s(l_tot, 0.0, alloc);
    for (usize k = 0; k < k_frames; ++k) {
        for (usize i = 0; i < m; ++i) {
            const f64 wi = window[i];
            s[k * hop + i] += wola ? (wi * wi) : wi;
        }
    }

    const usize stable_start = (n_ovw - 1) * hop;
    Array<f64> sorted(hop, 0.0, alloc);
    for (usize i = 0; i < hop; ++i) {
        sorted[i] = s[stable_start + i];
    }
    keel::sort(sorted);

    const f64 median = ((hop % 2) == 0)
                           ? (0.5 * (sorted[hop / 2 - 1] + sorted[hop / 2]))
                           : sorted[hop / 2];

    f64 max_dev = 0.0;
    for (usize i = 0; i < hop; ++i) {
        max_dev = keel::max(max_dev, keel::abs(s[stable_start + i] - median));
    }

    const f64 eps = Limits<f64>::kEpsilon;
    const f64 tol = (keel::abs(median) > 0.0) ? (keel::abs(median) * eps) : eps;
    out.is_cola = (max_dev <= tol);
    out.median = median;
    out.max_dev = max_dev;
    return SignalStatus::kOk;
}

} // namespace keel::signal
