#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

struct PsdPair {
    Array<f64> pxx;
    Array<f64> f;
};

SignalStatus compute_psd_rect(ArrayRef<f64> x, f64 fs, PsdPair& out, Allocator* alloc) {
    const usize n = x.size();
    const f64 eff_fs = (fs > 0.0) ? fs : kTau;
    const usize nbins = (n > 0) ? (n / 2 + 1) : 0;
    out.pxx = Array<f64>(nbins, 0.0, alloc);
    out.f = Array<f64>(nbins, 0.0, alloc);
    if (n == 0) return SignalStatus::kOk;
    return periodogram(x, {}, n, eff_fs, out.pxx.as_mut_ref(), out.f.as_mut_ref(), alloc);
}

f64 integrate_trapz_band(ArrayRef<f64> p, ArrayRef<f64> f, f64 f_lo, f64 f_hi) noexcept {
    const usize n = p.size();
    if (n < 2) return 0.0;
    if (f_lo > f_hi) keel::swap(f_lo, f_hi);
    f64 s = 0.0;
    for (usize i = 0; i + 1 < n; ++i) {
        f64 a = f[i];
        f64 b = f[i + 1];
        if (b <= f_lo) continue;
        if (a >= f_hi) break;
        f64 pa = p[i];
        f64 pb = p[i + 1];
        if (a < f_lo) {
            const f64 t = (f_lo - a) / (b - a);
            pa = pa + t * (pb - pa);
            a = f_lo;
        }
        if (b > f_hi) {
            const f64 t = (f_hi - f[i]) / (f[i + 1] - f[i]);
            pb = p[i] + t * (p[i + 1] - p[i]);
            b = f_hi;
        }
        s += 0.5 * (pa + pb) * (b - a);
    }
    return s;
}

f64 integrate_rect_full(ArrayRef<f64> p, ArrayRef<f64> f) noexcept {
    const usize n = p.size();
    if (n < 2) return 0.0;
    f64 total = 0.0;
    for (usize i = 0; i < n; ++i) {
        f64 w = 0.0;
        if (i == 0) {
            w = f[1] - f[0];
        } else if (i + 1 == n) {
            w = f[n - 1] - f[n - 2];
        } else {
            w = 0.5 * (f[i + 1] - f[i - 1]);
        }
        total += p[i] * w;
    }
    return total;
}

struct PsdMoments {
    f64 sum_p = 0.0;
    f64 mean = 0.0;
    f64 var = 0.0;
    f64 m3 = 0.0;
    f64 m4 = 0.0;
};

PsdMoments compute_psd_moments(ArrayRef<f64> p, ArrayRef<f64> f) noexcept {
    f64 sum_p = 0.0;
    for (usize i = 0; i < p.size(); ++i) sum_p += p[i];
    if (sum_p <= 0.0) return {};
    f64 mean = 0.0;
    for (usize i = 0; i < p.size(); ++i) mean += f[i] * p[i];
    mean /= sum_p;
    f64 var = 0.0;
    f64 m3 = 0.0;
    f64 m4 = 0.0;
    for (usize i = 0; i < p.size(); ++i) {
        const f64 d = f[i] - mean;
        const f64 d2 = d * d;
        var += d2 * p[i];
        m3 += d2 * d * p[i];
        m4 += d2 * d2 * p[i];
    }
    return {sum_p, mean, var / sum_p, m3 / sum_p, m4 / sum_p};
}

constexpr usize next_pow2_min256(usize n) noexcept {
    usize p = 256;
    while (p < n) p <<= 1;
    return p;
}

SignalStatus compute_psd_kaiser(ArrayRef<f64> x, f64 fs, PsdPair& out, Allocator* alloc) {
    const usize n = x.size();
    if (n == 0) {
        out.pxx = Array<f64>(alloc);
        out.f = Array<f64>(alloc);
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> win(n, 0.0, scratch.allocator());
    kaiser(win.as_mut_ref(), 38.0);
    const usize nfft = next_pow2_min256(n);
    const usize nbins = nfft / 2 + 1;
    out.pxx = Array<f64>(nbins, 0.0, alloc);
    out.f = Array<f64>(nbins, 0.0, alloc);
    const SignalStatus st =
        periodogram(x, win.as_ref(), nfft, kTau, out.pxx.as_mut_ref(), out.f.as_mut_ref(),
                    scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    if (fs > 0.0) {
        const f64 f_scale = fs / kTau;
        const f64 p_scale = as<f64>(nfft) / fs;
        for (usize i = 0; i < nbins; ++i) {
            out.f[i] *= f_scale;
            out.pxx[i] *= p_scale;
        }
    } else {
        const f64 p_scale = as<f64>(nfft) / kTau;
        for (usize i = 0; i < nbins; ++i) {
            out.pxx[i] *= p_scale;
        }
    }
    return SignalStatus::kOk;
}

constexpr i32 kSkirtBins = 8;
constexpr i32 kHarmonics = 6;
constexpr i32 kDcBins = 8;

struct HarmonicAnalysis {
    f64 p_sig = 0.0;
    f64 p_harm = 0.0;
    f64 p_noise = 0.0;
    f64 p_spur_max = 0.0;
    i32 f0_bin = 0;
};

HarmonicAnalysis analyse_harmonics(ArrayRef<f64> p, Allocator* scratch_alloc) {
    const i32 nbins = as<i32>(p.size());
    HarmonicAnalysis r{};
    if (nbins < 2) return r;

    i32 pk = kDcBins;
    f64 pk_v = -1.0;
    for (i32 i = kDcBins; i < nbins; ++i) {
        if (p[as<usize>(i)] > pk_v) {
            pk_v = p[as<usize>(i)];
            pk = i;
        }
    }
    r.f0_bin = pk;
    const i32 f0 = pk;

    Array<u8> mark(as<usize>(nbins), u8{0}, scratch_alloc);
    for (i32 j = keel::max(i32{0}, f0 - kSkirtBins);
         j < keel::min(nbins, f0 + kSkirtBins + 1); ++j) {
        mark[as<usize>(j)] = 1;
    }
    for (i32 h = 2; h <= kHarmonics; ++h) {
        const i32 bin = h * f0;
        if (bin >= nbins) break;
        for (i32 j = keel::max(i32{0}, bin - kSkirtBins);
             j < keel::min(nbins, bin + kSkirtBins + 1); ++j) {
            if (mark[as<usize>(j)] == 0) mark[as<usize>(j)] = 2;
        }
    }

    for (i32 i = 0; i < nbins; ++i) {
        const usize ui = as<usize>(i);
        if (i < kDcBins && mark[ui] == 0) continue;
        if (mark[ui] == 1) {
            r.p_sig += p[ui];
        } else if (mark[ui] == 2) {
            r.p_harm += p[ui];
        } else {
            r.p_noise += p[ui];
        }
    }

    for (i32 i = 1; i + 1 < nbins; ++i) {
        const usize ui = as<usize>(i);
        if (mark[ui] == 1 || i < kDcBins) continue;
        if (p[ui] >= p[ui - 1] && p[ui] >= p[ui + 1] && p[ui] > r.p_spur_max) {
            r.p_spur_max = p[ui];
        }
    }
    return r;
}

inline f64 safe_db(f64 num, f64 den) noexcept {
    if (num <= 0.0 || den <= 0.0) return Limits<f64>::quiet_nan();
    return 10.0 * keel::log10(num / den);
}

f64 leakage_to_overlap_ratio(f64 leakage) noexcept {
    static constexpr f64 kTable[21] = {
        0.83872, 0.83390, 0.82868, 0.82290, 0.81662, 0.80962, 0.80178,
        0.79290, 0.78298, 0.77140, 0.758072, 0.74220, 0.72306, 0.69930,
        0.66912, 0.62876, 0.57164, 0.48362, 0.33120, 0.08660, 0.0,
    };
    const f64 clamped = keel::max(0.0, keel::min(1.0, leakage));
    const f64 pos = clamped * 20.0;
    const usize idx = keel::min<usize>(19, as<usize>(keel::floor(pos)));
    const f64 frac = pos - as<f64>(idx);
    return kTable[idx] * (1.0 - frac) + kTable[idx + 1] * frac;
}

f64 kaiser_enbw_factor(f64 beta, Allocator* scratch_alloc) {
    if (beta <= 0.0) return 1.0;
    constexpr usize kRefLen = 1000;
    Array<f64> w(kRefLen, 0.0, scratch_alloc);
    kaiser(w.as_mut_ref(), beta);
    f64 sum_w = 0.0;
    f64 sum_w2 = 0.0;
    for (usize i = 0; i < kRefLen; ++i) {
        sum_w += w[i];
        sum_w2 += w[i] * w[i];
    }
    return (sum_w > 0.0) ? (as<f64>(kRefLen) * sum_w2 / (sum_w * sum_w)) : 1.0;
}

Array<f64> normalized_kaiser(usize len, f64 beta, Allocator* alloc) {
    Array<f64> out(len, 0.0, alloc);
    if (len == 0) return out;
    if (len == 1) {
        out[0] = 1.0;
        return out;
    }
    kaiser(out.as_mut_ref(), beta);
    f64 sum_w = 0.0;
    for (usize i = 0; i < len; ++i) sum_w += out[i];
    const f64 inv = (sum_w > 0.0) ? (1.0 / sum_w) : (1.0 / as<f64>(len));
    for (usize i = 0; i < len; ++i) out[i] *= inv;
    return out;
}

SignalStatus eval_segments_czt_power_real(ArrayRef<f64> segs, usize len, usize n_segs, usize n_f,
                                          c64 w, c64 a, f64 scale, f64* power_out,
                                          Allocator* scratch_alloc) {
    if (n_segs == 0 || len == 0) return SignalStatus::kOk;
    if (len == 1) {
        for (usize s = 0; s < n_segs; ++s) {
            const f64 v = segs[s];
            const f64 p0 = scale * v * v;
            for (usize k = 0; k < n_f; ++k) power_out[s * n_f + k] = p0;
        }
        return SignalStatus::kOk;
    }
    Array<c64> x_out(n_f * n_segs, c64(0.0, 0.0), scratch_alloc);
    const SignalStatus st =
        czt_batch_real(segs, len, n_segs, n_f, w, a, x_out.as_mut_ref(), scratch_alloc);
    if (st != SignalStatus::kOk) return st;
    for (usize s = 0; s < n_segs; ++s) {
        for (usize k = 0; k < n_f; ++k) {
            power_out[s * n_f + k] = scale * keel::norm(x_out[s * n_f + k]);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus eval_segments_czt_power_c64(ArrayRef<c64> segs, usize len, usize n_segs, usize n_f,
                                         c64 w, c64 a, f64 scale, f64* power_out,
                                         Allocator* scratch_alloc) {
    if (n_segs == 0 || len == 0) return SignalStatus::kOk;
    if (len == 1) {
        for (usize s = 0; s < n_segs; ++s) {
            const f64 p0 = scale * keel::norm(segs[s]);
            for (usize k = 0; k < n_f; ++k) power_out[s * n_f + k] = p0;
        }
        return SignalStatus::kOk;
    }
    Array<c64> x_out(n_f * n_segs, c64(0.0, 0.0), scratch_alloc);
    const SignalStatus st =
        czt_batch(segs, len, n_segs, n_f, w, a, x_out.as_mut_ref(), scratch_alloc);
    if (st != SignalStatus::kOk) return st;
    for (usize s = 0; s < n_segs; ++s) {
        for (usize k = 0; k < n_f; ++k) {
            power_out[s * n_f + k] = scale * keel::norm(x_out[s * n_f + k]);
        }
    }
    return SignalStatus::kOk;
}

template <typename T>
SignalStatus pspectrum_impl(ArrayRef<T> x_col_major, usize n_samples, usize n_channels,
                            bool is_complex, const PspectrumOptions& opts, PspectrumResult& out,
                            Allocator* alloc) {
    if (n_samples < 2 || n_channels == 0 || x_col_major.size() < n_samples * n_channels) {
        return SignalStatus::kInvalidArgument;
    }
    if (!keel::is_finite(opts.leakage) || opts.leakage < 0.0 || opts.leakage > 1.0) {
        return SignalStatus::kInvalidArgument;
    }
    if (opts.overlap_percent >= 100.0 ||
        (opts.overlap_percent < 0.0 && opts.overlap_percent != -1.0)) {
        return SignalStatus::kInvalidArgument;
    }
    if (!keel::is_finite(opts.fs) || opts.fs <= 0.0) {
        return SignalStatus::kInvalidArgument;
    }

    const f64 fs = opts.fs;
    const f64 beta = 40.0 * (1.0 - opts.leakage);
    const f64 r = (opts.overlap_percent >= 0.0) ? (opts.overlap_percent / 100.0)
                                                : leakage_to_overlap_ratio(opts.leakage);

    bool two_sided = opts.has_two_sided ? opts.two_sided : is_complex;
    f64 fmin = two_sided ? (-0.5 * fs) : 0.0;
    f64 fmax = 0.5 * fs;
    if (!keel::is_nan(opts.fmin) && !keel::is_nan(opts.fmax)) {
        if (opts.fmin >= opts.fmax || opts.fmin < -0.5 * fs - 1e-12 ||
            opts.fmax > 0.5 * fs + 1e-12) {
            return SignalStatus::kInvalidArgument;
        }
        fmin = opts.fmin;
        fmax = opts.fmax;
        if (fmin < 0.0) two_sided = true;
    }

    const bool is_spec = (opts.mode == PspectrumMode::kSpectrogram);
    const usize n_f = is_spec ? 1024 : 4096;
    const f64 scale = two_sided ? 1.0 : 2.0;

    out.n_freq = n_f;
    out.f = Array<f64>(n_f, 0.0, alloc);
    for (usize k = 0; k < n_f; ++k) {
        out.f[k] = fmin + (fmax - fmin) * as<f64>(k) / as<f64>(n_f - 1);
    }

    const c64 a_pt = keel::polar(1.0, kTau * fmin / fs);
    const c64 w_step = keel::polar(1.0, -kTau * (fmax - fmin) / (as<f64>(n_f - 1) * fs));

    ArenaTemp scratch = keel::get_scratch(alloc);

    if (is_spec) {
        if (n_channels > 1) return SignalStatus::kInvalidArgument;
        usize wlen = 0;
        if (opts.time_resolution > 0.0) {
            wlen = keel::min(n_samples, as<usize>(keel::max(1.0, keel::round(opts.time_resolution * fs))));
        } else {
            usize d_div = 2;
            if (n_samples >= 64) {
                const i32 exp2 = as<i32>(keel::floor(keel::log2(as<f64>(n_samples)))) - 7;
                if (exp2 <= 3) {
                    d_div = 8;
                } else if (exp2 >= 7) {
                    d_div = 128;
                } else {
                    d_div = usize{1} << as<usize>(exp2);
                }
            }
            wlen = (n_samples + d_div - 1) / d_div;
        }
        wlen = keel::max<usize>(1, keel::min(n_samples, wlen));
        usize nov = as<usize>(keel::floor(r * as<f64>(wlen)));
        if (nov >= wlen) nov = wlen - 1;
        const usize hop = keel::max<usize>(1, wlen - nov);
        const usize n_t = (n_samples > wlen) ? ((n_samples - wlen + hop - 1) / hop + 1) : 1;

        out.n_cols = n_t;
        out.p_col_major = Array<f64>(n_f * n_t, 0.0, alloc);
        out.t = Array<f64>(n_t, 0.0, alloc);
        const f64 time_step_scale = opts.is_norm_freq ? 1.0 : (1.0 / fs);
        for (usize m = 0; m < n_t; ++m) {
            out.t[m] = opts.t0 + (0.5 * as<f64>(wlen) + as<f64>(m * hop)) * time_step_scale;
        }

        const usize last_start = (n_t - 1) * hop;
        const usize last_len = n_samples - last_start;
        const usize n_full = (last_len == wlen) ? n_t : (n_t - 1);

        if (n_full > 0) {
            const Array<f64> w_full = normalized_kaiser(wlen, beta, scratch.allocator());
            Array<T> seg_mat(wlen * n_full, T{}, scratch.allocator());
            for (usize m = 0; m < n_full; ++m) {
                const usize s = m * hop;
                for (usize i = 0; i < wlen; ++i) {
                    seg_mat[m * wlen + i] = x_col_major[s + i] * w_full[i];
                }
            }
            SignalStatus st = SignalStatus::kOk;
            if constexpr (IsSame<T, c64>) {
                st = eval_segments_czt_power_c64(seg_mat.as_ref(), wlen, n_full, n_f, w_step, a_pt,
                                                 scale, out.p_col_major.data(), scratch.allocator());
            } else {
                st = eval_segments_czt_power_real(seg_mat.as_ref(), wlen, n_full, n_f, w_step, a_pt,
                                                  scale, out.p_col_major.data(), scratch.allocator());
            }
            if (st != SignalStatus::kOk) return st;
        }

        if (n_full < n_t) {
            const Array<f64> w_last = normalized_kaiser(last_len, beta, scratch.allocator());
            Array<T> seg_last(last_len, T{}, scratch.allocator());
            for (usize i = 0; i < last_len; ++i) {
                seg_last[i] = x_col_major[last_start + i] * w_last[i];
            }
            SignalStatus st = SignalStatus::kOk;
            if constexpr (IsSame<T, c64>) {
                st = eval_segments_czt_power_c64(seg_last.as_ref(), last_len, 1, n_f, w_step, a_pt,
                                                 scale, out.p_col_major.data() + n_full * n_f,
                                                 scratch.allocator());
            } else {
                st = eval_segments_czt_power_real(seg_last.as_ref(), last_len, 1, n_f, w_step, a_pt,
                                                  scale, out.p_col_major.data() + n_full * n_f,
                                                  scratch.allocator());
            }
            if (st != SignalStatus::kOk) return st;
        }
        return SignalStatus::kOk;
    }

    // Power mode
    usize wlen = 0;
    if (opts.time_resolution > 0.0) {
        wlen = keel::min(n_samples, as<usize>(keel::max(1.0, keel::round(opts.time_resolution * fs))));
    } else {
        const usize target = as<usize>(keel::ceil(1024.0 * kaiser_enbw_factor(beta, scratch.allocator())));
        wlen = keel::min(n_samples, keel::max<usize>(2, target));
    }
    wlen = keel::max<usize>(1, keel::min(n_samples, wlen));
    usize nov = as<usize>(keel::floor(r * as<f64>(wlen)));
    if (nov >= wlen) nov = wlen - 1;
    const usize hop = keel::max<usize>(1, wlen - nov);
    const usize n_t = (n_samples > wlen) ? ((n_samples - wlen + hop - 1) / hop + 1) : 1;

    const Array<f64> w_full = normalized_kaiser(wlen, beta, scratch.allocator());
    Array<usize> starts(n_t, usize{0}, scratch.allocator());
    if (n_t > 1) {
        for (usize m = 0; m < n_t; ++m) {
            starts[m] = as<usize>(keel::round_i64(
                as<f64>(m) * as<f64>(n_samples - wlen) / as<f64>(n_t - 1)));
        }
    }

    out.n_cols = n_channels;
    out.p_col_major = Array<f64>(n_f * n_channels, 0.0, alloc);
    out.t = Array<f64>(alloc);
    Array<f64> seg_pow(n_f * n_t, 0.0, scratch.allocator());
    Array<T> seg_mat(wlen * n_t, T{}, scratch.allocator());

    for (usize ch = 0; ch < n_channels; ++ch) {
        const T* col = x_col_major.data() + ch * n_samples;
        for (usize m = 0; m < n_t; ++m) {
            const usize s = starts[m];
            for (usize i = 0; i < wlen; ++i) {
                seg_mat[m * wlen + i] = col[s + i] * w_full[i];
            }
        }
        SignalStatus st = SignalStatus::kOk;
        if constexpr (IsSame<T, c64>) {
            st = eval_segments_czt_power_c64(seg_mat.as_ref(), wlen, n_t, n_f, w_step, a_pt, scale,
                                             seg_pow.data(), scratch.allocator());
        } else {
            st = eval_segments_czt_power_real(seg_mat.as_ref(), wlen, n_t, n_f, w_step, a_pt, scale,
                                              seg_pow.data(), scratch.allocator());
        }
        if (st != SignalStatus::kOk) return st;
        const f64 inv_nt = 1.0 / as<f64>(n_t);
        for (usize k = 0; k < n_f; ++k) {
            f64 acc = 0.0;
            for (usize m = 0; m < n_t; ++m) acc += seg_pow[m * n_f + k];
            out.p_col_major[ch * n_f + k] = acc * inv_nt;
        }
    }
    return SignalStatus::kOk;
}

} // namespace

SignalStatus bandpower_psd(ArrayRef<f64> pxx, ArrayRef<f64> f, f64 f_lo, f64 f_hi, bool use_range,
                           f64& out_power) noexcept {
    const usize n = pxx.size();
    if (f.size() != n || n < 2) return SignalStatus::kInvalidArgument;
    if (!use_range || (f_lo <= f[0] && f_hi >= f[n - 1])) {
        out_power = integrate_rect_full(pxx, f);
    } else {
        out_power = integrate_trapz_band(pxx, f, f_lo, f_hi);
    }
    return SignalStatus::kOk;
}

SignalStatus bandpower_time(ArrayRef<f64> x, f64 fs, f64 f_lo, f64 f_hi, f64& out_power,
                            Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    out_power = integrate_trapz_band(p.pxx.as_ref(), p.f.as_ref(), f_lo, f_hi);
    return SignalStatus::kOk;
}

SignalStatus meanfreq_time(ArrayRef<f64> x, f64 fs, f64& out_freq, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    f64 num = 0.0;
    f64 den = 0.0;
    for (usize i = 0; i + 1 < p.f.size(); ++i) {
        const f64 df = p.f[i + 1] - p.f[i];
        const f64 pmid = 0.5 * (p.pxx[i] + p.pxx[i + 1]);
        const f64 fmid = 0.5 * (p.f[i] + p.f[i + 1]);
        num += fmid * pmid * df;
        den += pmid * df;
    }
    out_freq = (den > 0.0) ? (num / den) : 0.0;
    return SignalStatus::kOk;
}

SignalStatus medfreq_time(ArrayRef<f64> x, f64 fs, f64& out_freq, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const usize n = p.f.size();
    if (n < 2) {
        out_freq = 0.0;
        return SignalStatus::kOk;
    }
    Array<f64> cum(n, 0.0, scratch.allocator());
    for (usize i = 1; i < n; ++i) {
        cum[i] = cum[i - 1] + 0.5 * (p.pxx[i - 1] + p.pxx[i]) * (p.f[i] - p.f[i - 1]);
    }
    const f64 total = cum[n - 1];
    if (total <= 0.0) {
        out_freq = 0.0;
        return SignalStatus::kOk;
    }
    const f64 half = 0.5 * total;
    for (usize i = 1; i < n; ++i) {
        if (cum[i] >= half) {
            const f64 t = (half - cum[i - 1]) / (cum[i] - cum[i - 1]);
            out_freq = p.f[i - 1] + t * (p.f[i] - p.f[i - 1]);
            return SignalStatus::kOk;
        }
    }
    out_freq = p.f[n - 1];
    return SignalStatus::kOk;
}

SignalStatus enbw(ArrayRef<f64> window, f64 fs, f64& out_bw) noexcept {
    const usize n = window.size();
    if (n == 0) {
        out_bw = 0.0;
        return SignalStatus::kOk;
    }
    f64 sum_sq = 0.0;
    f64 sum_w = 0.0;
    for (usize i = 0; i < n; ++i) {
        sum_sq += window[i] * window[i];
        sum_w += window[i];
    }
    const f64 bin = (sum_w * sum_w > 0.0) ? (as<f64>(n) * sum_sq / (sum_w * sum_w)) : 0.0;
    const f64 eff_fs = (fs > 0.0) ? fs : as<f64>(n);
    out_bw = bin * eff_fs / as<f64>(n);
    return SignalStatus::kOk;
}

SignalStatus obw_time(ArrayRef<f64> x, f64 fs, f64 pct, ObwResult& out, Allocator* alloc) {
    out = {};
    const usize n = x.size();
    if (n < 2) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> xf(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(x, xf.as_mut_ref());
    const usize n_out = n / 2 + 1;
    const f64 fs_hz = (fs > 0.0) ? fs : kTau;
    const f64 df = fs_hz / as<f64>(n);

    Array<f64> p(n_out, 0.0, scratch.allocator());
    Array<f64> f(n_out, 0.0, scratch.allocator());
    for (usize k = 0; k < n_out; ++k) {
        f64 mag2 = keel::norm(xf[k]);
        if (k > 0 && 2 * k != n) mag2 *= 2.0;
        p[k] = mag2 / (fs_hz * as<f64>(n));
        f[k] = as<f64>(k) * df;
    }

    Array<f64> cum(n_out, 0.0, scratch.allocator());
    cum[0] = p[0] * df;
    for (usize i = 1; i < n_out; ++i) cum[i] = cum[i - 1] + p[i] * df;
    const f64 total = cum[n_out - 1];
    if (total <= 0.0) return SignalStatus::kOk;

    const f64 p_low = (1.0 - pct) * 0.5 * total;
    const f64 p_high = (1.0 + pct) * 0.5 * total;
    auto find_f = [&](f64 target) noexcept {
        for (usize i = 1; i < n_out; ++i) {
            if (cum[i] >= target) {
                const f64 t = (target - cum[i - 1]) / keel::max(cum[i] - cum[i - 1], 1e-300);
                return f[i - 1] + t * (f[i] - f[i - 1]) + 0.5 * df;
            }
        }
        return f[n_out - 1] + 0.5 * df;
    };
    out.flo = find_f(p_low);
    out.fhi = find_f(p_high);
    out.bw = out.fhi - out.flo;
    out.power = pct * total;
    return SignalStatus::kOk;
}

SignalStatus powerbw_time(ArrayRef<f64> x, f64 fs, f64& out_bw, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const usize n = p.f.size();
    if (n < 2) {
        out_bw = 0.0;
        return SignalStatus::kOk;
    }
    usize pk = 0;
    f64 pk_v = p.pxx[0];
    for (usize i = 1; i < n; ++i) {
        if (p.pxx[i] > pk_v) {
            pk_v = p.pxx[i];
            pk = i;
        }
    }
    const f64 thresh = pk_v * 0.5;
    auto interp_cross = [&](usize i, usize j) noexcept {
        const f64 a = p.pxx[i];
        const f64 b = p.pxx[j];
        if (a == b) return p.f[i];
        const f64 t = (thresh - a) / (b - a);
        return p.f[i] + t * (p.f[j] - p.f[i]);
    };
    f64 f_lo = p.f[0];
    f64 f_hi = p.f[n - 1];
    for (usize i = pk; i > 0; --i) {
        if (p.pxx[i - 1] < thresh) {
            f_lo = interp_cross(i - 1, i);
            break;
        }
    }
    for (usize i = pk; i + 1 < n; ++i) {
        if (p.pxx[i + 1] < thresh) {
            f_hi = interp_cross(i, i + 1);
            break;
        }
    }
    out_bw = f_hi - f_lo;
    return SignalStatus::kOk;
}

SignalStatus spectral_crest(ArrayRef<f64> x, f64 fs, f64& out_val, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    if (p.pxx.empty()) {
        out_val = 0.0;
        return SignalStatus::kOk;
    }
    f64 mx = p.pxx[0];
    f64 sm = 0.0;
    for (usize i = 0; i < p.pxx.size(); ++i) {
        if (p.pxx[i] > mx) mx = p.pxx[i];
        sm += p.pxx[i];
    }
    const f64 mean = sm / as<f64>(p.pxx.size());
    out_val = (mean > 0.0) ? (mx / mean) : 0.0;
    return SignalStatus::kOk;
}

SignalStatus spectral_flatness(ArrayRef<f64> x, f64 fs, f64& out_val, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    if (p.pxx.empty()) {
        out_val = 0.0;
        return SignalStatus::kOk;
    }
    f64 log_sum = 0.0;
    f64 sm = 0.0;
    usize valid = 0;
    for (usize i = 0; i < p.pxx.size(); ++i) {
        sm += p.pxx[i];
        if (p.pxx[i] > 0.0) {
            log_sum += keel::log(p.pxx[i]);
            ++valid;
        }
    }
    if (valid == 0 || sm <= 0.0) {
        out_val = 0.0;
        return SignalStatus::kOk;
    }
    const f64 geom = keel::exp(log_sum / as<f64>(valid));
    const f64 arith = sm / as<f64>(p.pxx.size());
    out_val = geom / arith;
    return SignalStatus::kOk;
}

SignalStatus spectral_entropy(ArrayRef<f64> x, f64 fs, f64& out_val, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    if (p.pxx.size() < 2) {
        out_val = 0.0;
        return SignalStatus::kOk;
    }
    f64 sm = 0.0;
    for (usize i = 0; i < p.pxx.size(); ++i) sm += p.pxx[i];
    if (sm <= 0.0) {
        out_val = 0.0;
        return SignalStatus::kOk;
    }
    f64 h = 0.0;
    for (usize i = 0; i < p.pxx.size(); ++i) {
        const f64 q = p.pxx[i] / sm;
        if (q > 0.0) h -= q * keel::log(q);
    }
    out_val = h / keel::log(as<f64>(p.pxx.size()));
    return SignalStatus::kOk;
}

SignalStatus spectral_kurtosis(ArrayRef<f64> x, f64 fs, f64& out_val, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const PsdMoments m = compute_psd_moments(p.pxx.as_ref(), p.f.as_ref());
    out_val = (m.var > 0.0) ? (m.m4 / (m.var * m.var)) : 0.0;
    return SignalStatus::kOk;
}

SignalStatus spectral_skewness(ArrayRef<f64> x, f64 fs, f64& out_val, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_rect(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const PsdMoments m = compute_psd_moments(p.pxx.as_ref(), p.f.as_ref());
    out_val = (m.var > 0.0) ? (m.m3 / keel::pow(m.var, 1.5)) : 0.0;
    return SignalStatus::kOk;
}

SignalStatus snr_time(ArrayRef<f64> x, f64 fs, f64& out_db, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_kaiser(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const HarmonicAnalysis h = analyse_harmonics(p.pxx.as_ref(), scratch.allocator());
    out_db = safe_db(h.p_sig, h.p_noise);
    return SignalStatus::kOk;
}

SignalStatus sinad_time(ArrayRef<f64> x, f64 fs, f64& out_db, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_kaiser(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const HarmonicAnalysis h = analyse_harmonics(p.pxx.as_ref(), scratch.allocator());
    out_db = safe_db(h.p_sig, h.p_noise + h.p_harm);
    return SignalStatus::kOk;
}

SignalStatus thd_time(ArrayRef<f64> x, f64 fs, f64& out_db, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    PsdPair p;
    const SignalStatus st = compute_psd_kaiser(x, fs, p, scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    const HarmonicAnalysis h = analyse_harmonics(p.pxx.as_ref(), scratch.allocator());
    out_db = safe_db(h.p_harm, h.p_sig);
    return SignalStatus::kOk;
}

SignalStatus sfdr_time(ArrayRef<f64> x, f64 fs, f64& out_db, Allocator* alloc) {
    const usize n = x.size();
    if (n < 2) {
        out_db = Limits<f64>::quiet_nan();
        return SignalStatus::kOk;
    }
    const f64 fs_eff = (fs > 0.0) ? fs : 1.0;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> w(n, 0.0, scratch.allocator());
    kaiser(w.as_mut_ref(), 38.0);
    f64 sum_w = 0.0;
    f64 sum_w2 = 0.0;
    Array<f64> xw(n, 0.0, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        sum_w += w[i];
        sum_w2 += w[i] * w[i];
        xw[i] = x[i] * w[i];
    }
    const f64 rbw = (sum_w > 0.0) ? (fs_eff * sum_w2 / (sum_w * sum_w)) : 0.0;
    Array<c64> xf(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(xw.as_ref(), xf.as_mut_ref());
    const usize n_bins = n / 2 + 1;
    const f64 df = fs_eff / as<f64>(n);
    Array<f64> pxx(n_bins, 0.0, scratch.allocator());
    Array<f64> f(n_bins, 0.0, scratch.allocator());
    for (usize k = 0; k < n_bins; ++k) {
        f64 mag2 = keel::norm(xf[k]);
        if (k > 0 && 2 * k != n) mag2 *= 2.0;
        pxx[k] = mag2 / (fs_eff * sum_w2);
        f[k] = as<f64>(k) * df;
    }

    auto get_tone = [&](i32 idx_tone, i32& out_left, i32& out_right) noexcept -> f64 {
        i32 i_l = idx_tone - 1;
        i32 i_r = idx_tone + 1;
        while (i_l >= 0 && pxx[as<usize>(i_l)] <= pxx[as<usize>(i_l + 1)]) --i_l;
        while (i_r < as<i32>(n_bins) && pxx[as<usize>(i_r - 1)] >= pxx[as<usize>(i_r)]) ++i_r;
        ++i_l;
        --i_r;
        out_left = i_l;
        out_right = i_r;
        f64 power = 0.0;
        if (i_l < i_r) {
            const i32 len = i_r - i_l + 1;
            for (i32 k = 0; k < len; ++k) {
                f64 w_bin = 0.0;
                if (k == 0) {
                    w_bin = f[as<usize>(i_l + 1)] - f[as<usize>(i_l)];
                } else if (k + 1 == len) {
                    w_bin = f[as<usize>(i_r)] - f[as<usize>(i_r - 1)];
                } else {
                    w_bin = 0.5 * (f[as<usize>(i_l + k + 1)] - f[as<usize>(i_l + k - 1)]);
                }
                power += pxx[as<usize>(i_l + k)] * w_bin;
            }
        } else {
            power = pxx[as<usize>(i_r)] * df;
        }
        if (power < rbw * pxx[as<usize>(idx_tone)]) {
            power = rbw * pxx[as<usize>(idx_tone)];
        }
        return power;
    };

    auto argmax_pxx = [&]() noexcept -> i32 {
        i32 best = 0;
        for (usize i = 1; i < n_bins; ++i) {
            if (pxx[i] > pxx[as<usize>(best)]) best = as<i32>(i);
        }
        return best;
    };

    pxx[0] *= 2.0;
    i32 dc_l = 0;
    i32 dc_r = 0;
    (void)get_tone(0, dc_l, dc_r);
    for (i32 k = dc_l; k <= dc_r; ++k) pxx[as<usize>(k)] = 0.0;

    const i32 i_fund = argmax_pxx();
    i32 f_l = 0;
    i32 f_r = 0;
    const f64 p_fund = get_tone(i_fund, f_l, f_r);
    for (i32 k = f_l; k <= f_r; ++k) pxx[as<usize>(k)] = 0.0;

    const i32 spur_bin = argmax_pxx();
    const i32 i_left_bin = keel::max(i32{0}, spur_bin - 1);
    const i32 i_right_bin = keel::min(as<i32>(n_bins) - 1, spur_bin + 1);
    i32 i_spur = i_left_bin;
    for (i32 k = i_left_bin + 1; k <= i_right_bin; ++k) {
        if (pxx[as<usize>(k)] > pxx[as<usize>(i_spur)]) i_spur = k;
    }
    i32 s_l = 0;
    i32 s_r = 0;
    const f64 p_spur = get_tone(i_spur, s_l, s_r);

    out_db = safe_db(p_fund, p_spur);
    return SignalStatus::kOk;
}

SignalStatus pspectrum_real(ArrayRef<f64> x_col_major, usize n_samples, usize n_channels,
                            const PspectrumOptions& opts, PspectrumResult& out, Allocator* alloc) {
    return pspectrum_impl<f64>(x_col_major, n_samples, n_channels, /*is_complex=*/false, opts, out,
                               alloc);
}

SignalStatus pspectrum_c64(ArrayRef<c64> x_col_major, usize n_samples, usize n_channels,
                           const PspectrumOptions& opts, PspectrumResult& out, Allocator* alloc) {
    return pspectrum_impl<c64>(x_col_major, n_samples, n_channels, /*is_complex=*/true, opts, out,
                               alloc);
}

SignalStatus instfreq_hilbert(ArrayRef<f64> x, f64 fs, f64 t0, MutArrayRef<f64> f_out,
                              MutArrayRef<f64> t_out, Allocator* alloc) {
    const usize n = x.size();
    if (n < 2) return SignalStatus::kOk;
    if (f_out.size() < n - 1 || (!t_out.empty() && t_out.size() < n - 1) || fs <= 0.0) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> z(n, c64(0.0, 0.0), scratch.allocator());
    const SignalStatus st = hilbert(x, z.as_mut_ref(), scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    for (usize i = 0; i + 1 < n; ++i) {
        const f64 p1 = keel::atan2(z[i + 1].imag(), z[i + 1].real());
        const f64 p0 = keel::atan2(z[i].imag(), z[i].real());
        f64 dp = p1 - p0;
        while (dp > kPi) dp -= kTau;
        while (dp <= -kPi) dp += kTau;
        f_out[i] = dp * fs / kTau;
        if (!t_out.empty()) {
            t_out[i] = t0 + (as<f64>(i) + 0.5) / fs;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus instfreq_from_spec(ArrayRef<f64> p_col_major, usize n_freq, usize n_time,
                                ArrayRef<f64> f, MutArrayRef<f64> out_if) noexcept {
    if (f.size() < n_freq || p_col_major.size() < n_freq * n_time || out_if.size() < n_time) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize m = 0; m < n_time; ++m) {
        const f64* col = p_col_major.data() + m * n_freq;
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize k = 0; k < n_freq; ++k) {
            num += f[k] * col[k];
            den += col[k];
        }
        out_if[m] = (den > 0.0) ? (num / den) : 0.0;
    }
    return SignalStatus::kOk;
}

SignalStatus instbw_from_spec(ArrayRef<f64> p_col_major, usize n_freq, usize n_time,
                              ArrayRef<f64> f, MutArrayRef<f64> out_ibw) noexcept {
    if (f.size() < n_freq || p_col_major.size() < n_freq * n_time || out_ibw.size() < n_time) {
        return SignalStatus::kInvalidArgument;
    }
    const f64 scale = 2.0 * keel::sqrt(kPi);
    for (usize m = 0; m < n_time; ++m) {
        const f64* col = p_col_major.data() + m * n_freq;
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize k = 0; k < n_freq; ++k) {
            num += f[k] * col[k];
            den += col[k];
        }
        if (!(den > 0.0)) {
            out_ibw[m] = 0.0;
            continue;
        }
        const f64 mu = num / den;
        f64 var_num = 0.0;
        for (usize k = 0; k < n_freq; ++k) {
            const f64 df = f[k] - mu;
            var_num += col[k] * df * df;
        }
        out_ibw[m] = scale * keel::sqrt(keel::max(0.0, var_num / den));
    }
    return SignalStatus::kOk;
}

SignalStatus envspectrum(ArrayRef<f64> x, f64 fs, Array<f64>& es_out, Array<f64>& f_out,
                         Allocator* alloc) {
    const usize n = x.size();
    if (n == 0) {
        es_out = Array<f64>(alloc);
        f_out = Array<f64>(alloc);
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> env(n, 0.0, scratch.allocator());
    const SignalStatus st = envelope_analytic(x, env.as_mut_ref(), {}, scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    f64 mean = 0.0;
    for (usize i = 0; i < n; ++i) mean += env[i];
    mean /= as<f64>(n);
    for (usize i = 0; i < n; ++i) env[i] -= mean;

    Array<c64> z(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(env.as_ref(), z.as_mut_ref());
    const usize n_one = n / 2 + 1;

    es_out = Array<f64>(n_one, 0.0, alloc);
    f_out = Array<f64>(n_one, 0.0, alloc);
    const f64 f_nyq = (fs > 0.0) ? (fs * 0.5) : kPi;
    const f64 scale = 2.0 / as<f64>(n);
    const f64 denom = (n_one > 1) ? as<f64>(n_one - 1) : 1.0;
    for (usize i = 0; i < n_one; ++i) {
        const f64 a = keel::abs(z[i]) * scale;
        es_out[i] = (i == 0 || i + 1 == n_one) ? (0.5 * a) : a;
        f_out[i] = f_nyq * as<f64>(i) / denom;
    }
    return SignalStatus::kOk;
}

SignalStatus tachorpm(ArrayRef<f64> x, f64 fs, bool has_threshold, f64 threshold, i32 ppr,
                      Array<f64>& rpm_out, Array<f64>& time_out, Allocator* alloc) {
    const usize n = x.size();
    time_out = Array<f64>(alloc);
    rpm_out = Array<f64>(alloc);
    if (n == 0 || fs <= 0.0 || ppr <= 0) return SignalStatus::kOk;

    f64 thr = threshold;
    if (!has_threshold) {
        f64 lo = x[0];
        f64 hi = x[0];
        for (usize i = 1; i < n; ++i) {
            if (x[i] < lo) lo = x[i];
            if (x[i] > hi) hi = x[i];
        }
        thr = 0.5 * (lo + hi);
    }

    for (usize i = 0; i + 1 < n; ++i) {
        if (x[i] < thr && x[i + 1] >= thr) {
            const f64 a = x[i];
            const f64 b = x[i + 1];
            const f64 frac = (b == a) ? 0.0 : ((thr - a) / (b - a));
            time_out.push_back((as<f64>(i) + frac) / fs);
        }
    }

    rpm_out = Array<f64>(time_out.size(), 0.0, alloc);
    for (usize i = 0; i + 1 < time_out.size(); ++i) {
        const f64 dt = time_out[i + 1] - time_out[i];
        if (dt > 0.0) rpm_out[i] = 60.0 / (dt * as<f64>(ppr));
    }
    if (rpm_out.size() >= 2) {
        rpm_out[rpm_out.size() - 1] = rpm_out[rpm_out.size() - 2];
    }
    return SignalStatus::kOk;
}

SignalStatus rainflow(ArrayRef<f64> x, Array<RainflowCycle>& cycles_out, Allocator* alloc) {
    cycles_out = Array<RainflowCycle>(alloc);
    const usize n = x.size();
    if (n == 0) return SignalStatus::kOk;

    struct Turn {
        f64 v = 0.0;
        f64 idx = 0.0;
    };
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<Turn> tp(scratch.allocator());
    tp.push_back(Turn{x[0], 1.0});
    for (usize i = 1; i + 1 < n; ++i) {
        const f64 a = x[i - 1];
        const f64 b = x[i];
        const f64 c = x[i + 1];
        if ((b > a && b > c) || (b < a && b < c)) {
            tp.push_back(Turn{b, as<f64>(i + 1)});
        }
    }
    if (n > 1) {
        tp.push_back(Turn{x[n - 1], as<f64>(n)});
    } else {
        tp.push_back(Turn{x[0], 1.0});
    }

    Array<Turn> st(scratch.allocator());
    for (usize ti = 0; ti < tp.size(); ++ti) {
        st.push_back(tp[ti]);
        while (st.size() >= 3) {
            const usize m = st.size();
            const f64 rx = keel::abs(st[m - 1].v - st[m - 2].v);
            const f64 ry = keel::abs(st[m - 2].v - st[m - 3].v);
            if (rx < ry) break;
            if (m == 3) {
                const f64 rng = keel::abs(st[0].v - st[1].v);
                const f64 mn = 0.5 * (st[0].v + st[1].v);
                cycles_out.push_back(RainflowCycle{0.5, rng, mn, st[0].idx, st[1].idx});
                st.erase(0);
                break;
            }
            const f64 mn = 0.5 * (st[m - 2].v + st[m - 3].v);
            cycles_out.push_back(RainflowCycle{1.0, ry, mn, st[m - 3].idx, st[m - 2].idx});
            st[m - 3] = st[m - 1];
            st.resize(m - 2);
        }
    }
    for (usize i = 0; i + 1 < st.size(); ++i) {
        cycles_out.push_back(RainflowCycle{0.5, keel::abs(st[i].v - st[i + 1].v),
                                           0.5 * (st[i].v + st[i + 1].v), st[i].idx,
                                           st[i + 1].idx});
    }
    return SignalStatus::kOk;
}

SignalStatus tsa(ArrayRef<f64> x, f64 fs, ArrayRef<f64> rpm, f64 fs_rpm, i32 n_per_rev,
                 Array<f64>& avg_out, Array<f64>& theta_out, Allocator* alloc) {
    if (x.empty() || rpm.empty() || n_per_rev <= 0 || fs <= 0.0) {
        avg_out = Array<f64>(alloc);
        theta_out = Array<f64>(alloc);
        return SignalStatus::kOk;
    }

    const usize n = x.size();
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> theta(n, 0.0, scratch.allocator());
    f64 acc = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 t = as<f64>(i) / fs;
        const f64 idx_r = t * fs_rpm;
        const usize i0 = (idx_r > 0.0) ? as<usize>(keel::floor(idx_r)) : usize{0};
        const usize i1 = keel::min(i0 + 1, rpm.size() - 1);
        const f64 frac = keel::min(keel::max(idx_r - as<f64>(i0), 0.0), 1.0);
        const f64 rpm_i =
            (i0 < rpm.size()) ? (rpm[i0] + frac * (rpm[i1] - rpm[i0])) : rpm[rpm.size() - 1];
        const f64 omega = kTau * rpm_i / 60.0;
        theta[i] = acc;
        acc += omega / fs;
    }

    const usize bins = as<usize>(n_per_rev);
    Array<f64> sum_bin(bins, 0.0, scratch.allocator());
    Array<i32> cnt_bin(bins, i32{0}, scratch.allocator());
    for (usize i = 0; i < n; ++i) {
        f64 ang = keel::fmod(theta[i], kTau);
        if (ang < 0.0) ang += kTau;
        i32 b = as<i32>(ang / kTau * as<f64>(n_per_rev));
        if (b >= n_per_rev) b = n_per_rev - 1;
        if (b < 0) b = 0;
        sum_bin[as<usize>(b)] += x[i];
        cnt_bin[as<usize>(b)] += 1;
    }

    avg_out = Array<f64>(bins, 0.0, alloc);
    theta_out = Array<f64>(bins, 0.0, alloc);
    for (usize b = 0; b < bins; ++b) {
        avg_out[b] = (cnt_bin[b] > 0) ? (sum_bin[b] / as<f64>(cnt_bin[b])) : 0.0;
        theta_out[b] = kTau * as<f64>(b) / as<f64>(n_per_rev);
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
