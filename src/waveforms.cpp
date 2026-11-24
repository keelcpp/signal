#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

inline f64 gauspuls_alpha(f64 fc, f64 bw) noexcept {
    const f64 pifb = kPi * fc * bw;
    return -(pifb * pifb) / (4.0 * keel::log(0.5));
}

inline f64 rectpuls_scalar(f64 tv, f64 w) noexcept {
    return (keel::abs(tv) < 0.5 * w) ? 1.0 : 0.0;
}

inline f64 tripuls_scalar(f64 tv, f64 w) noexcept {
    const f64 half = 0.5 * w;
    const f64 a = keel::abs(tv);
    return (a >= half) ? 0.0 : (1.0 - a / half);
}

inline f64 gauspuls_scalar(f64 tv, f64 fc, f64 alpha) noexcept {
    return keel::exp(-alpha * tv * tv) * keel::cos(kTau * fc * tv);
}

} // namespace

SignalStatus rectpuls(ArrayRef<f64> t, f64 w, MutArrayRef<f64> out) noexcept {
    if (w <= 0.0 || out.size() < t.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < t.size(); ++i) out[i] = rectpuls_scalar(t[i], w);
    return SignalStatus::kOk;
}

SignalStatus tripuls(ArrayRef<f64> t, f64 w, MutArrayRef<f64> out) noexcept {
    if (w <= 0.0 || out.size() < t.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < t.size(); ++i) out[i] = tripuls_scalar(t[i], w);
    return SignalStatus::kOk;
}

SignalStatus gauspuls(ArrayRef<f64> t, f64 fc, f64 bw, MutArrayRef<f64> out) noexcept {
    if (fc <= 0.0 || bw <= 0.0 || out.size() < t.size()) return SignalStatus::kInvalidArgument;
    const f64 alpha = gauspuls_alpha(fc, bw);
    for (usize i = 0; i < t.size(); ++i) out[i] = gauspuls_scalar(t[i], fc, alpha);
    return SignalStatus::kOk;
}

SignalStatus pulstran_builtin(ArrayRef<f64> t, ArrayRef<f64> d, PulstranKind kind, f64 fc_or_w,
                              f64 bw, MutArrayRef<f64> out) noexcept {
    const usize n = t.size();
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < n; ++i) out[i] = 0.0;

    if (kind == PulstranKind::kRectpuls) {
        if (fc_or_w <= 0.0) return SignalStatus::kInvalidArgument;
        for (usize k = 0; k < d.size(); ++k) {
            const f64 dk = d[k];
            for (usize i = 0; i < n; ++i) out[i] += rectpuls_scalar(t[i] - dk, fc_or_w);
        }
    } else if (kind == PulstranKind::kTripuls) {
        if (fc_or_w <= 0.0) return SignalStatus::kInvalidArgument;
        for (usize k = 0; k < d.size(); ++k) {
            const f64 dk = d[k];
            for (usize i = 0; i < n; ++i) out[i] += tripuls_scalar(t[i] - dk, fc_or_w);
        }
    } else {
        if (fc_or_w <= 0.0 || bw <= 0.0) return SignalStatus::kInvalidArgument;
        const f64 alpha = gauspuls_alpha(fc_or_w, bw);
        for (usize k = 0; k < d.size(); ++k) {
            const f64 dk = d[k];
            for (usize i = 0; i < n; ++i) out[i] += gauspuls_scalar(t[i] - dk, fc_or_w, alpha);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus square(ArrayRef<f64> t, f64 duty, MutArrayRef<f64> out) noexcept {
    if (duty < 0.0 || duty > 100.0 || out.size() < t.size()) {
        return SignalStatus::kInvalidArgument;
    }
    const f64 threshold = duty / 100.0;
    for (usize i = 0; i < t.size(); ++i) {
        f64 frac = t[i] / kTau;
        frac -= keel::floor(frac);
        out[i] = (frac < threshold) ? 1.0 : -1.0;
    }
    return SignalStatus::kOk;
}

SignalStatus sawtooth(ArrayRef<f64> t, f64 width, MutArrayRef<f64> out) noexcept {
    if (width < 0.0 || width > 1.0 || out.size() < t.size()) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < t.size(); ++i) {
        f64 frac = t[i] / kTau;
        frac -= keel::floor(frac);
        if (width == 0.0) {
            out[i] = -2.0 * frac + 1.0;
        } else if (width == 1.0) {
            out[i] = 2.0 * frac - 1.0;
        } else if (frac < width) {
            out[i] = 2.0 * frac / width - 1.0;
        } else {
            out[i] = 2.0 * (1.0 - frac) / (1.0 - width) - 1.0;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus sinc(ArrayRef<f64> t, MutArrayRef<f64> out) noexcept {
    if (out.size() < t.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < t.size(); ++i) {
        const f64 tv = t[i];
        if (tv == 0.0) {
            out[i] = 1.0;
        } else {
            const f64 pt = kPi * tv;
            out[i] = keel::sin(pt) / pt;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus gmonopuls(ArrayRef<f64> t, f64 fc, MutArrayRef<f64> out) noexcept {
    if (fc <= 0.0 || out.size() < t.size()) return SignalStatus::kInvalidArgument;
    const f64 sqrt_e = keel::sqrt(keel::exp(1.0));
    const f64 k = 2.0 * sqrt_e * kPi * fc;
    for (usize i = 0; i < t.size(); ++i) {
        const f64 tv = t[i];
        const f64 pft = kPi * fc * tv;
        out[i] = k * tv * keel::exp(-2.0 * pft * pft);
    }
    return SignalStatus::kOk;
}

SignalStatus diric(ArrayRef<f64> x, i32 n, MutArrayRef<f64> out) noexcept {
    if (n < 1 || out.size() < x.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < x.size(); ++i) {
        const f64 xv = x[i];
        const f64 s = keel::sin(xv * 0.5);
        if (keel::abs(s) < 1e-12) {
            const i64 k = keel::round_i64(xv / kTau);
            const i64 sign = ((k * as<i64>(n - 1)) & 1) ? -1 : 1;
            out[i] = as<f64>(sign);
        } else {
            out[i] = keel::sin(as<f64>(n) * xv * 0.5) / (as<f64>(n) * s);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus chirp(ArrayRef<f64> t, f64 f0, f64 t1, f64 f1, ChirpMethod method, f64 phi_deg,
                   ChirpQuadShape quad_shape, MutArrayRef<f64> out) noexcept {
    if (t1 <= 0.0 || out.size() < t.size()) return SignalStatus::kInvalidArgument;
    const usize n = t.size();
    const f64 phi_rad = phi_deg * (kPi / 180.0);

    if (method == ChirpMethod::kLinear) {
        const f64 k = (f1 - f0) / t1;
        for (usize i = 0; i < n; ++i) {
            const f64 tv = t[i];
            out[i] = keel::cos(kTau * (f0 * tv + 0.5 * k * tv * tv) + phi_rad);
        }
        return SignalStatus::kOk;
    }

    if (method == ChirpMethod::kQuadratic) {
        const bool want_convex = (quad_shape == ChirpQuadShape::kConvex);
        const bool want_concave = (quad_shape == ChirpQuadShape::kConcave);
        const bool reverse_vertex = (f1 > f0 && want_convex) || (f1 < f0 && want_concave);
        const f64 k = (f1 - f0) / (t1 * t1);
        if (reverse_vertex && n > 0) {
            const f64 t_end = t[n - 1];
            for (usize i = 0; i < n; ++i) {
                const f64 t_flip = t[n - 1 - i];
                const f64 phase =
                    kTau * (f1 * (t_end - t_flip) + (k / 3.0) * t_flip * t_flip * t_flip) + phi_rad;
                out[i] = keel::cos(phase);
            }
        } else {
            for (usize i = 0; i < n; ++i) {
                const f64 tv = t[i];
                const f64 phase = kTau * (f0 * tv + (k / 3.0) * tv * tv * tv) + phi_rad;
                out[i] = keel::cos(phase);
            }
        }
        return SignalStatus::kOk;
    }

    // Logarithmic
    if (f0 <= 0.0 || f1 <= 0.0 || f0 == f1) return SignalStatus::kInvalidArgument;
    const f64 beta = keel::pow(f1 / f0, 1.0 / t1);
    const f64 log_beta = keel::log(beta);
    for (usize i = 0; i < n; ++i) {
        const f64 phase = kTau * f0 * (keel::pow(beta, t[i]) - 1.0) / log_beta + phi_rad;
        out[i] = keel::cos(phase);
    }
    return SignalStatus::kOk;
}

SignalStatus modulate(ArrayRef<f64> x_col_major, usize len, usize cols, f64 fc, f64 fs,
                      ModulationMethod method, bool has_opt, f64 opt,
                      MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (fs <= 0.0) return SignalStatus::kInvalidArgument;
    const usize total = len * cols;
    if (x_col_major.size() < total || out_col_major.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    if (total == 0) return SignalStatus::kOk;

    if (method == ModulationMethod::kAm) {
        for (usize c = 0; c < cols; ++c) {
            for (usize r = 0; r < len; ++r) {
                const f64 t = as<f64>(r) / fs;
                out_col_major[c * len + r] = x_col_major[c * len + r] * keel::cos(kTau * fc * t);
            }
        }
        return SignalStatus::kOk;
    }

    if (method == ModulationMethod::kAmssb) {
        ArenaTemp scratch = keel::get_scratch(alloc);
        Array<c64> h(len, c64(0.0, 0.0), scratch.allocator());
        for (usize c = 0; c < cols; ++c) {
            const ArrayRef<f64> col(x_col_major.data() + c * len, len);
            const SignalStatus st = hilbert(col, h.as_mut_ref(), scratch.allocator());
            if (st != SignalStatus::kOk) return st;
            for (usize r = 0; r < len; ++r) {
                const f64 t = as<f64>(r) / fs;
                const f64 ang = kTau * fc * t;
                out_col_major[c * len + r] =
                    col[r] * keel::cos(ang) + h[r].imag() * keel::sin(ang);
            }
        }
        return SignalStatus::kOk;
    }

    if (method == ModulationMethod::kAmdsbTc) {
        f64 offset = opt;
        if (!has_opt) {
            offset = x_col_major[0];
            for (usize i = 1; i < total; ++i) {
                if (x_col_major[i] < offset) offset = x_col_major[i];
            }
        }
        for (usize c = 0; c < cols; ++c) {
            for (usize r = 0; r < len; ++r) {
                const f64 t = as<f64>(r) / fs;
                out_col_major[c * len + r] =
                    (x_col_major[c * len + r] - offset) * keel::cos(kTau * fc * t);
            }
        }
        return SignalStatus::kOk;
    }

    if (method == ModulationMethod::kFm) {
        f64 kf = opt;
        if (!has_opt) {
            f64 x_max = 0.0;
            for (usize i = 0; i < total; ++i) {
                const f64 v = keel::abs(x_col_major[i]);
                if (v > x_max) x_max = v;
            }
            kf = (x_max > 0.0) ? ((fc / fs) * kTau / x_max) : 0.0;
        }
        for (usize c = 0; c < cols; ++c) {
            f64 cum = 0.0;
            for (usize r = 0; r < len; ++r) {
                cum += x_col_major[c * len + r];
                const f64 t = as<f64>(r) / fs;
                out_col_major[c * len + r] = keel::cos(kTau * fc * t + kf * cum);
            }
        }
        return SignalStatus::kOk;
    }

    // PM
    f64 kp = opt;
    if (!has_opt) {
        f64 x_max = 0.0;
        for (usize i = 0; i < total; ++i) {
            const f64 v = keel::abs(x_col_major[i]);
            if (v > x_max) x_max = v;
        }
        kp = (x_max > 0.0) ? (kPi / x_max) : 0.0;
    }
    for (usize c = 0; c < cols; ++c) {
        for (usize r = 0; r < len; ++r) {
            const f64 t = as<f64>(r) / fs;
            out_col_major[c * len + r] = keel::cos(kTau * fc * t + kp * x_col_major[c * len + r]);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus demod(ArrayRef<f64> y_col_major, usize len, usize cols, f64 fc, f64 fs,
                   ModulationMethod method, bool has_opt, f64 opt, MutArrayRef<f64> out_col_major,
                   Allocator* alloc) {
    if (fs <= 0.0 || method == ModulationMethod::kAmssb) {
        return SignalStatus::kInvalidArgument;
    }
    const usize total = len * cols;
    if (y_col_major.size() < total || out_col_major.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    if (total == 0) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);

    if (method == ModulationMethod::kFm || method == ModulationMethod::kPm) {
        const f64 p1 = has_opt ? opt : 1.0;
        Array<c64> h(len, c64(0.0, 0.0), scratch.allocator());
        Array<f64> ang(len, 0.0, scratch.allocator());
        for (usize c = 0; c < cols; ++c) {
            const ArrayRef<f64> col(y_col_major.data() + c * len, len);
            const SignalStatus st = hilbert(col, h.as_mut_ref(), scratch.allocator());
            if (st != SignalStatus::kOk) return st;
            for (usize r = 0; r < len; ++r) {
                const f64 t = as<f64>(r) / fs;
                const c64 yq = h[r] * keel::polar(1.0, -kTau * fc * t);
                ang[r] = keel::arg(yq);
            }
            if (method == ModulationMethod::kFm) {
                for (usize r = 1; r < len; ++r) {
                    while (ang[r] - ang[r - 1] > kPi) ang[r] -= kTau;
                    while (ang[r] - ang[r - 1] < -kPi) ang[r] += kTau;
                }
                out_col_major[c * len + 0] = 0.0;
                for (usize r = 1; r < len; ++r) {
                    out_col_major[c * len + r] = (1.0 / p1) * (ang[r] - ang[r - 1]);
                }
            } else {
                for (usize r = 0; r < len; ++r) {
                    out_col_major[c * len + r] = (1.0 / p1) * ang[r];
                }
            }
        }
        return SignalStatus::kOk;
    }

    // AM / AMDSB-SC / AMDSB-TC
    Array<f64> mixed(len, 0.0, scratch.allocator());
    const f64 wn = fc * 2.0 / fs;
    const bool use_lpf = (wn > 0.0 && wn < 1.0);
    TfResult tf;
    if (use_lpf) {
        const SignalStatus st = butter(5, ArrayRef<f64>(&wn, 1), FilterType::kLowpass, false, tf,
                                       scratch.allocator());
        if (st != SignalStatus::kOk) return st;
    }
    const f64 offset = (method == ModulationMethod::kAmdsbTc && has_opt) ? opt : 0.0;

    for (usize c = 0; c < cols; ++c) {
        const f64* ycol = y_col_major.data() + c * len;
        f64* ocol = out_col_major.data() + c * len;
        for (usize r = 0; r < len; ++r) {
            const f64 t = as<f64>(r) / fs;
            mixed[r] = ycol[r] * keel::cos(kTau * fc * t);
        }
        if (use_lpf) {
            const SignalStatus st =
                filtfilt(tf.b.as_ref(), tf.a.as_ref(), mixed.as_ref(), MutArrayRef<f64>(ocol, len),
                         scratch.allocator());
            if (st != SignalStatus::kOk) return st;
        } else {
            for (usize r = 0; r < len; ++r) ocol[r] = mixed[r];
        }
        if (offset != 0.0) {
            for (usize r = 0; r < len; ++r) ocol[r] -= offset;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus vco(ArrayRef<f64> x_col_major, usize len, usize cols, f64 fc, f64 range1, f64 fs,
                 MutArrayRef<f64> out_col_major) noexcept {
    if (fs <= 0.0) return SignalStatus::kInvalidArgument;
    const usize total = len * cols;
    if (x_col_major.size() < total || out_col_major.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < total; ++i) {
        if (x_col_major[i] > 1.0 || x_col_major[i] < -1.0) {
            return SignalStatus::kInvalidArgument;
        }
    }
    for (usize c = 0; c < cols; ++c) {
        f64 cum = 0.0;
        for (usize n = 0; n < len; ++n) {
            const usize idx = c * len + n;
            cum += x_col_major[idx];
            const f64 t = as<f64>(n) / fs;
            out_col_major[idx] = keel::cos(kTau * fc * t + range1 * cum);
        }
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
