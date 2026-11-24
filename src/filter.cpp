#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::signal {

namespace {

void copy_f64(MutArrayRef<f64> dst, ArrayRef<f64> src) noexcept {
    const usize n = src.size();
    for (usize i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

void reverse_f64(MutArrayRef<f64> data) noexcept {
    usize lo = 0;
    usize hi = data.size();
    while (lo + 1 < hi) {
        --hi;
        keel::swap(data[lo], data[hi]);
        ++lo;
    }
}

SignalStatus apply_sos_cascade(ArrayRef<f64> sos, usize num_sections, ArrayRef<f64> src,
                               MutArrayRef<f64> dst, MutArrayRef<f64> scratch_buf,
                               ArrayRef<f64> zi) noexcept {
    const usize n = src.size();
    ArrayRef<f64> current_src = src;
    MutArrayRef<f64> current_dst = dst.slice(0, n);
    MutArrayRef<f64> scratch_view = scratch_buf.slice(0, n);
    const bool has_zi = !zi.empty();

    for (usize s = 0; s < num_sections; ++s) {
        const f64 a0 = sos[3 * num_sections + s];
        if (a0 == 0.0) {
            return SignalStatus::kZeroLeadingCoeff;
        }
        const f64 b0 = sos[0 * num_sections + s] / a0;
        const f64 b1 = sos[1 * num_sections + s] / a0;
        const f64 b2 = sos[2 * num_sections + s] / a0;
        const f64 a1 = sos[4 * num_sections + s] / a0;
        const f64 a2 = sos[5 * num_sections + s] / a0;

        SignalStatus st = SignalStatus::kOk;
        if (has_zi) {
            st = biquad_df2t_state(b0, b1, b2, a1, a2, current_src, current_dst, zi[2 * s + 0],
                                   zi[2 * s + 1]);
        } else {
            st = biquad_df2t(b0, b1, b2, a1, a2, current_src, current_dst);
        }
        if (st != SignalStatus::kOk) return st;

        if (s + 1 < num_sections) {
            if (current_dst.data() == dst.data()) {
                current_src = dst.slice(0, n);
                current_dst = scratch_view;
            } else {
                current_src = scratch_view;
                current_dst = dst.slice(0, n);
            }
        }
    }
    if (current_dst.data() != dst.data()) {
        copy_f64(dst, scratch_view);
    }
    return SignalStatus::kOk;
}

} // namespace

SignalStatus filter_df2t(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> x, MutArrayRef<f64> y,
                         ArrayRef<f64> zi, MutArrayRef<f64> zf, Allocator* alloc) {
    const usize len = x.size();
    if (y.size() < len) {
        return SignalStatus::kInvalidArgument;
    }
    const usize nb = b.size();
    const usize na = a.size();
    const usize nfilt = keel::max(nb, na);
    if (nfilt == 0) {
        for (usize n = 0; n < len; ++n) {
            y[n] = 0.0;
        }
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> z(nfilt, 0.0, scratch.allocator());
    const usize zi_len = zi.size();
    for (usize i = 0; i < nfilt; ++i) {
        z[i] = (i < zi_len) ? zi[i] : 0.0;
    }
    for (usize n = 0; n < len; ++n) {
        const f64 xn = x[n];
        const f64 yn = (nb > 0 ? b[0] : 0.0) * xn + z[0];
        y[n] = yn;
        for (usize i = 1; i < nfilt; ++i) {
            z[i - 1] = (i < nb ? b[i] : 0.0) * xn - (i < na ? a[i] : 0.0) * yn +
                       (i < nfilt - 1 ? z[i] : 0.0);
        }
    }
    if (!zf.empty() && nfilt > 1) {
        const usize count = keel::min(zf.size(), nfilt - 1);
        for (usize i = 0; i < count; ++i) {
            zf[i] = z[i];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus filter_df2t_c64(ArrayRef<c64> b, ArrayRef<c64> a, ArrayRef<c64> x, MutArrayRef<c64> y,
                             ArrayRef<c64> zi, MutArrayRef<c64> zf, Allocator* alloc) {
    const usize len = x.size();
    if (y.size() < len) {
        return SignalStatus::kInvalidArgument;
    }
    const usize nb = b.size();
    const usize na = a.size();
    const usize nfilt = keel::max(nb, na);
    const c64 zero(0.0, 0.0);
    if (nfilt == 0) {
        for (usize n = 0; n < len; ++n) {
            y[n] = zero;
        }
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> z(nfilt, zero, scratch.allocator());
    const usize zi_len = zi.size();
    for (usize i = 0; i < nfilt; ++i) {
        z[i] = (i < zi_len) ? zi[i] : zero;
    }
    for (usize n = 0; n < len; ++n) {
        const c64 xn = x[n];
        const c64 yn = (nb > 0 ? b[0] : zero) * xn + z[0];
        y[n] = yn;
        for (usize i = 1; i < nfilt; ++i) {
            z[i - 1] = (i < nb ? b[i] : zero) * xn - (i < na ? a[i] : zero) * yn +
                       (i < nfilt - 1 ? z[i] : zero);
        }
    }
    if (!zf.empty() && nfilt > 1) {
        const usize count = keel::min(zf.size(), nfilt - 1);
        for (usize i = 0; i < count; ++i) {
            zf[i] = z[i];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus biquad_df2t(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2, ArrayRef<f64> x,
                         MutArrayRef<f64> y) noexcept {
    const usize n = x.size();
    if (y.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    f64 s1 = 0.0;
    f64 s2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 xi = x[i];
        const f64 yi = b0 * xi + s1;
        s1 = b1 * xi - a1 * yi + s2;
        s2 = b2 * xi - a2 * yi;
        y[i] = yi;
    }
    return SignalStatus::kOk;
}

SignalStatus biquad_df2t_state(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2, ArrayRef<f64> x,
                               MutArrayRef<f64> y, f64 s1_init, f64 s2_init,
                               BiquadState* final_state) noexcept {
    const usize n = x.size();
    if (y.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    f64 s1 = s1_init;
    f64 s2 = s2_init;
    for (usize i = 0; i < n; ++i) {
        const f64 xi = x[i];
        const f64 yi = b0 * xi + s1;
        s1 = b1 * xi - a1 * yi + s2;
        s2 = b2 * xi - a2 * yi;
        y[i] = yi;
    }
    if (final_state) {
        final_state->s1 = s1;
        final_state->s2 = s2;
    }
    return SignalStatus::kOk;
}

BiquadState biquad_zi(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2) noexcept {
    const f64 denom = 1.0 + a1 + a2;
    if (keel::abs(denom) < 1e-300) {
        return BiquadState{0.0, 0.0};
    }
    const f64 s1 = (b1 + b2 - b0 * (a1 + a2)) / denom;
    const f64 s2 = (b2 + a1 * b2 - a2 * b0 - a2 * b1) / denom;
    return BiquadState{s1, s2};
}

SignalStatus sosfilt_col_major(ArrayRef<f64> sos, usize num_sections, ArrayRef<f64> x,
                               MutArrayRef<f64> y, ArrayRef<f64> zi, Allocator* alloc) {
    const usize n = x.size();
    if (y.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (num_sections == 0) {
        copy_f64(y, x);
        return SignalStatus::kOk;
    }
    if (sos.size() < 6 * num_sections) {
        return SignalStatus::kInvalidArgument;
    }
    if (!zi.empty() && zi.size() < 2 * num_sections) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> buf(n, scratch.allocator());
    return apply_sos_cascade(sos, num_sections, x, y, buf, zi);
}

SignalStatus sosfiltfilt_col_major(ArrayRef<f64> sos, usize num_sections, ArrayRef<f64> x,
                                   MutArrayRef<f64> y, Allocator* alloc) {
    const usize n = x.size();
    if (y.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (num_sections == 0) {
        copy_f64(y, x);
        return SignalStatus::kOk;
    }
    if (sos.size() < 6 * num_sections) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp local = keel::get_scratch(alloc);
    Array<f64> zi_vec(2 * num_sections, local.allocator());
    Array<f64> dc_gain(num_sections, local.allocator());

    for (usize s = 0; s < num_sections; ++s) {
        const f64 a0 = sos[3 * num_sections + s];
        if (a0 == 0.0) {
            return SignalStatus::kZeroLeadingCoeff;
        }
        const f64 b0 = sos[0 * num_sections + s] / a0;
        const f64 b1 = sos[1 * num_sections + s] / a0;
        const f64 b2 = sos[2 * num_sections + s] / a0;
        const f64 a1 = sos[4 * num_sections + s] / a0;
        const f64 a2 = sos[5 * num_sections + s] / a0;

        const BiquadState z = biquad_zi(b0, b1, b2, a1, a2);
        zi_vec[2 * s + 0] = z.s1;
        zi_vec[2 * s + 1] = z.s2;

        const f64 num = b0 + b1 + b2;
        const f64 den = 1.0 + a1 + a2;
        dc_gain[s] = (keel::abs(den) > 1e-300) ? (num / den) : 1.0;
    }

    usize edge = 6 * num_sections;
    if (edge >= n) {
        edge = n - 1;
    }
    const usize ext_len = n + 2 * edge;

    Array<f64> ext(ext_len, local.allocator());
    for (usize i = 0; i < edge; ++i) {
        ext[i] = 2.0 * x[0] - x[edge - i];
    }
    copy_f64(MutArrayRef<f64>(ext).slice(edge, n), x);
    for (usize i = 0; i < edge; ++i) {
        ext[edge + n + i] = 2.0 * x[n - 1] - x[n - 2 - i];
    }

    Array<f64> fwd(ext_len, local.allocator());
    Array<f64> bwd(ext_len, local.allocator());
    Array<f64> cascade_scratch(ext_len, local.allocator());
    Array<f64> zi_scaled(2 * num_sections, local.allocator());

    auto scale_zi = [&](f64 cascade_input) noexcept {
        f64 scale = cascade_input;
        for (usize s = 0; s < num_sections; ++s) {
            zi_scaled[2 * s + 0] = zi_vec[2 * s + 0] * scale;
            zi_scaled[2 * s + 1] = zi_vec[2 * s + 1] * scale;
            scale *= dc_gain[s];
        }
    };

    scale_zi(ext[0]);
    SignalStatus st = apply_sos_cascade(sos, num_sections, ext, fwd, cascade_scratch, zi_scaled);
    if (st != SignalStatus::kOk) return st;

    reverse_f64(fwd);
    scale_zi(fwd[0]);
    st = apply_sos_cascade(sos, num_sections, fwd, bwd, cascade_scratch, zi_scaled);
    if (st != SignalStatus::kOk) return st;

    reverse_f64(bwd);
    copy_f64(y, ArrayRef<f64>(bwd).slice(edge, n));
    return SignalStatus::kOk;
}

SignalStatus filtfilt(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> x, MutArrayRef<f64> y,
                      Allocator* alloc) {
    const usize nb = b.size();
    const usize na = a.size();
    const usize nx = x.size();
    if (y.size() < nx) {
        return SignalStatus::kInvalidArgument;
    }
    if (na == 0 || a[0] == 0.0) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (nx == 0) {
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 a0 = a[0];
    Array<f64> bn(nb, scratch.allocator());
    Array<f64> an(na, scratch.allocator());
    for (usize i = 0; i < nb; ++i) bn[i] = b[i] / a0;
    for (usize i = 0; i < na; ++i) an[i] = a[i] / a0;

    const usize nfilt = keel::max(nb, na);
    const usize state_len = (nfilt > 0) ? (nfilt - 1) : 0;

    Array<f64> zi(state_len, 0.0, scratch.allocator());
    if (state_len > 0) {
        Array<f64> rhs(state_len, 0.0, scratch.allocator());
        f64 sum_rhs = 0.0;
        f64 sum_a = 1.0;
        const f64 b0 = (nb > 0) ? bn[0] : 0.0;
        for (usize k = 0; k < state_len; ++k) {
            const f64 bk1 = (k + 1 < nb) ? bn[k + 1] : 0.0;
            const f64 ak1 = (k + 1 < na) ? an[k + 1] : 0.0;
            rhs[k] = bk1 - b0 * ak1;
            sum_rhs += rhs[k];
            sum_a += ak1;
        }
        if (keel::abs(sum_a) > 1e-14) {
            zi[0] = sum_rhs / sum_a;
            for (usize k = 0; k + 1 < state_len; ++k) {
                const f64 ak1 = (k + 1 < na) ? an[k + 1] : 0.0;
                zi[k + 1] = zi[k] + ak1 * zi[0] - rhs[k];
            }
        }
    }

    usize n_edge = 3 * state_len;
    if (n_edge >= nx) {
        n_edge = nx - 1;
    }
    const usize ext_len = nx + 2 * n_edge;
    Array<f64> ext(ext_len, scratch.allocator());
    for (usize i = 0; i < n_edge; ++i) {
        ext[i] = 2.0 * x[0] - x[n_edge - i];
    }
    for (usize i = 0; i < nx; ++i) {
        ext[n_edge + i] = x[i];
    }
    for (usize i = 0; i < n_edge; ++i) {
        ext[n_edge + nx + i] = 2.0 * x[nx - 1] - x[nx - 2 - i];
    }

    Array<f64> zi_scaled(state_len, 0.0, scratch.allocator());
    for (usize k = 0; k < state_len; ++k) {
        zi_scaled[k] = zi[k] * ext[0];
    }
    Array<f64> fwd(ext_len, scratch.allocator());
    SignalStatus st = filter_df2t(bn.as_ref(), an.as_ref(), ext.as_ref(), fwd.as_mut_ref(),
                                  zi_scaled.as_ref(), {}, scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    reverse_f64(fwd.as_mut_ref());
    for (usize k = 0; k < state_len; ++k) {
        zi_scaled[k] = zi[k] * fwd[0];
    }
    Array<f64> bwd(ext_len, scratch.allocator());
    st = filter_df2t(bn.as_ref(), an.as_ref(), fwd.as_ref(), bwd.as_mut_ref(),
                     zi_scaled.as_ref(), {}, scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    reverse_f64(bwd.as_mut_ref());
    copy_f64(y, ArrayRef<f64>(bwd).slice(n_edge, nx));
    return SignalStatus::kOk;
}

namespace {

inline c64 eval_freqz_real_at(ArrayRef<f64> b, ArrayRef<f64> a, f64 w) noexcept {
    const c64 ejw(keel::cos(w), -keel::sin(w));
    c64 num(0.0, 0.0);
    c64 den(0.0, 0.0);
    c64 ejwk(1.0, 0.0);
    for (usize i = 0; i < b.size(); ++i) {
        num += c64(b[i], 0.0) * ejwk;
        ejwk *= ejw;
    }
    ejwk = c64(1.0, 0.0);
    for (usize i = 0; i < a.size(); ++i) {
        den += c64(a[i], 0.0) * ejwk;
        ejwk *= ejw;
    }
    return num / den;
}

inline c64 eval_freqz_c64_at(ArrayRef<c64> b, ArrayRef<c64> a, f64 w) noexcept {
    const c64 ejw(keel::cos(w), -keel::sin(w));
    c64 num(0.0, 0.0);
    c64 den(0.0, 0.0);
    c64 ejwk(1.0, 0.0);
    for (usize i = 0; i < b.size(); ++i) {
        num += b[i] * ejwk;
        ejwk *= ejw;
    }
    ejwk = c64(1.0, 0.0);
    for (usize i = 0; i < a.size(); ++i) {
        den += a[i] * ejwk;
        ejwk *= ejw;
    }
    return num / den;
}

void unwrap_phase_pi(MutArrayRef<f64> p) noexcept {
    const usize n = p.size();
    if (n < 2) return;
    f64 offset = 0.0;
    for (usize i = 1; i < n; ++i) {
        const f64 diff = p[i] + offset - p[i - 1];
        if (diff > kPi) {
            offset -= kTau;
        } else if (diff < -kPi) {
            offset += kTau;
        }
        p[i] += offset;
    }
}

bool is_trivial_denom(ArrayRef<f64> a) noexcept {
    if (a.empty()) return true;
    if (keel::abs(a[0] - 1.0) > 1e-12) return false;
    for (usize i = 1; i < a.size(); ++i) {
        if (keel::abs(a[i]) > 1e-12) return false;
    }
    return true;
}

} // namespace

SignalStatus freqz(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> w,
                   MutArrayRef<c64> h_out) noexcept {
    const usize n = w.size();
    if (h_out.size() < n) return SignalStatus::kInvalidArgument;
    for (usize k = 0; k < n; ++k) {
        h_out[k] = eval_freqz_real_at(b, a, w[k]);
    }
    return SignalStatus::kOk;
}

SignalStatus freqz_c64(ArrayRef<c64> b, ArrayRef<c64> a, ArrayRef<f64> w,
                       MutArrayRef<c64> h_out) noexcept {
    const usize n = w.size();
    if (h_out.size() < n) return SignalStatus::kInvalidArgument;
    for (usize k = 0; k < n; ++k) {
        h_out[k] = eval_freqz_c64_at(b, a, w[k]);
    }
    return SignalStatus::kOk;
}

SignalStatus freqz_grid(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, bool whole, f64 fs,
                        MutArrayRef<c64> h_out, MutArrayRef<f64> freq_out) noexcept {
    if (h_out.size() < npts || (!freq_out.empty() && freq_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    if (npts == 0) return SignalStatus::kOk;
    const f64 span = whole ? kTau : kPi;
    const f64 hz_span = whole ? fs : (0.5 * fs);
    const f64 inv_n = 1.0 / as<f64>(npts);
    for (usize k = 0; k < npts; ++k) {
        const f64 fk = as<f64>(k);
        const f64 w = span * fk * inv_n;
        if (!freq_out.empty()) {
            freq_out[k] = (fs > 0.0) ? (hz_span * fk * inv_n) : w;
        }
        h_out[k] = eval_freqz_real_at(b, a, w);
    }
    return SignalStatus::kOk;
}

SignalStatus freqz_grid_c64(ArrayRef<c64> b, ArrayRef<c64> a, usize npts, bool whole, f64 fs,
                            MutArrayRef<c64> h_out, MutArrayRef<f64> freq_out) noexcept {
    if (h_out.size() < npts || (!freq_out.empty() && freq_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    if (npts == 0) return SignalStatus::kOk;
    const f64 span = whole ? kTau : kPi;
    const f64 hz_span = whole ? fs : (0.5 * fs);
    const f64 inv_n = 1.0 / as<f64>(npts);
    for (usize k = 0; k < npts; ++k) {
        const f64 fk = as<f64>(k);
        const f64 w = span * fk * inv_n;
        if (!freq_out.empty()) {
            freq_out[k] = (fs > 0.0) ? (hz_span * fk * inv_n) : w;
        }
        h_out[k] = eval_freqz_c64_at(b, a, w);
    }
    return SignalStatus::kOk;
}

SignalStatus phasez(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, f64 fs,
                    MutArrayRef<f64> phi_out, MutArrayRef<f64> freq_out) noexcept {
    if (phi_out.size() < npts || (!freq_out.empty() && freq_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    if (npts == 0) return SignalStatus::kOk;
    const f64 hz_span = 0.5 * fs;
    const f64 inv_n = 1.0 / as<f64>(npts);
    constexpr f64 kZeroEps = 1e-300;
    for (usize k = 0; k < npts; ++k) {
        const f64 fk = as<f64>(k);
        const f64 w = kPi * fk * inv_n;
        if (!freq_out.empty()) {
            freq_out[k] = (fs > 0.0) ? (hz_span * fk * inv_n) : w;
        }
        const c64 h = eval_freqz_real_at(b, a, w);
        const f64 mag2 = h.real() * h.real() + h.imag() * h.imag();
        phi_out[k] = (mag2 < kZeroEps) ? Limits<f64>::quiet_nan() : keel::atan2(h.imag(), h.real());
    }
    unwrap_phase_pi(phi_out.slice(0, npts));
    return SignalStatus::kOk;
}

SignalStatus phasez_c64(ArrayRef<c64> b, ArrayRef<c64> a, usize npts, f64 fs,
                        MutArrayRef<f64> phi_out, MutArrayRef<f64> freq_out) noexcept {
    if (phi_out.size() < npts || (!freq_out.empty() && freq_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    if (npts == 0) return SignalStatus::kOk;
    const f64 hz_span = 0.5 * fs;
    const f64 inv_n = 1.0 / as<f64>(npts);
    constexpr f64 kZeroEps = 1e-300;
    for (usize k = 0; k < npts; ++k) {
        const f64 fk = as<f64>(k);
        const f64 w = kPi * fk * inv_n;
        if (!freq_out.empty()) {
            freq_out[k] = (fs > 0.0) ? (hz_span * fk * inv_n) : w;
        }
        const c64 h = eval_freqz_c64_at(b, a, w);
        const f64 mag2 = h.real() * h.real() + h.imag() * h.imag();
        phi_out[k] = (mag2 < kZeroEps) ? Limits<f64>::quiet_nan() : keel::atan2(h.imag(), h.real());
    }
    unwrap_phase_pi(phi_out.slice(0, npts));
    return SignalStatus::kOk;
}

SignalStatus grpdelay(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, f64 fs,
                      MutArrayRef<f64> gd_out, MutArrayRef<f64> freq_out, Allocator* alloc) {
    if (gd_out.size() < npts || (!freq_out.empty() && freq_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    const usize nb = b.size();
    const usize na = a.size();
    if (npts == 0 || nb == 0 || na == 0) {
        for (usize k = 0; k < npts; ++k) {
            gd_out[k] = 0.0;
            if (!freq_out.empty()) freq_out[k] = 0.0;
        }
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize nc = nb + na - 1;
    Array<f64> c(nc, 0.0, scratch.allocator());
    for (usize i = 0; i < nb; ++i) {
        for (usize j = 0; j < na; ++j) {
            c[i + j] += b[i] * a[na - 1 - j];
        }
    }

    const f64 offset = as<f64>(na) - 1.0;
    const f64 hz_span = 0.5 * fs;
    const f64 inv_n = 1.0 / as<f64>(npts);
    for (usize k = 0; k < npts; ++k) {
        const f64 fk = as<f64>(k);
        const f64 w = kPi * fk * inv_n;
        if (!freq_out.empty()) {
            freq_out[k] = (fs > 0.0) ? (hz_span * fk * inv_n) : w;
        }
        const c64 ejw(keel::cos(w), -keel::sin(w));
        c64 C(0.0, 0.0);
        c64 CR(0.0, 0.0);
        c64 ejwk(1.0, 0.0);
        for (usize n = 0; n < nc; ++n) {
            C += c[n] * ejwk;
            CR += (as<f64>(n) * c[n]) * ejwk;
            ejwk *= ejw;
        }
        const f64 cmag2 = C.real() * C.real() + C.imag() * C.imag();
        gd_out[k] = (cmag2 < 1e-300) ? 0.0 : ((CR / C).real() - offset);
    }
    return SignalStatus::kOk;
}

SignalStatus impzlength(ArrayRef<f64> b, ArrayRef<f64> a, usize& out_len, Allocator* alloc) {
    if (is_trivial_denom(a)) {
        out_len = keel::max<usize>(b.size(), 1);
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    f64 rho = 0.0;
    if (a.size() >= 2) {
        const Array<c64> r = keel::poly::roots(a, scratch.allocator());
        for (usize i = 0; i < r.size(); ++i) {
            rho = keel::max(rho, keel::abs(r[i]));
        }
    }
    if (!(rho > 0.0)) {
        out_len = keel::max<usize>(b.size(), 1);
        return SignalStatus::kOk;
    }
    if (rho >= 1.0) {
        out_len = 8192;
        return SignalStatus::kOk;
    }
    constexpr f64 kDecayThresh = 5e-5;
    const f64 n_est = keel::log(kDecayThresh) / keel::log(rho);
    i64 n_int = as<i64>(keel::floor(n_est));
    if (n_int < 1) n_int = 1;
    if (n_int > 8192) n_int = 8192;
    out_len = as<usize>(n_int);
    return SignalStatus::kOk;
}

SignalStatus impz(ArrayRef<f64> b, ArrayRef<f64> a, usize n, MutArrayRef<f64> h_out,
                  MutArrayRef<f64> t_out, Allocator* alloc) {
    if (h_out.size() < n || (!t_out.empty() && t_out.size() < n)) {
        return SignalStatus::kInvalidArgument;
    }
    if (a.empty() || a[0] == 0.0) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (n == 0) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> bn(b.size(), scratch.allocator());
    Array<f64> an(a.size(), scratch.allocator());
    const f64 a0 = a[0];
    for (usize i = 0; i < b.size(); ++i) bn[i] = b[i] / a0;
    for (usize i = 0; i < a.size(); ++i) an[i] = a[i] / a0;

    Array<f64> x(n, 0.0, scratch.allocator());
    x[0] = 1.0;
    const SignalStatus st =
        filter_df2t(bn.as_ref(), an.as_ref(), x.as_ref(), h_out.slice(0, n), {}, {},
                    scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    if (!t_out.empty()) {
        for (usize i = 0; i < n; ++i) t_out[i] = as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus stepz(ArrayRef<f64> b, ArrayRef<f64> a, usize n, MutArrayRef<f64> s_out,
                   MutArrayRef<f64> t_out, Allocator* alloc) {
    if (s_out.size() < n || (!t_out.empty() && t_out.size() < n)) {
        return SignalStatus::kInvalidArgument;
    }
    if (a.empty() || a[0] == 0.0) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (n == 0) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> bn(b.size(), scratch.allocator());
    Array<f64> an(a.size(), scratch.allocator());
    const f64 a0 = a[0];
    for (usize i = 0; i < b.size(); ++i) bn[i] = b[i] / a0;
    for (usize i = 0; i < a.size(); ++i) an[i] = a[i] / a0;

    Array<f64> x(n, 1.0, scratch.allocator());
    const SignalStatus st =
        filter_df2t(bn.as_ref(), an.as_ref(), x.as_ref(), s_out.slice(0, n), {}, {},
                    scratch.allocator());
    if (st != SignalStatus::kOk) return st;
    if (!t_out.empty()) {
        for (usize i = 0; i < n; ++i) t_out[i] = as<f64>(i);
    }
    return SignalStatus::kOk;
}

SignalStatus phasedelay(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, MutArrayRef<f64> pd_out,
                        MutArrayRef<f64> w_out) noexcept {
    if (pd_out.size() < npts || w_out.size() < npts) {
        return SignalStatus::kInvalidArgument;
    }
    const SignalStatus st = phasez(b, a, npts, 0.0, pd_out, w_out);
    if (st != SignalStatus::kOk) return st;
    const f64 nan_val = Limits<f64>::quiet_nan();
    for (usize i = 0; i < npts; ++i) {
        pd_out[i] = (w_out[i] == 0.0) ? nan_val : (-pd_out[i] / w_out[i]);
    }
    return SignalStatus::kOk;
}

SignalStatus zerophase(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, MutArrayRef<f64> hr_out,
                       MutArrayRef<f64> w_out) noexcept {
    if (hr_out.size() < npts || (!w_out.empty() && w_out.size() < npts)) {
        return SignalStatus::kInvalidArgument;
    }
    if (npts == 0) return SignalStatus::kOk;
    const f64 tau = (b.size() > 0) ? (as<f64>(b.size()) - 1.0) * 0.5 : 0.0;
    const f64 inv_n = 1.0 / as<f64>(npts);
    for (usize k = 0; k < npts; ++k) {
        const f64 w = kPi * as<f64>(k) * inv_n;
        if (!w_out.empty()) w_out[k] = w;
        const c64 h = eval_freqz_real_at(b, a, w);
        const c64 twist = keel::polar(1.0, w * tau);
        hr_out[k] = (h * twist).real();
    }
    return SignalStatus::kOk;
}

SignalStatus polyscale(ArrayRef<f64> p_col_major, usize rows, usize cols, ArrayRef<f64> scale,
                       MutArrayRef<f64> out_col_major, Allocator* alloc) {
    if (rows == 0 || cols == 0) return SignalStatus::kOk;
    const usize total = rows * cols;
    if (p_col_major.size() < total || out_col_major.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    const bool scalar_s = (scale.size() == 1);
    if (!scalar_s && scale.size() != cols) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> factor(cols, 1.0, scratch.allocator());
    if (scalar_s) {
        const f64 a = scale[0];
        for (usize k = 1; k < cols; ++k) {
            factor[k] = factor[k - 1] * a;
        }
    } else {
        for (usize k = 1; k < cols; ++k) {
            const f64 base = scale[k];
            f64 acc = 1.0;
            for (usize e = 0; e < k; ++e) acc *= base;
            factor[k] = acc;
        }
    }
    for (usize k = 0; k < cols; ++k) {
        const f64 fk = factor[k];
        for (usize i = 0; i < rows; ++i) {
            const usize lin = k * rows + i;
            out_col_major[lin] = p_col_major[lin] * fk;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus polyscale_c64(ArrayRef<c64> p_col_major, usize rows, usize cols, ArrayRef<c64> scale,
                           MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (rows == 0 || cols == 0) return SignalStatus::kOk;
    const usize total = rows * cols;
    if (p_col_major.size() < total || out_col_major.size() < total) {
        return SignalStatus::kInvalidArgument;
    }
    const bool scalar_s = (scale.size() == 1);
    if (!scalar_s && scale.size() != cols) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> factor(cols, c64(1.0, 0.0), scratch.allocator());
    if (scalar_s) {
        const c64 a = scale[0];
        for (usize k = 1; k < cols; ++k) {
            factor[k] = factor[k - 1] * a;
        }
    } else {
        for (usize k = 1; k < cols; ++k) {
            const c64 base = scale[k];
            c64 acc(1.0, 0.0);
            for (usize e = 0; e < k; ++e) acc *= base;
            factor[k] = acc;
        }
    }
    for (usize k = 0; k < cols; ++k) {
        const c64 fk = factor[k];
        for (usize i = 0; i < rows; ++i) {
            const usize lin = k * rows + i;
            out_col_major[lin] = p_col_major[lin] * fk;
        }
    }
    return SignalStatus::kOk;
}

SignalStatus polystab(ArrayRef<f64> a, Array<f64>& out, Allocator* alloc) {
    const usize n = a.size();
    out = Array<f64>(alloc);
    if (n == 0) return SignalStatus::kOk;
    if (n == 1) {
        out.push_back(a[0]);
        return SignalStatus::kOk;
    }
    f64 gain = 0.0;
    bool found = false;
    for (usize k = 0; k < n; ++k) {
        if (a[k] != 0.0) {
            gain = a[k];
            found = true;
            break;
        }
    }
    if (!found) {
        out.resize(n);
        for (usize k = 0; k < n; ++k) out[k] = 0.0;
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> r = keel::poly::roots(a, scratch.allocator());
    if (r.empty()) {
        out.push_back(gain);
        return SignalStatus::kOk;
    }
    for (usize k = 0; k < r.size(); ++k) {
        const f64 mag = keel::abs(r[k]);
        if (mag > 1.0) {
            r[k] = r[k] / (mag * mag);
        }
    }
    Array<f64> monic = keel::poly::from_roots(r.as_ref(), scratch.allocator());
    out.resize(monic.size());
    for (usize k = 0; k < monic.size(); ++k) {
        out[k] = monic[k] * gain;
    }
    return SignalStatus::kOk;
}

SignalStatus polystab_c64(ArrayRef<c64> a, Array<c64>& out, Allocator* alloc) {
    const usize n = a.size();
    out = Array<c64>(alloc);
    if (n == 0) return SignalStatus::kOk;
    if (n == 1) {
        out.push_back(a[0]);
        return SignalStatus::kOk;
    }
    c64 gain(0.0, 0.0);
    bool found = false;
    for (usize k = 0; k < n; ++k) {
        if (a[k] != c64(0.0, 0.0)) {
            gain = a[k];
            found = true;
            break;
        }
    }
    if (!found) {
        out.resize(n, c64(0.0, 0.0));
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> r = keel::poly::roots(a, scratch.allocator());
    if (r.empty()) {
        out.push_back(gain);
        return SignalStatus::kOk;
    }
    for (usize k = 0; k < r.size(); ++k) {
        const f64 mag = keel::abs(r[k]);
        if (mag > 1.0) {
            r[k] = r[k] / (mag * mag);
        }
    }
    Array<c64> monic = keel::poly::from_roots_c64(r.as_ref(), scratch.allocator());
    out.resize(monic.size());
    for (usize k = 0; k < monic.size(); ++k) {
        out[k] = monic[k] * gain;
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
