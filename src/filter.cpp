#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

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

} // namespace keel::signal
