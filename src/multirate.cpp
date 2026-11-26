#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

constexpr usize next_pow2(usize n) noexcept {
    if (n <= 1) return 1;
    usize p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace

SignalStatus upfirdn(ArrayRef<f64> x, ArrayRef<f64> h, usize p, usize q, MutArrayRef<f64> out,
                     Allocator* alloc) {
    if (p == 0 || q == 0) {
        return SignalStatus::kInvalidArgument;
    }
    const usize lx = x.size();
    const usize lh = h.size();
    if (lx == 0 || lh == 0) {
        return SignalStatus::kOk;
    }
    const usize out_len = upfirdn_length(lx, lh, p, q);
    if (out.size() < out_len) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> x_copy(scratch.allocator());
    const f64* x_base = x.data();
    if (out.data() < x_base + lx && x_base < out.data() + out_len) {
        x_copy = Array<f64>::from(x, scratch.allocator());
        x_base = x_copy.data();
    }

    // Polyphase reversed-subfilter decomposition: pack each phase's taps in
    // reverse order so every output sample reduces to a contiguous unit-stride
    // dot product without allocating or computing dropped samples.
    Array<usize> sub_off(p, scratch.allocator());
    Array<usize> sub_len(p, scratch.allocator());
    usize total_packed = 0;
    for (usize ph = 0; ph < p; ++ph) {
        sub_off[ph] = total_packed;
        const usize len = (ph < lh) ? ((lh - 1 - ph) / p + 1) : 0;
        sub_len[ph] = len;
        total_packed += len;
    }
    Array<f64> h_poly(total_packed, scratch.allocator());
    for (usize ph = 0; ph < p; ++ph) {
        const usize len = sub_len[ph];
        f64* hp = h_poly.data() + sub_off[ph];
        for (usize u = 0; u < len; ++u) {
            hp[u] = h[ph + (len - 1 - u) * p];
        }
    }

    const i64 ilx = as<i64>(lx);
    for (usize m = 0; m < out_len; ++m) {
        const usize n = m * q;
        const usize ph = n % p;
        const usize len = sub_len[ph];
        if (len == 0) {
            out[m] = 0.0;
            continue;
        }
        const i64 k_hi = as<i64>(n / p);
        const i64 k_start = k_hi - as<i64>(len - 1);
        const i64 u_lo = (k_start < 0) ? -k_start : 0;
        const i64 u_hi = min(as<i64>(len), ilx - k_start);
        if (u_lo >= u_hi) {
            out[m] = 0.0;
            continue;
        }
        const f64* xp = x_base + (k_start + u_lo);
        const f64* hp = h_poly.data() + sub_off[ph] + u_lo;
        const usize cnt = as<usize>(u_hi - u_lo);
        f64 a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        usize u = 0;
        for (; u + 4 <= cnt; u += 4) {
            a0 += xp[u] * hp[u];
            a1 += xp[u + 1] * hp[u + 1];
            a2 += xp[u + 2] * hp[u + 2];
            a3 += xp[u + 3] * hp[u + 3];
        }
        f64 acc = (a0 + a1) + (a2 + a3);
        for (; u < cnt; ++u) {
            acc += xp[u] * hp[u];
        }
        out[m] = acc;
    }
    return SignalStatus::kOk;
}

SignalStatus fftfilt(ArrayRef<f64> b, ArrayRef<f64> x, usize nfft, MutArrayRef<f64> out,
                     Allocator* alloc) {
    const usize nb = b.size();
    const usize nx = x.size();
    if (out.size() < nx) {
        return SignalStatus::kInvalidArgument;
    }
    if (nb == 0 || nx == 0) {
        for (usize i = 0; i < nx; ++i) out[i] = 0.0;
        return SignalStatus::kOk;
    }

    usize block_l = 0;
    if (nfft >= nb) {
        block_l = nfft - nb + 1;
    } else {
        block_l = keel::max(nb, usize{32});
        nfft = next_pow2(block_l + nb - 1);
        block_l = nfft - nb + 1;
    }
    if (block_l < 1) block_l = 1;
    if (nfft < block_l + nb - 1) {
        nfft = next_pow2(block_l + nb - 1);
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> x_copy(scratch.allocator());
    ArrayRef<f64> x_src = x;
    if (out.data() < x.data() + nx && x.data() < out.data() + nx) {
        x_copy = Array<f64>::from(x, scratch.allocator());
        x_src = x_copy.as_ref();
    }

    const usize n_half = nfft / 2 + 1;
    Array<f64> b_pad(nfft, 0.0, scratch.allocator());
    for (usize i = 0; i < nb; ++i) b_pad[i] = b[i];
    Array<c64> bf(n_half, c64(0.0, 0.0), scratch.allocator());
    keel::fft::rfft_1d(b_pad.as_ref(), bf.as_mut_ref());

    Array<f64> x_pad(nfft, 0.0, scratch.allocator());
    Array<c64> xf(n_half, c64(0.0, 0.0), scratch.allocator());
    Array<f64> y_blk(nfft, 0.0, scratch.allocator());

    for (usize i = 0; i < nx; ++i) out[i] = 0.0;

    for (usize pos = 0; pos < nx; pos += block_l) {
        const usize l = keel::min(block_l, nx - pos);
        for (usize i = 0; i < l; ++i) x_pad[i] = x_src[pos + i];
        for (usize i = l; i < nfft; ++i) x_pad[i] = 0.0;
        keel::fft::rfft_1d(x_pad.as_ref(), xf.as_mut_ref());
        for (usize i = 0; i < n_half; ++i) xf[i] *= bf[i];
        keel::fft::irfft_1d(xf.as_ref(), y_blk.as_mut_ref(), nfft);
        const usize blk_out = l + nb - 1;
        for (usize i = 0; i < blk_out && pos + i < nx; ++i) {
            out[pos + i] += y_blk[i];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus fftfilt_c64(ArrayRef<c64> b, ArrayRef<c64> x, usize nfft, MutArrayRef<c64> out,
                         Allocator* alloc) {
    const usize nb = b.size();
    const usize nx = x.size();
    if (out.size() < nx) {
        return SignalStatus::kInvalidArgument;
    }
    if (nb == 0 || nx == 0) {
        for (usize i = 0; i < nx; ++i) out[i] = c64(0.0, 0.0);
        return SignalStatus::kOk;
    }

    usize block_l = 0;
    if (nfft >= nb) {
        block_l = nfft - nb + 1;
    } else {
        block_l = keel::max(nb, usize{32});
        nfft = next_pow2(block_l + nb - 1);
        block_l = nfft - nb + 1;
    }
    if (block_l < 1) block_l = 1;
    if (nfft < block_l + nb - 1) {
        nfft = next_pow2(block_l + nb - 1);
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> x_copy(scratch.allocator());
    ArrayRef<c64> x_src = x;
    if (out.data() < x.data() + nx && x.data() < out.data() + nx) {
        x_copy = Array<c64>::from(x, scratch.allocator());
        x_src = x_copy.as_ref();
    }

    Array<c64> bf(nfft, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < nb; ++i) bf[i] = b[i];
    keel::fft::fft_1d_inplace(bf.as_mut_ref());

    Array<c64> xf(nfft, c64(0.0, 0.0), scratch.allocator());
    for (usize i = 0; i < nx; ++i) out[i] = c64(0.0, 0.0);

    for (usize pos = 0; pos < nx; pos += block_l) {
        const usize l = keel::min(block_l, nx - pos);
        for (usize i = 0; i < l; ++i) xf[i] = x_src[pos + i];
        for (usize i = l; i < nfft; ++i) xf[i] = c64(0.0, 0.0);
        keel::fft::fft_1d_inplace(xf.as_mut_ref());
        for (usize i = 0; i < nfft; ++i) xf[i] *= bf[i];
        keel::fft::ifft_1d_inplace(xf.as_mut_ref());
        const usize blk_out = l + nb - 1;
        for (usize i = 0; i < blk_out && pos + i < nx; ++i) {
            out[pos + i] += xf[i];
        }
    }
    return SignalStatus::kOk;
}

SignalStatus decimate(ArrayRef<f64> x, usize factor, Array<f64>& out, Allocator* alloc) {
    if (factor == 0) {
        return SignalStatus::kInvalidArgument;
    }
    const usize nx = x.size();
    if (factor == 1 || nx == 0) {
        out = Array<f64>(x, alloc);
        return SignalStatus::kOk;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 wn_val = 0.8 / as<f64>(factor);
    TfResult tf;
    SignalStatus st = cheby1(8, 0.05, ArrayRef<f64>(&wn_val, 1), FilterType::kLowpass, false, tf,
                             scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    Array<f64> filtered(nx, 0.0, scratch.allocator());
    st = filtfilt(tf.b.as_ref(), tf.a.as_ref(), x, filtered.as_mut_ref(), scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    const usize out_len = (nx + factor - 1) / factor;
    const usize nbeg = (nx - 1) % factor;
    out = Array<f64>(out_len, 0.0, alloc);
    for (usize i = 0; i < out_len; ++i) {
        out[i] = filtered[nbeg + i * factor];
    }
    return SignalStatus::kOk;
}

constexpr usize gcd_usize(usize a, usize b) noexcept {
    while (b != 0) {
        const usize t = a % b;
        a = b;
        b = t;
    }
    return a;
}

SignalStatus resample(ArrayRef<f64> x, usize p, usize q, i32 n_order, f64 beta, Array<f64>& y_out,
                      Array<f64>* b_out, Allocator* alloc) {
    if (p == 0 || q == 0 || !keel::is_finite(beta) || beta < 0.0) {
        return SignalStatus::kInvalidArgument;
    }
    const usize g = gcd_usize(p, q);
    p /= g;
    q /= g;

    const usize nx = x.size();
    const i32 n_eff = (n_order > 0) ? n_order : 10;
    const usize pqmax = keel::max(p, q);
    const f64 fc = 1.0 / (2.0 * as<f64>(pqmax));
    const f64 w_cut = 2.0 * fc;
    const usize l_len_u = 2 * as<usize>(n_eff) * pqmax + 1;
    if (l_len_u > as<usize>(Limits<i32>::kMax)) {
        return SignalStatus::kInvalidArgument;
    }
    const i32 l_len = as<i32>(l_len_u);
    const i32 m_half = (l_len - 1) / 2;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> kw(as<usize>(l_len), 0.0, scratch.allocator());
    kaiser(kw.as_mut_ref(), beta);

    Array<f64> fk(as<usize>(l_len), 0.0, scratch.allocator());
    f64 sum_fk = 0.0;
    for (i32 i = 0; i < l_len; ++i) {
        const i32 k = i - m_half;
        const f64 fbv = (k == 0)
                            ? w_cut
                            : (keel::sin(kPi * w_cut * as<f64>(k)) / (kPi * as<f64>(k)));
        fk[as<usize>(i)] = fbv * kw[as<usize>(i)];
        sum_fk += fk[as<usize>(i)];
    }

    const f64 scale = (keel::abs(sum_fk) > 0.0) ? (as<f64>(p) / sum_fk) : 0.0;
    if (b_out != nullptr) {
        *b_out = Array<f64>(as<usize>(l_len), 0.0, alloc);
        for (i32 i = 0; i < l_len; ++i) {
            (*b_out)[as<usize>(i)] = scale * fk[as<usize>(i)];
        }
    }

    const i64 l_half = (l_len - 1) / 2;
    const i64 nz = as<i64>(q) - (l_half % as<i64>(q));
    const i64 l_half2 = l_half + nz;

    const usize hh_len = as<usize>(nz) + as<usize>(l_len);
    Array<f64> hh(hh_len, 0.0, scratch.allocator());
    for (i32 i = 0; i < l_len; ++i) {
        hh[as<usize>(nz) + as<usize>(i)] = scale * fk[as<usize>(i)];
    }

    const usize yfn = upfirdn_length(nx, hh_len, p, q);
    Array<f64> yfull(yfn, 0.0, scratch.allocator());
    const SignalStatus st =
        upfirdn(x, hh.as_ref(), p, q, yfull.as_mut_ref(), scratch.allocator());
    if (st != SignalStatus::kOk) return st;

    const i64 delay = l_half2 / as<i64>(q);
    const i64 ly = as<i64>((nx * p + q - 1) / q);
    const usize out_len = (ly > 0) ? as<usize>(ly) : 0;
    y_out = Array<f64>(out_len, 0.0, alloc);
    for (usize i = 0; i < out_len; ++i) {
        const i64 idx = delay + as<i64>(i);
        y_out[i] = (idx >= 0 && as<usize>(idx) < yfn) ? yfull[as<usize>(idx)] : 0.0;
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
