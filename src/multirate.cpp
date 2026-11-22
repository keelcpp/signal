#include "keel/signal.h"

#include "keel/array.h"
#include "keel/fft.h"

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

    // Polyphase reversed-subfilter decomposition: pack each phase's taps in
    // reverse order so every output sample reduces to a contiguous unit-stride
    // dot product without allocating or computing dropped samples.
    Array<usize> sub_off(p, alloc);
    Array<usize> sub_len(p, alloc);
    usize total_packed = 0;
    for (usize ph = 0; ph < p; ++ph) {
        sub_off[ph] = total_packed;
        const usize len = (ph < lh) ? ((lh - 1 - ph) / p + 1) : 0;
        sub_len[ph] = len;
        total_packed += len;
    }
    Array<f64> h_poly(total_packed, alloc);
    for (usize ph = 0; ph < p; ++ph) {
        const usize len = sub_len[ph];
        f64* hp = h_poly.data() + sub_off[ph];
        for (usize u = 0; u < len; ++u) {
            hp[u] = h[ph + (len - 1 - u) * p];
        }
    }

    const f64* x_base = x.data();
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

    const usize n_half = nfft / 2 + 1;
    Array<f64> b_pad(nfft, 0.0, alloc);
    for (usize i = 0; i < nb; ++i) b_pad[i] = b[i];
    Array<c64> bf(n_half, c64(0.0, 0.0), alloc);
    keel::fft::rfft_1d(b_pad.as_ref(), bf.as_mut_ref());

    Array<f64> x_pad(nfft, 0.0, alloc);
    Array<c64> xf(n_half, c64(0.0, 0.0), alloc);
    Array<f64> y_blk(nfft, 0.0, alloc);

    for (usize i = 0; i < nx; ++i) out[i] = 0.0;

    for (usize pos = 0; pos < nx; pos += block_l) {
        const usize l = keel::min(block_l, nx - pos);
        for (usize i = 0; i < l; ++i) x_pad[i] = x[pos + i];
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

    Array<c64> bf(nfft, c64(0.0, 0.0), alloc);
    for (usize i = 0; i < nb; ++i) bf[i] = b[i];
    keel::fft::fft_1d_inplace(bf.as_mut_ref());

    Array<c64> xf(nfft, c64(0.0, 0.0), alloc);
    for (usize i = 0; i < nx; ++i) out[i] = c64(0.0, 0.0);

    for (usize pos = 0; pos < nx; pos += block_l) {
        const usize l = keel::min(block_l, nx - pos);
        for (usize i = 0; i < l; ++i) xf[i] = x[pos + i];
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

} // namespace keel::signal
