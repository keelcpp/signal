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
    const usize up_len = (lx - 1) * p + 1;
    const usize conv_len = up_len + lh - 1;
    const usize out_len = (conv_len + q - 1) / q;
    if (out.size() < out_len) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> conv(conv_len, 0.0, alloc);
    for (usize i = 0; i < lx; ++i) {
        const f64 xi = x[i];
        if (xi == 0.0) continue;
        const usize base = i * p;
        for (usize j = 0; j < lh; ++j) {
            conv[base + j] += xi * h[j];
        }
    }
    for (usize k = 0; k < out_len; ++k) {
        out[k] = conv[k * q];
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
