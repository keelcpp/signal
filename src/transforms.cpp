#include "keel/signal.h"

#include "keel/arena.h"
#include "keel/array.h"
#include "keel/fft.h"
#include "keel/math.h"

namespace keel::signal {

c64 goertzel_bin(ArrayRef<f64> x, f64 k_zero_based) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return c64(0.0, 0.0);
    }
    const f64 w = kTau * k_zero_based / as<f64>(n);
    const f64 cw = keel::cos(w);
    const f64 sw = keel::sin(w);
    const f64 coeff = 2.0 * cw;

    f64 s_prev = 0.0;
    f64 s_prev2 = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 s = x[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    const f64 y_re = s_prev - cw * s_prev2;
    const f64 y_im = sw * s_prev2;
    const f64 phase = w * as<f64>(n - 1);
    const f64 pcw = keel::cos(phase);
    const f64 psw = keel::sin(phase);
    return c64(y_re * pcw + y_im * psw, y_im * pcw - y_re * psw);
}

c64 goertzel_bin_c64(ArrayRef<c64> x, f64 k_zero_based) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return c64(0.0, 0.0);
    }
    const f64 w = kTau * k_zero_based / as<f64>(n);
    const f64 cw = keel::cos(w);
    const f64 sw = keel::sin(w);
    const f64 coeff = 2.0 * cw;

    c64 s_prev(0.0, 0.0);
    c64 s_prev2(0.0, 0.0);
    for (usize i = 0; i < n; ++i) {
        const c64 s = x[i] + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    const c64 y = s_prev - c64(cw, -sw) * s_prev2;
    const f64 phase = w * as<f64>(n - 1);
    const f64 pcw = keel::cos(phase);
    const f64 psw = keel::sin(phase);
    return y * c64(pcw, -psw);
}

SignalStatus unwrap_phase(ArrayRef<f64> phase, MutArrayRef<f64> out, f64 period) noexcept {
    const usize n = phase.size();
    if (out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (n == 0) {
        return SignalStatus::kOk;
    }
    if (period <= 0.0) {
        for (usize i = 0; i < n; ++i) {
            out[i] = phase[i];
        }
        return SignalStatus::kOk;
    }
    f64 prev_in = phase[0];
    f64 prev_out = prev_in;
    out[0] = prev_out;
    for (usize i = 1; i < n; ++i) {
        const f64 cur_in = phase[i];
        f64 d = cur_in - prev_in;
        d -= period * keel::round(d / period);
        prev_out += d;
        out[i] = prev_out;
        prev_in = cur_in;
    }
    return SignalStatus::kOk;
}

SignalStatus dftmtx(usize n, MutArrayRef<c64> out_col_major) noexcept {
    if (n == 0 || out_col_major.size() < n * n) {
        return SignalStatus::kInvalidArgument;
    }
    const f64 base = -kTau / as<f64>(n);
    for (usize k = 0; k < n; ++k) {
        for (usize m = 0; m < n; ++m) {
            const f64 a = base * as<f64>(k) * as<f64>(m);
            out_col_major[m * n + k] = c64(keel::cos(a), keel::sin(a));
        }
    }
    return SignalStatus::kOk;
}

SignalStatus dst(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept {
    const usize n = x.size();
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    if (n == 0) return SignalStatus::kOk;
    const f64 scale = kPi / as<f64>(n + 1);
    for (usize k = 0; k < n; ++k) {
        f64 s = 0.0;
        for (usize m = 0; m < n; ++m) {
            s += x[m] * keel::sin(scale * as<f64>(m + 1) * as<f64>(k + 1));
        }
        out[k] = s;
    }
    return SignalStatus::kOk;
}

SignalStatus idst(ArrayRef<f64> y, MutArrayRef<f64> out) noexcept {
    const usize n = y.size();
    const SignalStatus st = dst(y, out);
    if (st != SignalStatus::kOk || n == 0) return st;
    const f64 scale = 2.0 / as<f64>(n + 1);
    for (usize i = 0; i < n; ++i) out[i] *= scale;
    return SignalStatus::kOk;
}

SignalStatus rceps(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = x.size();
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    if (n == 0) return SignalStatus::kOk;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> buf(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(x, buf.as_mut_ref());
    for (usize i = 0; i < n; ++i) {
        const f64 mag = keel::abs(buf[i]);
        buf[i] = c64(keel::log(keel::max(mag, 1e-300)), 0.0);
    }
    keel::fft::ifft_1d_inplace(buf.as_mut_ref());
    for (usize i = 0; i < n; ++i) out[i] = buf[i].real();
    return SignalStatus::kOk;
}

SignalStatus rceps_minphase(ArrayRef<f64> x, MutArrayRef<f64> cep_out,
                            MutArrayRef<f64> minphase_out, Allocator* alloc) {
    const usize n = x.size();
    if (cep_out.size() < n || minphase_out.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    const SignalStatus st = rceps(x, cep_out, alloc);
    if (st != SignalStatus::kOk || n == 0) return st;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> wbuf(n, c64(0.0, 0.0), scratch.allocator());
    const usize khalf = (n - 1) / 2;
    for (usize i = 0; i < n; ++i) {
        f64 w = 0.0;
        if (i == 0) {
            w = 1.0;
        } else if (i <= khalf) {
            w = 2.0;
        } else if (n % 2 == 0 && i == n / 2) {
            w = 1.0;
        }
        wbuf[i] = c64(w * cep_out[i], 0.0);
    }
    keel::fft::fft_1d_inplace(wbuf.as_mut_ref());
    for (usize i = 0; i < n; ++i) wbuf[i] = keel::exp(wbuf[i]);
    keel::fft::ifft_1d_inplace(wbuf.as_mut_ref());
    for (usize i = 0; i < n; ++i) minphase_out[i] = wbuf[i].real();
    return SignalStatus::kOk;
}

SignalStatus cceps(ArrayRef<f64> x, MutArrayRef<f64> out, f64* nd_out, Allocator* alloc) {
    const usize n = x.size();
    if (nd_out) *nd_out = 0.0;
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    if (n == 0) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> buf(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(x, buf.as_mut_ref());

    Array<f64> ph(n, 0.0, scratch.allocator());
    Array<f64> lmag(n, 0.0, scratch.allocator());
    f64 prev_phase = 0.0;
    for (usize i = 0; i < n; ++i) {
        const f64 mag = keel::abs(buf[i]);
        f64 phase = keel::arg(buf[i]);
        while (phase - prev_phase > kPi) phase -= kTau;
        while (phase - prev_phase < -kPi) phase += kTau;
        prev_phase = phase;
        ph[i] = phase;
        lmag[i] = keel::log(keel::max(mag, 1e-300));
    }
    const usize nh = (n + 1) / 2;
    const usize idx = (n == 1) ? 0 : nh;
    const f64 nd = keel::round(ph[idx] / kPi);
    for (usize i = 0; i < n; ++i) {
        ph[i] -= kPi * nd * as<f64>(i) / as<f64>(nh);
    }
    if (nd_out) *nd_out = nd;
    for (usize i = 0; i < n; ++i) buf[i] = c64(lmag[i], ph[i]);
    keel::fft::ifft_1d_inplace(buf.as_mut_ref());
    for (usize i = 0; i < n; ++i) out[i] = buf[i].real();
    return SignalStatus::kOk;
}

SignalStatus icceps(ArrayRef<f64> c, MutArrayRef<f64> out, Allocator* alloc) {
    const usize n = c.size();
    if (out.size() < n) return SignalStatus::kInvalidArgument;
    if (n == 0) return SignalStatus::kOk;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> buf(n, c64(0.0, 0.0), scratch.allocator());
    keel::fft::fft_1d_real(c, buf.as_mut_ref());
    for (usize i = 0; i < n; ++i) buf[i] = keel::exp(buf[i]);
    keel::fft::ifft_1d_inplace(buf.as_mut_ref());
    for (usize i = 0; i < n; ++i) out[i] = buf[i].real();
    return SignalStatus::kOk;
}

namespace {

usize czt_next_pow2(usize n) noexcept {
    usize p = 1;
    while (p < n) p <<= 1;
    return p;
}

} // namespace

SignalStatus czt_batch(ArrayRef<c64> x_col_major, usize n_rows, usize n_cols, usize m, c64 w, c64 a,
                       MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (m == 0) return SignalStatus::kInvalidArgument;
    if (n_rows == 0 || n_cols == 0) return SignalStatus::kOk;
    if (x_col_major.size() < n_rows * n_cols || out_col_major.size() < m * n_cols) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize conv_len = czt_next_pow2(n_rows + m - 1);

    Array<c64> pre(n_rows, c64(0.0, 0.0), scratch.allocator());
    for (usize n = 0; n < n_rows; ++n) {
        const f64 fn = as<f64>(n);
        const f64 n2_half = 0.5 * fn * fn;
        pre[n] = keel::pow(a, -fn) * keel::pow(w, n2_half);
    }

    Array<c64> post(m, c64(0.0, 0.0), scratch.allocator());
    Array<c64> h(conv_len, c64(0.0, 0.0), scratch.allocator());
    for (usize k = 0; k < m; ++k) {
        const f64 fk = as<f64>(k);
        const f64 k2_half = 0.5 * fk * fk;
        post[k] = keel::pow(w, k2_half);
        h[k] = keel::pow(w, -k2_half);
    }
    for (usize n = 1; n < n_rows; ++n) {
        const f64 fn = as<f64>(n);
        const f64 n2_half = 0.5 * fn * fn;
        h[conv_len - n] = keel::pow(w, -n2_half);
    }

    auto plan_res = keel::fft::FftPlan1D::create(conv_len, scratch.allocator());
    if (!plan_res.is_ok()) return SignalStatus::kInvalidArgument;
    const keel::fft::FftPlan1D& plan = plan_res.value();
    plan.exec_inplace(h.as_mut_ref(), keel::fft::FftDirection::Forward);

    Array<c64> g(conv_len, c64(0.0, 0.0), scratch.allocator());
    for (usize col = 0; col < n_cols; ++col) {
        const c64* xcol = x_col_major.data() + col * n_rows;
        for (usize n = 0; n < n_rows; ++n) g[n] = xcol[n] * pre[n];
        for (usize n = n_rows; n < conv_len; ++n) g[n] = c64(0.0, 0.0);
        plan.exec_inplace(g.as_mut_ref(), keel::fft::FftDirection::Forward);
        for (usize i = 0; i < conv_len; ++i) g[i] *= h[i];
        plan.exec_inplace(g.as_mut_ref(), keel::fft::FftDirection::Inverse);
        c64* ycol = out_col_major.data() + col * m;
        for (usize k = 0; k < m; ++k) ycol[k] = g[k] * post[k];
    }
    return SignalStatus::kOk;
}

SignalStatus czt_batch_real(ArrayRef<f64> x_col_major, usize n_rows, usize n_cols, usize m, c64 w,
                            c64 a, MutArrayRef<c64> out_col_major, Allocator* alloc) {
    if (m == 0) return SignalStatus::kInvalidArgument;
    if (n_rows == 0 || n_cols == 0) return SignalStatus::kOk;
    if (x_col_major.size() < n_rows * n_cols || out_col_major.size() < m * n_cols) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize conv_len = czt_next_pow2(n_rows + m - 1);

    Array<c64> pre(n_rows, c64(0.0, 0.0), scratch.allocator());
    for (usize n = 0; n < n_rows; ++n) {
        const f64 fn = as<f64>(n);
        const f64 n2_half = 0.5 * fn * fn;
        pre[n] = keel::pow(a, -fn) * keel::pow(w, n2_half);
    }

    Array<c64> post(m, c64(0.0, 0.0), scratch.allocator());
    Array<c64> h(conv_len, c64(0.0, 0.0), scratch.allocator());
    for (usize k = 0; k < m; ++k) {
        const f64 fk = as<f64>(k);
        const f64 k2_half = 0.5 * fk * fk;
        post[k] = keel::pow(w, k2_half);
        h[k] = keel::pow(w, -k2_half);
    }
    for (usize n = 1; n < n_rows; ++n) {
        const f64 fn = as<f64>(n);
        const f64 n2_half = 0.5 * fn * fn;
        h[conv_len - n] = keel::pow(w, -n2_half);
    }

    auto plan_res = keel::fft::FftPlan1D::create(conv_len, scratch.allocator());
    if (!plan_res.is_ok()) return SignalStatus::kInvalidArgument;
    const keel::fft::FftPlan1D& plan = plan_res.value();
    plan.exec_inplace(h.as_mut_ref(), keel::fft::FftDirection::Forward);

    Array<c64> g(conv_len, c64(0.0, 0.0), scratch.allocator());
    for (usize col = 0; col < n_cols; ++col) {
        const f64* xcol = x_col_major.data() + col * n_rows;
        for (usize n = 0; n < n_rows; ++n) g[n] = xcol[n] * pre[n];
        for (usize n = n_rows; n < conv_len; ++n) g[n] = c64(0.0, 0.0);
        plan.exec_inplace(g.as_mut_ref(), keel::fft::FftDirection::Forward);
        for (usize i = 0; i < conv_len; ++i) g[i] *= h[i];
        plan.exec_inplace(g.as_mut_ref(), keel::fft::FftDirection::Inverse);
        c64* ycol = out_col_major.data() + col * m;
        for (usize k = 0; k < m; ++k) ycol[k] = g[k] * post[k];
    }
    return SignalStatus::kOk;
}

SignalStatus czt(ArrayRef<c64> x, usize m, c64 w, c64 a, MutArrayRef<c64> out, Allocator* alloc) {
    if (x.empty()) return SignalStatus::kOk;
    return czt_batch(x, x.size(), 1, m, w, a, out, alloc);
}

} // namespace keel::signal
