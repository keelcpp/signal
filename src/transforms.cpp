#include "keel/signal.h"

#include "keel/math.h"

namespace keel::signal {

c64 goertzel_bin(ArrayRef<f64> x, f64 k_zero_based) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return c64(0.0, 0.0);
    }
    const f64 w = kTau * k_zero_based / static_cast<f64>(n);
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
    const f64 phase = w * static_cast<f64>(n - 1);
    const f64 pcw = keel::cos(phase);
    const f64 psw = keel::sin(phase);
    return c64(y_re * pcw + y_im * psw, y_im * pcw - y_re * psw);
}

c64 goertzel_bin_c64(ArrayRef<c64> x, f64 k_zero_based) noexcept {
    const usize n = x.size();
    if (n == 0) {
        return c64(0.0, 0.0);
    }
    const f64 w = kTau * k_zero_based / static_cast<f64>(n);
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
    const f64 phase = w * static_cast<f64>(n - 1);
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

} // namespace keel::signal
