#include "keel/signal.h"

#include "keel/array.h"
#include "keel/math.h"

#include <cstdio>

namespace {

using namespace keel;
namespace sig = keel::signal;

i32 g_failures = 0;

void check_true(bool cond, const char* expr, i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failures;
    }
}

void check_near(f64 actual, f64 expected, f64 tol, const char* expr, i32 line) {
    if (!(keel::abs(actual - expected) <= tol)) {
        std::fprintf(stderr, "FAIL line %d: %s (actual=%.15g, expected=%.15g, tol=%.3g)\n",
                     line, expr, actual, expected, tol);
        ++g_failures;
    }
}

#define CHECK(cond) check_true((cond), #cond, __LINE__)
#define CHECK_NEAR(a, e, t) check_near((a), (e), (t), #a " ~= " #e, __LINE__)

void test_windows() {
    f64 w[16]{};

    // Single-sample edge cases
    sig::rectwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::hann(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::hamming(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::blackman(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::bartlett(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::triang(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::kaiser(1, 5.0, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::tukeywin(1, 0.5, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::flattopwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::parzenwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::nuttallwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::blackmanharris(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::bohmanwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::barthannwin(1, w); CHECK_NEAR(w[0], 1.0, 1e-15);
    CHECK(sig::gausswin(1, 2.5, w) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[0], 1.0, 1e-15);
    CHECK(sig::chebwin(1, 100.0, w) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[0], 1.0, 1e-15);

    // Hann N=5: 0, 0.5, 1, 0.5, 0
    sig::hann(5, w);
    CHECK_NEAR(w[0], 0.0, 1e-14);
    CHECK_NEAR(w[1], 0.5, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[3], 0.5, 1e-14);
    CHECK_NEAR(w[4], 0.0, 1e-14);

    // Hamming N=5: 0.08, 0.54, 1.0, 0.54, 0.08
    sig::hamming(5, w);
    CHECK_NEAR(w[0], 0.08, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[4], 0.08, 1e-14);

    // Bartlett N=5: 0, 0.5, 1, 0.5, 0
    sig::bartlett(5, w);
    CHECK_NEAR(w[0], 0.0, 1e-14);
    CHECK_NEAR(w[1], 0.5, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[4], 0.0, 1e-14);

    // Triang N=4 & N=5
    sig::triang(4, w);
    CHECK_NEAR(w[0], 0.25, 1e-14);
    CHECK_NEAR(w[1], 0.75, 1e-14);
    CHECK_NEAR(w[2], 0.75, 1e-14);
    CHECK_NEAR(w[3], 0.25, 1e-14);
    sig::triang(5, w);
    CHECK_NEAR(w[0], 1.0 / 3.0, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);

    // Kaiser symmetry and peak
    sig::kaiser(9, 4.0, w);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    for (usize i = 0; i < 4; ++i) CHECK_NEAR(w[i], w[8 - i], 1e-14);

    // Tukey r=0 -> rect, r=1 -> hann
    f64 h[9]{};
    sig::hann(9, h);
    sig::tukeywin(9, 1.0, w);
    for (usize i = 0; i < 9; ++i) CHECK_NEAR(w[i], h[i], 1e-14);
    sig::tukeywin(9, 0.0, w);
    for (usize i = 0; i < 9; ++i) CHECK_NEAR(w[i], 1.0, 1e-14);

    // Blackman, flattop, parzen, nuttall, blackmanharris, bohman, barthann symmetry
    auto check_sym = [&](usize n) {
        for (usize i = 0; i < n / 2; ++i) {
            CHECK_NEAR(w[i], w[n - 1 - i], 1e-13);
        }
    };
    sig::blackman(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::flattopwin(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-6);
    sig::parzenwin(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::nuttallwin(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::blackmanharris(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::bohmanwin(11, w); check_sym(11); CHECK_NEAR(w[0], 0.0, 1e-14); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::barthannwin(11, w); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);

    // Gausswin validation & values
    CHECK(sig::gausswin(8, 0.0, w) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::gausswin(8, -1.0, w) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::gausswin(9, 2.5, w) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    check_sym(9);

    // Chebwin validation & symmetry (odd & even)
    CHECK(sig::chebwin(8, 0.0, w) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::chebwin(9, 80.0, w) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    check_sym(9);
    CHECK(sig::chebwin(10, 60.0, w) == sig::SignalStatus::kOk);
    check_sym(10);

    // Taylorwin validation & symmetry
    CHECK(sig::taylorwin(10, 1, -30.0, w) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::taylorwin(10, 4, 0.0, w) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::taylorwin(11, 4, -30.0, w) == sig::SignalStatus::kOk);
    check_sym(11);
    CHECK(w[5] > w[0]);

    // DPSS orthonormality and concentration ratios
    constexpr usize N = 32;
    constexpr usize K = 4;
    f64 tapers[N * K]{};
    f64 ratios[K]{};
    CHECK(sig::dpss(N, 3.0, K, tapers, ratios) == sig::SignalStatus::kOk);
    for (usize j = 0; j < K; ++j) {
        CHECK(ratios[j] > 0.0 && ratios[j] <= 1.0);
        if (j > 0) CHECK(ratios[j] <= ratios[j - 1] + 1e-12);
        for (usize j2 = j; j2 < K; ++j2) {
            f64 dot = 0.0;
            for (usize i = 0; i < N; ++i) {
                dot += tapers[j * N + i] * tapers[j2 * N + i];
            }
            CHECK_NEAR(dot, (j == j2) ? 1.0 : 0.0, 1e-11);
        }
    }
}

void test_conv() {
    const f64 a[3] = {1.0, 2.0, 3.0};
    const f64 b[2] = {4.0, 5.0};
    f64 c[4]{};
    sig::conv_1d(a, 3, b, 2, c);
    CHECK_NEAR(c[0], 4.0, 1e-15);
    CHECK_NEAR(c[1], 13.0, 1e-15);
    CHECK_NEAR(c[2], 22.0, 1e-15);
    CHECK_NEAR(c[3], 15.0, 1e-15);

    // 2D conv (column-major): 2x2 * 2x2 -> 3x3
    // A = [[1, 3], [2, 4]], B = [[1, 0], [0, 2]]
    const f64 A[4] = {1.0, 2.0, 3.0, 4.0};
    const f64 B[4] = {1.0, 0.0, 0.0, 2.0};
    f64 C[9]{};
    sig::conv_2d(A, 2, 2, B, 2, 2, C);
    // Expected column-major:
    // col 0: [1, 2, 0]
    // col 1: [3, 4+2=6, 4]
    // col 2: [0, 6, 8]
    CHECK_NEAR(C[0], 1.0, 1e-15);
    CHECK_NEAR(C[1], 2.0, 1e-15);
    CHECK_NEAR(C[2], 0.0, 1e-15);
    CHECK_NEAR(C[3], 3.0, 1e-15);
    CHECK_NEAR(C[4], 6.0, 1e-15);
    CHECK_NEAR(C[5], 4.0, 1e-15);
    CHECK_NEAR(C[6], 0.0, 1e-15);
    CHECK_NEAR(C[7], 6.0, 1e-15);
    CHECK_NEAR(C[8], 8.0, 1e-15);
}

void test_iir_and_sos() {
    // DF2T chunk continuity test
    const f64 b[3] = {0.2, 0.4, 0.2};
    const f64 a[3] = {1.0, -0.5, 0.25};
    const f64 x[8] = {1.0, -2.0, 3.0, 0.5, -1.0, 4.0, 2.0, -0.5};
    f64 y_full[8]{};
    sig::filter_df2t(b, 3, a, 3, x, 8, y_full);

    f64 y_part[8]{};
    f64 zf[2]{};
    sig::filter_df2t(b, 3, a, 3, x, 3, y_part, nullptr, 0, zf);
    sig::filter_df2t(b, 3, a, 3, x + 3, 5, y_part + 3, zf, 2, nullptr);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_part[i], y_full[i], 1e-14);
    }

    // Complex DF2T matches real when imag == 0
    c64 bc[3] = {c64(0.2, 0.0), c64(0.4, 0.0), c64(0.2, 0.0)};
    c64 ac[3] = {c64(1.0, 0.0), c64(-0.5, 0.0), c64(0.25, 0.0)};
    c64 xc[8]{};
    c64 yc[8]{};
    for (usize i = 0; i < 8; ++i) xc[i] = c64(x[i], 0.0);
    sig::filter_df2t_c64(bc, 3, ac, 3, xc, 8, yc);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(yc[i].real(), y_full[i], 1e-14);
        CHECK_NEAR(yc[i].imag(), 0.0, 1e-14);
    }

    // Biquad steady-state initial conditions: constant 1 -> constant DC gain
    const f64 dc = (b[0] + b[1] + b[2]) / (1.0 + a[1] + a[2]);
    const sig::BiquadState zi = sig::biquad_zi(b[0], b[1], b[2], a[1], a[2]);
    f64 ones[10]{};
    f64 y_ss[10]{};
    for (usize i = 0; i < 10; ++i) ones[i] = 1.0;
    sig::biquad_df2t_state(b[0], b[1], b[2], a[1], a[2], ones, y_ss, 10, zi.s1, zi.s2);
    for (usize i = 0; i < 10; ++i) {
        CHECK_NEAR(y_ss[i], dc, 1e-14);
    }

    // SOS 2-section cascade (column-major 2x6)
    // Section 0: [b0, b1, b2, 1, a1, a2], Section 1: [0.5, 0.5, 0, 1, 0, 0]
    const f64 sos[12] = {
        b[0], 0.5,   // b0 col
        b[1], 0.5,   // b1 col
        b[2], 0.0,   // b2 col
        1.0,  1.0,   // a0 col
        a[1], 0.0,   // a1 col
        a[2], 0.0,   // a2 col
    };
    f64 y_sos[8]{};
    CHECK(sig::sosfilt_col_major(sos, 2, x, y_sos, 8) == sig::SignalStatus::kOk);

    f64 tmp[8]{};
    f64 ref[8]{};
    sig::biquad_df2t(b[0], b[1], b[2], a[1], a[2], x, tmp, 8);
    sig::biquad_df2t(0.5, 0.5, 0.0, 0.0, 0.0, tmp, ref, 8);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_sos[i], ref[i], 1e-14);
    }

    // Zero leading coeff check
    f64 bad_sos[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(sig::sosfilt_col_major(bad_sos, 1, x, y_sos, 8) == sig::SignalStatus::kZeroLeadingCoeff);
    CHECK(sig::sosfiltfilt_col_major(bad_sos, 1, x, y_sos, 8) == sig::SignalStatus::kZeroLeadingCoeff);

    // sosfiltfilt on constant signal preserves constant * dc^2
    constexpr usize M = 40;
    f64 c_in[M]{};
    f64 c_out[M]{};
    for (usize i = 0; i < M; ++i) c_in[i] = 3.5;
    CHECK(sig::sosfiltfilt_col_major(sos, 2, c_in, c_out, M) == sig::SignalStatus::kOk);
    const f64 expected_dc = 3.5 * dc * dc;
    for (usize i = 0; i < M; ++i) {
        CHECK_NEAR(c_out[i], expected_dc, 1e-12);
    }
}

void test_goertzel_and_unwrap() {
    constexpr usize N = 16;
    f64 xr[N]{};
    c64 xc[N]{};
    for (usize n = 0; n < N; ++n) {
        xr[n] = keel::cos(kTau * 3.0 * static_cast<f64>(n) / static_cast<f64>(N)) +
                0.5 * keel::sin(kTau * 5.0 * static_cast<f64>(n) / static_cast<f64>(N));
        xc[n] = c64(xr[n], 0.25 * static_cast<f64>(n));
    }

    const f64 bins[3] = {0.0, 3.0, 4.25};
    for (usize bi = 0; bi < 3; ++bi) {
        const f64 k = bins[bi];
        c64 ref_r(0.0, 0.0);
        c64 ref_c(0.0, 0.0);
        for (usize n = 0; n < N; ++n) {
            const f64 ang = -kTau * k * static_cast<f64>(n) / static_cast<f64>(N);
            const c64 tw(keel::cos(ang), keel::sin(ang));
            ref_r = ref_r + c64(xr[n], 0.0) * tw;
            ref_c = ref_c + xc[n] * tw;
        }
        const c64 gr = sig::goertzel_bin(xr, N, k);
        const c64 gc = sig::goertzel_bin_c64(xc, N, k);
        CHECK_NEAR(gr.real(), ref_r.real(), 1e-11);
        CHECK_NEAR(gr.imag(), ref_r.imag(), 1e-11);
        CHECK_NEAR(gc.real(), ref_c.real(), 1e-11);
        CHECK_NEAR(gc.imag(), ref_c.imag(), 1e-11);
    }

    // Unwrap synthetic wrapped linear ramp (also test in-place)
    constexpr usize P = 20;
    f64 true_p[P]{};
    f64 wrapped[P]{};
    f64 out_p[P]{};
    for (usize i = 0; i < P; ++i) {
        true_p[i] = 0.7 * static_cast<f64>(i) - 5.0;
        wrapped[i] = true_p[i] - kTau * keel::round(true_p[i] / kTau);
    }
    sig::unwrap_phase(wrapped, P, out_p);
    const f64 offset = out_p[0] - true_p[0];
    for (usize i = 0; i < P; ++i) {
        CHECK_NEAR(out_p[i] - offset, true_p[i], 1e-12);
    }
    // In-place
    sig::unwrap_phase(wrapped, P, wrapped);
    for (usize i = 0; i < P; ++i) {
        CHECK_NEAR(wrapped[i], out_p[i], 1e-14);
    }
}

} // namespace

int main() {
    test_windows();
    test_conv();
    test_iir_and_sos();
    test_goertzel_and_unwrap();

    if (g_failures == 0) {
        std::printf("ALL keel::signal TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
