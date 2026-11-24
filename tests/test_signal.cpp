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
    auto w_slice = [&](usize n) { return MutArrayRef<f64>(w, n); };

    // Single-sample edge cases
    sig::rectwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::hann(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::hamming(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::blackman(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::bartlett(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::triang(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::kaiser(w_slice(1), 5.0); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::tukeywin(w_slice(1), 0.5); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::flattopwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::parzenwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::nuttallwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::blackmanharris(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::bohmanwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    sig::barthannwin(w_slice(1)); CHECK_NEAR(w[0], 1.0, 1e-15);
    CHECK(sig::gausswin(w_slice(1), 2.5) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[0], 1.0, 1e-15);
    CHECK(sig::chebwin(w_slice(1), 100.0) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[0], 1.0, 1e-15);

    // Hann N=5: 0, 0.5, 1, 0.5, 0
    sig::hann(w_slice(5));
    CHECK_NEAR(w[0], 0.0, 1e-14);
    CHECK_NEAR(w[1], 0.5, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[3], 0.5, 1e-14);
    CHECK_NEAR(w[4], 0.0, 1e-14);

    // Hamming N=5: 0.08, 0.54, 1.0, 0.54, 0.08
    sig::hamming(w_slice(5));
    CHECK_NEAR(w[0], 0.08, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[4], 0.08, 1e-14);

    // Bartlett N=5: 0, 0.5, 1, 0.5, 0
    sig::bartlett(w_slice(5));
    CHECK_NEAR(w[0], 0.0, 1e-14);
    CHECK_NEAR(w[1], 0.5, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);
    CHECK_NEAR(w[4], 0.0, 1e-14);

    // Triang N=4 & N=5
    sig::triang(w_slice(4));
    CHECK_NEAR(w[0], 0.25, 1e-14);
    CHECK_NEAR(w[1], 0.75, 1e-14);
    CHECK_NEAR(w[2], 0.75, 1e-14);
    CHECK_NEAR(w[3], 0.25, 1e-14);
    sig::triang(w_slice(5));
    CHECK_NEAR(w[0], 1.0 / 3.0, 1e-14);
    CHECK_NEAR(w[2], 1.0, 1e-14);

    // Kaiser symmetry and peak
    sig::kaiser(w_slice(9), 4.0);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    for (usize i = 0; i < 4; ++i) CHECK_NEAR(w[i], w[8 - i], 1e-14);

    // Tukey r=0 -> rect, r=1 -> hann
    f64 h[9]{};
    sig::hann(h);
    sig::tukeywin(w_slice(9), 1.0);
    for (usize i = 0; i < 9; ++i) CHECK_NEAR(w[i], h[i], 1e-14);
    sig::tukeywin(w_slice(9), 0.0);
    for (usize i = 0; i < 9; ++i) CHECK_NEAR(w[i], 1.0, 1e-14);

    auto check_sym = [&](usize n) {
        for (usize i = 0; i < n / 2; ++i) {
            CHECK_NEAR(w[i], w[n - 1 - i], 1e-13);
        }
    };
    sig::blackman(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::flattopwin(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-6);
    sig::parzenwin(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::nuttallwin(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::blackmanharris(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::bohmanwin(w_slice(11)); check_sym(11); CHECK_NEAR(w[0], 0.0, 1e-14); CHECK_NEAR(w[5], 1.0, 1e-14);
    sig::barthannwin(w_slice(11)); check_sym(11); CHECK_NEAR(w[5], 1.0, 1e-14);

    // Gausswin validation & values
    CHECK(sig::gausswin(w_slice(8), 0.0) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::gausswin(w_slice(8), -1.0) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::gausswin(w_slice(9), 2.5) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    check_sym(9);

    // Chebwin validation & symmetry (odd & even)
    CHECK(sig::chebwin(w_slice(8), 0.0) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::chebwin(w_slice(9), 80.0) == sig::SignalStatus::kOk);
    CHECK_NEAR(w[4], 1.0, 1e-14);
    check_sym(9);
    CHECK(sig::chebwin(w_slice(10), 60.0) == sig::SignalStatus::kOk);
    check_sym(10);

    // Taylorwin validation & symmetry
    CHECK(sig::taylorwin(w_slice(10), 1, -30.0) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::taylorwin(w_slice(10), 4, 0.0) == sig::SignalStatus::kInvalidArgument);
    CHECK(sig::taylorwin(w_slice(11), 4, -30.0) == sig::SignalStatus::kOk);
    check_sym(11);
    CHECK(w[5] > w[0]);

    // DPSS orthonormality and concentration ratios
    constexpr usize N = 32;
    constexpr usize K = 4;
    f64 tapers[N * K]{};
    f64 ratios[K]{};
    CHECK(sig::dpss(N, 3.0, tapers, ratios) == sig::SignalStatus::kOk);
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
    CHECK(sig::conv_1d(a, b, c) == sig::SignalStatus::kOk);
    CHECK_NEAR(c[0], 4.0, 1e-15);
    CHECK_NEAR(c[1], 13.0, 1e-15);
    CHECK_NEAR(c[2], 22.0, 1e-15);
    CHECK_NEAR(c[3], 15.0, 1e-15);

    // Undersized output rejected
    CHECK(sig::conv_1d(a, b, MutArrayRef<f64>(c, 3)) == sig::SignalStatus::kInvalidArgument);

    // 2D conv (column-major): 2x2 * 2x2 -> 3x3
    const f64 A[4] = {1.0, 2.0, 3.0, 4.0};
    const f64 B[4] = {1.0, 0.0, 0.0, 2.0};
    f64 C[9]{};
    CHECK(sig::conv_2d(A, 2, 2, B, 2, 2, C) == sig::SignalStatus::kOk);
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
    CHECK(sig::filter_df2t(b, a, x, y_full) == sig::SignalStatus::kOk);

    f64 y_part[8]{};
    f64 zf[2]{};
    CHECK(sig::filter_df2t(b, a, ArrayRef<f64>(x, 3), MutArrayRef<f64>(y_part, 3), {}, zf) ==
          sig::SignalStatus::kOk);
    CHECK(sig::filter_df2t(b, a, ArrayRef<f64>(x + 3, 5), MutArrayRef<f64>(y_part + 3, 5), zf, {}) ==
          sig::SignalStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_part[i], y_full[i], 1e-14);
    }

    // Complex DF2T matches real when imag == 0
    c64 bc[3] = {c64(0.2, 0.0), c64(0.4, 0.0), c64(0.2, 0.0)};
    c64 ac[3] = {c64(1.0, 0.0), c64(-0.5, 0.0), c64(0.25, 0.0)};
    c64 xc[8]{};
    c64 yc[8]{};
    for (usize i = 0; i < 8; ++i) xc[i] = c64(x[i], 0.0);
    CHECK(sig::filter_df2t_c64(bc, ac, xc, yc) == sig::SignalStatus::kOk);
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
    CHECK(sig::biquad_df2t_state(b[0], b[1], b[2], a[1], a[2], ones, y_ss, zi.s1, zi.s2) ==
          sig::SignalStatus::kOk);
    for (usize i = 0; i < 10; ++i) {
        CHECK_NEAR(y_ss[i], dc, 1e-14);
    }

    // SOS 2-section cascade (column-major 2x6)
    const f64 sos[12] = {
        b[0], 0.5,   // b0 col
        b[1], 0.5,   // b1 col
        b[2], 0.0,   // b2 col
        1.0,  1.0,   // a0 col
        a[1], 0.0,   // a1 col
        a[2], 0.0,   // a2 col
    };
    f64 y_sos[8]{};
    CHECK(sig::sosfilt_col_major(sos, 2, x, y_sos) == sig::SignalStatus::kOk);

    f64 tmp[8]{};
    f64 ref[8]{};
    CHECK(sig::biquad_df2t(b[0], b[1], b[2], a[1], a[2], x, tmp) == sig::SignalStatus::kOk);
    CHECK(sig::biquad_df2t(0.5, 0.5, 0.0, 0.0, 0.0, tmp, ref) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 8; ++i) {
        CHECK_NEAR(y_sos[i], ref[i], 1e-14);
    }

    // Zero leading coeff check
    f64 bad_sos[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    CHECK(sig::sosfilt_col_major(bad_sos, 1, x, y_sos) == sig::SignalStatus::kZeroLeadingCoeff);
    CHECK(sig::sosfiltfilt_col_major(bad_sos, 1, x, y_sos) == sig::SignalStatus::kZeroLeadingCoeff);

    // sosfiltfilt on constant signal preserves constant * dc^2
    constexpr usize M = 40;
    f64 c_in[M]{};
    f64 c_out[M]{};
    for (usize i = 0; i < M; ++i) c_in[i] = 3.5;
    CHECK(sig::sosfiltfilt_col_major(sos, 2, c_in, c_out) == sig::SignalStatus::kOk);
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
        xr[n] = keel::cos(kTau * 3.0 * keel::as<f64>(n) / keel::as<f64>(N)) +
                0.5 * keel::sin(kTau * 5.0 * keel::as<f64>(n) / keel::as<f64>(N));
        xc[n] = c64(xr[n], 0.25 * keel::as<f64>(n));
    }

    const f64 bins[3] = {0.0, 3.0, 4.25};
    for (usize bi = 0; bi < 3; ++bi) {
        const f64 k = bins[bi];
        c64 ref_r(0.0, 0.0);
        c64 ref_c(0.0, 0.0);
        for (usize n = 0; n < N; ++n) {
            const f64 ang = -kTau * k * keel::as<f64>(n) / keel::as<f64>(N);
            const c64 tw(keel::cos(ang), keel::sin(ang));
            ref_r = ref_r + c64(xr[n], 0.0) * tw;
            ref_c = ref_c + xc[n] * tw;
        }
        const c64 gr = sig::goertzel_bin(xr, k);
        const c64 gc = sig::goertzel_bin_c64(xc, k);
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
        true_p[i] = 0.7 * keel::as<f64>(i) - 5.0;
        wrapped[i] = true_p[i] - kTau * keel::round(true_p[i] / kTau);
    }
    CHECK(sig::unwrap_phase(wrapped, out_p) == sig::SignalStatus::kOk);
    const f64 offset = out_p[0] - true_p[0];
    for (usize i = 0; i < P; ++i) {
        CHECK_NEAR(out_p[i] - offset, true_p[i], 1e-12);
    }
    // In-place
    CHECK(sig::unwrap_phase(wrapped, wrapped) == sig::SignalStatus::kOk);
    for (usize i = 0; i < P; ++i) {
        CHECK_NEAR(wrapped[i], out_p[i], 1e-14);
    }
}

void test_dct_fwht_hilbert() {
    // DCT types 1..4 roundtrip (both small direct N=10 and FFT-accelerated N=32 & non-pow2 N=25)
    for (usize len : {usize{10}, usize{25}, usize{32}}) {
        keel::Array<f64> x(len, 0.0), y(len, 0.0), z(len, 0.0);
        for (usize i = 0; i < len; ++i) {
            x[i] = keel::sin(0.37 * keel::as<f64>(i)) + 0.5 * keel::cos(1.1 * keel::as<f64>(i));
        }
        for (i32 type = 1; type <= 4; ++type) {
            CHECK(sig::dct(x.as_ref(), y.as_mut_ref(), type) == sig::SignalStatus::kOk);
            CHECK(sig::idct(y.as_ref(), z.as_mut_ref(), type) == sig::SignalStatus::kOk);
            for (usize i = 0; i < len; ++i) {
                CHECK_NEAR(z[i], x[i], 1e-11);
            }
        }
    }

    // FWHT roundtrip across all 3 orderings
    const f64 fw_in[8] = {1.0, 2.0, -1.0, 4.0, 0.5, -2.5, 3.0, 1.5};
    f64 fw_y[8]{}, fw_z[8]{};
    for (sig::FwhtOrdering ord : {sig::FwhtOrdering::kSequency, sig::FwhtOrdering::kHadamard,
                                  sig::FwhtOrdering::kDyadic}) {
        CHECK(sig::fwht(fw_in, fw_y, ord) == sig::SignalStatus::kOk);
        CHECK_NEAR(fw_y[0], 1.0625, 1e-12); // DC = mean(fw_in)
        CHECK(sig::ifwht(fw_y, fw_z, ord) == sig::SignalStatus::kOk);
        for (usize i = 0; i < 8; ++i) {
            CHECK_NEAR(fw_z[i], fw_in[i], 1e-12);
        }
    }

    // Hilbert analytic signal on non-power-of-2 cosine -> constant magnitude 1
    constexpr usize HN = 60;
    f64 hx[HN]{};
    c64 hy[HN]{};
    for (usize i = 0; i < HN; ++i) {
        hx[i] = keel::cos(kTau * 5.0 * keel::as<f64>(i) / keel::as<f64>(HN));
    }
    CHECK(sig::hilbert(hx, hy) == sig::SignalStatus::kOk);
    for (usize i = 0; i < HN; ++i) {
        CHECK_NEAR(hy[i].real(), hx[i], 1e-11);
        CHECK_NEAR(keel::abs(hy[i]), 1.0, 1e-11);
    }

    // Envelope (analytic, FIR, RMS)
    f64 up[HN]{}, lo[HN]{};
    CHECK(sig::envelope_analytic(hx, up, lo) == sig::SignalStatus::kOk);
    CHECK_NEAR(up[30], 1.0, 1e-10);
    CHECK_NEAR(lo[30], -1.0, 1e-10);
    CHECK(sig::envelope_fir(hx, 15, up, lo) == sig::SignalStatus::kOk);
    CHECK(up[30] > 0.85 && up[30] < 1.15);
    CHECK(sig::envelope_rms(hx, 12, up, lo) == sig::SignalStatus::kOk);
    CHECK_NEAR(up[30], kInvSqrt2, 0.05);
}

void test_smoothing_and_findpeaks() {
    // medfilt1 spike removal
    const f64 mx[7] = {1.0, 1.0, 100.0, 1.0, -50.0, 1.0, 1.0};
    f64 my[7]{};
    CHECK(sig::medfilt1(mx, 3, false, my) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 7; ++i) {
        CHECK_NEAR(my[i], 1.0, 1e-12);
    }

    // fillgaps on pure cosine with 3 missing samples
    constexpr usize GN = 40;
    f64 gx[GN]{}, gy[GN]{};
    for (usize i = 0; i < GN; ++i) gx[i] = keel::cos(0.4 * keel::as<f64>(i));
    gx[18] = Limits<f64>::quiet_nan();
    gx[19] = Limits<f64>::quiet_nan();
    gx[20] = Limits<f64>::quiet_nan();
    CHECK(sig::fillgaps(gx, 0, 4, gy) == sig::SignalStatus::kOk);
    for (usize i = 18; i <= 20; ++i) {
        CHECK_NEAR(gy[i], keel::cos(0.4 * keel::as<f64>(i)), 1e-2);
    }

    // sgolay & sgolayfilt preserve exact quadratic polynomial
    f64 sx[15]{}, sy[15]{};
    for (usize i = 0; i < 15; ++i) {
        const f64 t = keel::as<f64>(i);
        sx[i] = 2.0 - 3.0 * t + 0.5 * t * t;
    }
    CHECK(sig::sgolayfilt(sx, 2, 7, sy) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 15; ++i) {
        CHECK_NEAR(sy[i], sx[i], 1e-11);
    }

    // findpeaks with prominence & min distance
    const f64 px[11] = {0.0, 1.0, 5.0, 2.0, 3.0, 1.0, 0.0, 2.0, 8.0, 1.0, 0.0};
    sig::PeakOptions opts;
    opts.use_min_prom = true;
    opts.min_prom = 2.0;
    sig::PeakResult pr;
    CHECK(sig::findpeaks(px, {}, opts, pr) == sig::SignalStatus::kOk);
    CHECK(pr.values.size() == 2);
    CHECK_NEAR(pr.values[0], 5.0, 1e-12);
    CHECK_NEAR(pr.locations[0], 3.0, 1e-12);
    CHECK_NEAR(pr.values[1], 8.0, 1e-12);
    CHECK_NEAR(pr.locations[1], 9.0, 1e-12);
}

void test_spectral_and_stft() {
    // Periodogram & pwelch peak bin on tone at fs/8
    constexpr usize N = 256;
    const f64 fs = 256.0;
    f64 x[N]{};
    for (usize i = 0; i < N; ++i) {
        x[i] = keel::cos(kTau * 32.0 * keel::as<f64>(i) / fs);
    }
    f64 pxx[N / 2 + 1]{}, freq[N / 2 + 1]{};
    CHECK(sig::periodogram(x, {}, N, fs, pxx, freq) == sig::SignalStatus::kOk);
    usize peak_bin = 0;
    for (usize i = 1; i < N / 2 + 1; ++i) {
        if (pxx[i] > pxx[peak_bin]) peak_bin = i;
    }
    CHECK(peak_bin == 32);
    CHECK_NEAR(freq[32], 32.0, 1e-12);

    // mscohere(x, 2*x) == 1 and tfestimate(x, 2*x) == 2
    f64 y[N]{};
    for (usize i = 0; i < N; ++i) y[i] = 2.0 * x[i];
    f64 win[64]{};
    sig::hamming(win);
    f64 cxy[65]{}, cf[65]{};
    c64 txy[65]{};
    CHECK(sig::mscohere(x, y, win, 32, 128, fs, cxy, cf) == sig::SignalStatus::kOk);
    CHECK_NEAR(cxy[16], 1.0, 1e-10);
    CHECK(sig::tfestimate(x, y, win, 32, 128, fs, txy, cf) == sig::SignalStatus::kOk);
    CHECK_NEAR(txy[16].real(), 2.0, 1e-10);
    CHECK_NEAR(txy[16].imag(), 0.0, 1e-10);

    // pyulear & pburg peak near w = 2*pi*32/256 = pi/4
    f64 ar_p[129]{}, ar_f[129]{};
    CHECK(sig::pburg(x, 4, 256, ar_p, ar_f) == sig::SignalStatus::kOk);
    usize ar_peak = 0;
    for (usize i = 1; i < 129; ++i) {
        if (ar_p[i] > ar_p[ar_peak]) ar_peak = i;
    }
    CHECK(ar_peak >= 31 && ar_peak <= 33);

    // STFT + ISTFT exact roundtrip with periodic Hann window & 75% overlap + iscola
    f64 hann_p[64]{};
    for (usize i = 0; i < 64; ++i) {
        hann_p[i] = 0.5 * (1.0 - keel::cos(kTau * keel::as<f64>(i) / 64.0));
    }
    sig::ColaResult cola;
    CHECK(sig::iscola(hann_p, 48, false, cola) == sig::SignalStatus::kOk);
    CHECK(cola.is_cola);
    CHECK_NEAR(cola.median, 2.0, 1e-12);

    const usize n_frames = (N - 64) / 16 + 1;
    keel::Array<c64> S(64 * n_frames, c64(0.0, 0.0));
    CHECK(sig::stft(x, hann_p, 48, 64, sig::StftRange::kTwoSided, S.as_mut_ref()) ==
          sig::SignalStatus::kOk);
    f64 x_rec[N]{};
    CHECK(sig::istft(S.as_ref(), 64, n_frames, hann_p, 48, 64, sig::StftRange::kTwoSided, x_rec) ==
          sig::SignalStatus::kOk);
    for (usize i = 32; i < N - 32; ++i) {
        CHECK_NEAR(x_rec[i], x[i], 1e-12);
    }
}

void test_iir_design() {
    // Analog prototypes
    sig::ZpkResult zpk;
    CHECK(sig::buttap(4, zpk) == sig::SignalStatus::kOk);
    CHECK(zpk.z.empty());
    CHECK(zpk.p.size() == 4);
    CHECK_NEAR(zpk.k, 1.0, 1e-12);
    for (usize i = 0; i < 4; ++i) {
        CHECK(zpk.p[i].real() < 0.0);
        CHECK_NEAR(keel::abs(zpk.p[i]), 1.0, 1e-12);
    }

    CHECK(sig::cheb1ap(3, 1.0, zpk) == sig::SignalStatus::kOk);
    CHECK(zpk.z.empty() && zpk.p.size() == 3);
    CHECK(sig::cheb2ap(4, 40.0, zpk) == sig::SignalStatus::kOk);
    CHECK(zpk.z.size() == 4 && zpk.p.size() == 4);
    CHECK(sig::besselap(3, zpk) == sig::SignalStatus::kOk);
    CHECK(zpk.z.empty() && zpk.p.size() == 3);
    CHECK(sig::ellipap(4, 1.0, 40.0, zpk) == sig::SignalStatus::kOk);
    CHECK(zpk.z.size() == 4 && zpk.p.size() == 4);

    // lp2lp / lp2hp / lp2bp / lp2bs
    sig::ZpkResult bp_zpk, tx;
    CHECK(sig::buttap(2, bp_zpk) == sig::SignalStatus::kOk);
    CHECK(sig::lp2lp(bp_zpk.z.as_ref(), bp_zpk.p.as_ref(), bp_zpk.k, 10.0, tx) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(keel::abs(tx.p[0]), 10.0, 1e-12);
    CHECK(sig::lp2hp(bp_zpk.z.as_ref(), bp_zpk.p.as_ref(), bp_zpk.k, 5.0, tx) ==
          sig::SignalStatus::kOk);
    CHECK(tx.z.size() == 2 && tx.p.size() == 2);
    CHECK(sig::lp2bp(bp_zpk.z.as_ref(), bp_zpk.p.as_ref(), bp_zpk.k, 10.0, 2.0, tx) ==
          sig::SignalStatus::kOk);
    CHECK(tx.z.size() == 2 && tx.p.size() == 4);
    CHECK(sig::lp2bs(bp_zpk.z.as_ref(), bp_zpk.p.as_ref(), bp_zpk.k, 10.0, 2.0, tx) ==
          sig::SignalStatus::kOk);
    CHECK(tx.z.size() == 4 && tx.p.size() == 4);

    // bilinear, impinvar, freqs on 1st-order RC H(s) = 1 / (s + 1)
    const f64 b1[1] = {1.0};
    const f64 a1[2] = {1.0, 1.0};
    sig::TfResult tf;
    CHECK(sig::bilinear(b1, a1, 10.0, 0.0, tf) == sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 2 && tf.a.size() == 2);
    // DC gain = (b0+b1)/(a0+a1) == 1
    CHECK_NEAR((tf.b[0] + tf.b[1]) / (tf.a[0] + tf.a[1]), 1.0, 1e-12);

    CHECK(sig::impinvar(b1, a1, 10.0, tf) == sig::SignalStatus::kOk);
    CHECK(tf.b.size() >= 1 && tf.a.size() == 2);
    CHECK_NEAR(tf.a[1], -keel::exp(-0.1), 1e-12);

    const f64 w[2] = {0.0, 1.0};
    c64 h[2]{};
    CHECK(sig::freqs(b1, a1, w, h) == sig::SignalStatus::kOk);
    CHECK_NEAR(h[0].real(), 1.0, 1e-12);
    CHECK_NEAR(keel::abs(h[1]), 1.0 / keel::kSqrt2, 1e-12);

    // Digital Butterworth lowpass: DC gain = 1, Nyquist gain = 0, -3dB at Wn
    const f64 wn_lp[1] = {0.4};
    CHECK(sig::butter(4, wn_lp, sig::FilterType::kLowpass, false, tf) == sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 5 && tf.a.size() == 5);
    f64 sum_b = 0.0, sum_a = 0.0, alt_b = 0.0;
    for (usize i = 0; i < 5; ++i) {
        sum_b += tf.b[i];
        sum_a += tf.a[i];
        alt_b += (i % 2 == 0 ? 1.0 : -1.0) * tf.b[i];
    }
    CHECK_NEAR(sum_b / sum_a, 1.0, 1e-12);
    CHECK_NEAR(alt_b, 0.0, 1e-12);

    // Chebyshev I/II, Elliptic bandpass, Bessel analog
    const f64 wn_bp[2] = {0.2, 0.5};
    CHECK(sig::cheby1(3, 1.0, wn_lp, sig::FilterType::kHighpass, false, tf) ==
          sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 4 && tf.a.size() == 4);
    CHECK(sig::cheby2(3, 40.0, wn_lp, sig::FilterType::kLowpass, false, tf) ==
          sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 4 && tf.a.size() == 4);
    CHECK(sig::ellip(2, 1.0, 40.0, wn_bp, sig::FilterType::kBandpass, false, tf) ==
          sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 5 && tf.a.size() == 5);
    const f64 wn_an[1] = {100.0};
    CHECK(sig::besself(3, wn_an, sig::FilterType::kLowpass, true, tf) == sig::SignalStatus::kOk);
    CHECK(tf.b.size() == 4 && tf.a.size() == 4);

    // Order estimators
    const f64 wp[1] = {0.2}, ws[1] = {0.3};
    sig::OrderResult ord;
    CHECK(sig::buttord(wp, ws, 3.0, 40.0, false, ord) == sig::SignalStatus::kOk);
    CHECK(ord.order >= 10 && ord.wn_count == 1);
    sig::OrderResult ord_c1, ord_c2, ord_el;
    CHECK(sig::cheb1ord(wp, ws, 3.0, 40.0, false, ord_c1) == sig::SignalStatus::kOk);
    CHECK(sig::cheb2ord(wp, ws, 3.0, 40.0, false, ord_c2) == sig::SignalStatus::kOk);
    CHECK(sig::ellipord(wp, ws, 3.0, 40.0, false, ord_el) == sig::SignalStatus::kOk);
    CHECK(ord_el.order <= ord_c1.order && ord_c1.order <= ord.order);

    const f64 f_fir[2] = {1500.0, 2000.0};
    const f64 a_fir[2] = {1.0, 0.0};
    const f64 dev_fir[2] = {0.01, 0.1};
    sig::FirpmOrdResult pm;
    CHECK(sig::firpmord(f_fir, a_fir, dev_fir, 8000.0, pm) == sig::SignalStatus::kOk);
    CHECK(pm.order > 0 && pm.f.size() == 4 && pm.a.size() == 4 && pm.w.size() == 2);

    sig::KaiserOrdResult ko;
    CHECK(sig::kaiserord(f_fir, a_fir, dev_fir, 8000.0, ko) == sig::SignalStatus::kOk);
    CHECK(ko.order > 0 && ko.wn.size() == 1 && ko.ftype == sig::KaiserBandType::kLow);
}

void test_multirate() {
    // upfirdn: p=1, q=1 matches linear convolution
    const f64 x[4] = {1.0, 2.0, 3.0, 4.0};
    const f64 h[3] = {0.5, 1.0, 0.5};
    f64 conv_ref[6]{};
    CHECK(sig::conv_1d(x, h, conv_ref) == sig::SignalStatus::kOk);
    f64 y1[6]{};
    CHECK(sig::upfirdn_length(4, 3, 1, 1) == 6);
    CHECK(sig::upfirdn(x, h, 1, 1, y1) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 6; ++i) {
        CHECK_NEAR(y1[i], conv_ref[i], 1e-12);
    }

    // upfirdn p=3, q=2 matches explicit upsample -> conv -> downsample
    const usize p = 3, q = 2;
    const usize up_len = (4 - 1) * p + 1;
    f64 x_up[10]{};
    for (usize i = 0; i < 4; ++i) x_up[i * p] = x[i];
    f64 full[12]{};
    CHECK(sig::conv_1d(keel::ArrayRef<f64>(x_up, up_len), h, full) == sig::SignalStatus::kOk);
    const usize out_len = sig::upfirdn_length(4, 3, p, q);
    CHECK(out_len == 6);
    f64 ypq[6]{};
    CHECK(sig::upfirdn(x, h, p, q, ypq) == sig::SignalStatus::kOk);
    for (usize i = 0; i < out_len; ++i) {
        CHECK_NEAR(ypq[i], full[i * q], 1e-12);
    }

    // upfirdn pure decimation p=1, q=4 with longer filter matches explicit conv
    const f64 xd[12] = {1, -2, 3, 4, -1, 2, 5, -3, 0.5, 1.5, -0.5, 2.5};
    const f64 hd[7] = {0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1};
    f64 cdec[18]{};
    CHECK(sig::conv_1d(xd, hd, cdec) == sig::SignalStatus::kOk);
    const usize dec_len = sig::upfirdn_length(12, 7, 1, 4);
    CHECK(dec_len == 5);
    f64 ydec[5]{};
    CHECK(sig::upfirdn(xd, hd, 1, 4, ydec) == sig::SignalStatus::kOk);
    for (usize i = 0; i < dec_len; ++i) {
        CHECK_NEAR(ydec[i], cdec[i * 4], 1e-12);
    }

    // fftfilt matches direct FIR filter_df2t (real and complex)
    f64 sig_x[100]{};
    for (usize i = 0; i < 100; ++i) {
        sig_x[i] = keel::sin(0.13 * keel::as<f64>(i)) + 0.3 * keel::cos(0.7 * keel::as<f64>(i));
    }
    const f64 b_fir[7] = {0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05};
    const f64 a_one[1] = {1.0};
    f64 y_ref[100]{}, y_fft[100]{};
    CHECK(sig::filter_df2t(b_fir, a_one, sig_x, y_ref) == sig::SignalStatus::kOk);
    CHECK(sig::fftfilt(b_fir, sig_x, 0, y_fft) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 100; ++i) {
        CHECK_NEAR(y_fft[i], y_ref[i], 1e-12);
    }

    c64 b_c[3] = {c64(1.0, 0.5), c64(0.5, -0.25), c64(0.25, 0.1)};
    c64 x_c[40]{}, y_cref[40]{}, y_cfft[40]{};
    const c64 a_cone[1] = {c64(1.0, 0.0)};
    for (usize i = 0; i < 40; ++i) {
        x_c[i] = c64(keel::cos(0.2 * keel::as<f64>(i)), keel::sin(0.3 * keel::as<f64>(i)));
    }
    CHECK(sig::filter_df2t_c64(b_c, a_cone, x_c, y_cref) == sig::SignalStatus::kOk);
    CHECK(sig::fftfilt_c64(b_c, x_c, 16, y_cfft) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 40; ++i) {
        CHECK_NEAR(y_cfft[i].real(), y_cref[i].real(), 1e-12);
        CHECK_NEAR(y_cfft[i].imag(), y_cref[i].imag(), 1e-12);
    }
}

void test_fir_design() {
    // 1. fir1_single lowpass DC gain == 1 and symmetry
    f64 h1[21]{};
    CHECK(sig::fir1_single(20, 0.4, false, h1) == sig::SignalStatus::kOk);
    f64 sum1 = 0.0;
    for (usize i = 0; i < 21; ++i) sum1 += h1[i];
    CHECK_NEAR(sum1, 1.0, 1e-12);
    CHECK_NEAR(h1[0], h1[20], 1e-12);

    // fir1_multiband bandpass
    const f64 wn_bp[2] = {0.2, 0.6};
    keel::Array<f64> h_bp;
    CHECK(sig::fir1_multiband(24, wn_bp, false, {}, true, h_bp) == sig::SignalStatus::kOk);
    CHECK(h_bp.size() == 25);
    CHECK_NEAR(h_bp[0], h_bp[24], 1e-12);

    // 2. fir2 lowpass
    const f64 f2[4] = {0.0, 0.4, 0.4, 1.0};
    const f64 m2[4] = {1.0, 1.0, 0.0, 0.0};
    keel::Array<f64> h2;
    CHECK(sig::fir2(30, f2, m2, 0, 0, {}, h2) == sig::SignalStatus::kOk);
    CHECK(h2.size() == 31);
    f64 dc2 = 0.0;
    for (usize i = 0; i < h2.size(); ++i) dc2 += h2[i];
    CHECK_NEAR(dc2, 1.0, 0.05);

    // 3. firls Type I lowpass
    const f64 fls[4] = {0.0, 0.3, 0.4, 1.0};
    const f64 als[4] = {1.0, 1.0, 0.0, 0.0};
    f64 hls[21]{};
    CHECK(sig::firls(20, fls, als, {}, sig::FirFtype::kStandard, hls) == sig::SignalStatus::kOk);
    CHECK_NEAR(hls[0], hls[20], 1e-12);
    f64 dcls = 0.0;
    for (usize i = 0; i < 21; ++i) dcls += hls[i];
    CHECK_NEAR(dcls, 1.0, 0.05);

    // 4. firpm Type I lowpass & Type IV differentiator
    f64 hpm[21]{};
    f64 err = 0.0;
    CHECK(sig::firpm(20, fls, als, {}, sig::FirFtype::kStandard, hpm, err) ==
          sig::SignalStatus::kOk);
    CHECK(err > 0.0 && err < 0.15);
    CHECK_NEAR(hpm[0], hpm[20], 1e-12);
    f64 hpm40[41]{};
    f64 err40 = 0.0;
    CHECK(sig::firpm(40, fls, als, {}, sig::FirFtype::kStandard, hpm40, err40) ==
          sig::SignalStatus::kOk);
    CHECK(err40 > 0.0 && err40 < err);

    const f64 fdiff[2] = {0.1, 0.9};
    const f64 adiff[2] = {0.1, 0.9};
    f64 hdiff[16]{};
    CHECK(sig::firpm(15, fdiff, adiff, {}, sig::FirFtype::kDifferentiator, hdiff, err) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(hdiff[0], -hdiff[15], 1e-12);
}

void test_filter_analysis_and_poly() {
    // 1. filtfilt zero-phase property on symmetric signal
    const f64 wn = 0.25;
    sig::TfResult tf;
    CHECK(sig::butter(4, keel::ArrayRef<f64>(&wn, 1), sig::FilterType::kLowpass, false, tf) ==
          sig::SignalStatus::kOk);
    keel::Array<f64> x(128, 0.0);
    for (usize i = 0; i < 128; ++i) {
        x[i] = keel::sin(2.0 * keel::kPi * 0.03 * keel::as<f64>(i)) +
               0.5 * keel::sin(2.0 * keel::kPi * 0.4 * keel::as<f64>(i));
    }
    keel::Array<f64> y(128, 0.0);
    CHECK(sig::filtfilt(tf.b.as_ref(), tf.a.as_ref(), x.as_ref(), y.as_mut_ref()) ==
          sig::SignalStatus::kOk);
    // Low-frequency component should remain in phase at center samples
    CHECK_NEAR(y[64], keel::sin(2.0 * keel::kPi * 0.03 * 64.0), 0.15);

    // 2. freqz / phasez / grpdelay / phasedelay / zerophase
    const f64 b_fir[5] = {0.2, 0.2, 0.2, 0.2, 0.2};
    const f64 a_fir[1] = {1.0};
    c64 h_grid[16]{};
    f64 w_grid[16]{};
    CHECK(sig::freqz_grid(b_fir, a_fir, 16, false, 0.0, h_grid, w_grid) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(h_grid[0].real(), 1.0, 1e-12);
    CHECK_NEAR(h_grid[0].imag(), 0.0, 1e-12);

    f64 gd[16]{};
    CHECK(sig::grpdelay(b_fir, a_fir, 16, 0.0, gd, w_grid) == sig::SignalStatus::kOk);
    CHECK_NEAR(gd[0], 2.0, 1e-12);
    CHECK_NEAR(gd[1], 2.0, 1e-12);

    f64 pd[16]{};
    CHECK(sig::phasedelay(b_fir, a_fir, 16, pd, w_grid) == sig::SignalStatus::kOk);
    CHECK(keel::is_nan(pd[0]));
    CHECK_NEAR(pd[1], 2.0, 1e-12);

    f64 hr[16]{};
    CHECK(sig::zerophase(b_fir, a_fir, 16, hr, w_grid) == sig::SignalStatus::kOk);
    CHECK_NEAR(hr[0], 1.0, 1e-12);

    // 3. impzlength / impz / stepz
    usize ilen = 0;
    CHECK(sig::impzlength(tf.b.as_ref(), tf.a.as_ref(), ilen) == sig::SignalStatus::kOk);
    CHECK(ilen > 10 && ilen <= 8192);
    keel::Array<f64> hi(32, 0.0);
    keel::Array<f64> ti(32, 0.0);
    CHECK(sig::impz(b_fir, a_fir, 32, hi.as_mut_ref(), ti.as_mut_ref()) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(hi[0], 0.2, 1e-12);
    CHECK_NEAR(hi[4], 0.2, 1e-12);
    CHECK_NEAR(hi[5], 0.0, 1e-12);
    CHECK(sig::stepz(b_fir, a_fir, 32, hi.as_mut_ref(), ti.as_mut_ref()) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(hi[10], 1.0, 1e-12);

    // 4. polyscale / polystab
    const f64 p_in[3] = {1.0, -3.0, 2.0}; // roots 1, 2
    const f64 sc = 0.5;
    f64 p_sc[3]{};
    CHECK(sig::polyscale(p_in, 1, 3, keel::ArrayRef<f64>(&sc, 1), p_sc) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(p_sc[0], 1.0, 1e-12);
    CHECK_NEAR(p_sc[1], -1.5, 1e-12);
    CHECK_NEAR(p_sc[2], 0.5, 1e-12);

    keel::Array<f64> p_st;
    CHECK(sig::polystab(p_in, p_st) == sig::SignalStatus::kOk);
    // Root 2 reflects to 0.5 -> (z - 1)(z - 0.5) = z^2 - 1.5z + 0.5
    CHECK(p_st.size() == 3);
    CHECK_NEAR(p_st[0], 1.0, 1e-9);
    CHECK_NEAR(p_st[1], -1.5, 1e-9);
    CHECK_NEAR(p_st[2], 0.5, 1e-9);
}

void test_conversions() {
    const f64 wn = 0.3;
    sig::TfResult tf;
    CHECK(sig::butter(4, keel::ArrayRef<f64>(&wn, 1), sig::FilterType::kLowpass, false, tf) ==
          sig::SignalStatus::kOk);

    // tf2sos -> sos2tf roundtrip
    sig::SosResult sos;
    CHECK(sig::tf2sos(tf.b.as_ref(), tf.a.as_ref(), sos) == sig::SignalStatus::kOk);
    CHECK(sos.num_sections == 2);

    sig::TfResult tf2;
    CHECK(sig::sos2tf(sos.sos_col_major.as_ref(), sos.num_sections, sos.gain, tf2) ==
          sig::SignalStatus::kOk);
    CHECK(tf2.b.size() == tf.b.size());
    CHECK(tf2.a.size() == tf.a.size());
    for (usize i = 0; i < tf.b.size(); ++i) {
        CHECK_NEAR(tf2.b[i], tf.b[i], 1e-4); // 4-fold zero at z=-1
        CHECK_NEAR(tf2.a[i], tf.a[i], 1e-9);
    }

    // sos2zp
    sig::ZpkResult zpk;
    CHECK(sig::sos2zp(sos.sos_col_major.as_ref(), sos.num_sections, sos.gain, zpk) ==
          sig::SignalStatus::kOk);
    CHECK(zpk.p.size() == 4);

    // tf2ss -> ss2tf roundtrip
    sig::StateSpaceResult ss;
    CHECK(sig::tf2ss(tf.b.as_ref(), tf.a.as_ref(), ss) == sig::SignalStatus::kOk);
    CHECK(ss.order == 4);
    sig::TfResult tf3;
    CHECK(sig::ss2tf(ss.a_col_major.as_ref(), ss.order, ss.b.as_ref(), ss.c.as_ref(), ss.d, tf3) ==
          sig::SignalStatus::kOk);
    for (usize i = 0; i < tf.b.size(); ++i) {
        CHECK_NEAR(tf3.b[i], tf.b[i], 1e-8);
        CHECK_NEAR(tf3.a[i], tf.a[i], 1e-8);
    }
}

void test_conv_xcorr_transforms_multirate() {
    // cconv & deconv
    const f64 a[3] = {1.0, 2.0, 3.0};
    const f64 b[2] = {4.0, 5.0};
    f64 cc[4]{};
    CHECK(sig::cconv(a, b, 0, cc) == sig::SignalStatus::kOk);
    CHECK_NEAR(cc[0], 4.0, 1e-12);
    CHECK_NEAR(cc[1], 13.0, 1e-12);
    CHECK_NEAR(cc[2], 22.0, 1e-12);
    CHECK_NEAR(cc[3], 15.0, 1e-12);

    keel::Array<f64> q, r;
    CHECK(sig::deconv(cc, b, q, r) == sig::SignalStatus::kOk);
    CHECK(q.size() == 3);
    CHECK_NEAR(q[0], 1.0, 1e-12);
    CHECK_NEAR(q[1], 2.0, 1e-12);
    CHECK_NEAR(q[2], 3.0, 1e-12);

    // xcorr / finddelay / xcov / xcorr2
    const f64 s1[8] = {0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    const f64 s2[8] = {0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0};
    i64 delay = 0;
    CHECK(sig::finddelay(s1, s2, 0, delay) == sig::SignalStatus::kOk);
    CHECK(delay == 2);

    keel::Array<f64> xc, xlags;
    CHECK(sig::xcov(s1, s1, 3, sig::XcorrScale::kCoeff, xc, xlags) == sig::SignalStatus::kOk);
    CHECK(xc.size() == 7);
    CHECK_NEAR(xc[3], 1.0, 1e-12);

    f64 m2[9]{};
    const f64 m1[4] = {1.0, 2.0, 3.0, 4.0};
    CHECK(sig::xcorr2(m1, 2, 2, m1, 2, 2, m2) == sig::SignalStatus::kOk);
    CHECK_NEAR(m2[4], 30.0, 1e-12);

    // dst / idst
    f64 ds[4]{};
    f64 ids[4]{};
    const f64 v4[4] = {1.0, 2.0, 3.0, 4.0};
    CHECK(sig::dst(v4, ds) == sig::SignalStatus::kOk);
    CHECK(sig::idst(ds, ids) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 4; ++i) CHECK_NEAR(ids[i], v4[i], 1e-12);

    // cceps / icceps roundtrip & rceps
    const f64 seq[5] = {1.0, 0.5, 0.25, 0.1, 0.0};
    f64 ce[5]{};
    f64 rec[5]{};
    f64 nd = 0.0;
    CHECK(sig::cceps(seq, ce, &nd) == sig::SignalStatus::kOk);
    CHECK(sig::icceps(ce, rec) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 5; ++i) CHECK_NEAR(rec[i], seq[i], 1e-10);

    f64 rc[5]{};
    f64 mp[5]{};
    CHECK(sig::rceps_minphase(seq, rc, mp) == sig::SignalStatus::kOk);
    CHECK(mp[0] > 0.0);

    // czt matches DFT when w = exp(-2*pi*i/N), a = 1
    c64 cx[4] = {c64(1.0, 0.0), c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, 0.0)};
    c64 cy[4]{};
    CHECK(sig::czt(cx, 4, keel::polar(1.0, -keel::kTau / 4.0), c64(1.0, 0.0), cy) ==
          sig::SignalStatus::kOk);
    CHECK_NEAR(cy[0].real(), 10.0, 1e-11);
    CHECK_NEAR(cy[1].real(), -2.0, 1e-11);
    CHECK_NEAR(cy[1].imag(), 2.0, 1e-11);

    // decimate & resample
    keel::Array<f64> sig64(64, 1.0);
    keel::Array<f64> dec_out, res_y, res_b;
    CHECK(sig::decimate(sig64.as_ref(), 2, dec_out) == sig::SignalStatus::kOk);
    CHECK(dec_out.size() == 32);
    // Order-8 Chebyshev I (Rp=0.05 dB) zero-phase DC gain = 10^(-0.1/20) ~= 0.98855
    CHECK_NEAR(dec_out[16], 0.988553, 1e-3);

    CHECK(sig::resample(sig64.as_ref(), 3, 2, 10, 5.0, res_y, &res_b) == sig::SignalStatus::kOk);
    CHECK(res_y.size() == 96);
    CHECK_NEAR(res_y[48], 1.0, 1e-2);
}

void test_modeling_and_subspace() {
    // AR(2) process: x[n] = 0.75*x[n-1] - 0.5*x[n-2] + impulse/excitation
    keel::Array<f64> x(256, 0.0);
    x[0] = 1.0;
    for (usize i = 1; i < 256; ++i) {
        const f64 prev1 = x[i - 1];
        const f64 prev2 = (i >= 2) ? x[i - 2] : 0.0;
        x[i] = 0.75 * prev1 - 0.5 * prev2;
    }

    sig::LevinsonResult yw, bg;
    CHECK(sig::aryule(x.as_ref(), 2, yw) == sig::SignalStatus::kOk);
    CHECK(yw.a.size() == 3);
    CHECK_NEAR(yw.a[1], -0.75, 0.05);
    CHECK_NEAR(yw.a[2], 0.5, 0.05);

    // Burg AR(2) on pure sinusoid cos(2*pi*0.1*n) -> exact a = [1, -2*cos(0.2*pi), 1]
    keel::Array<f64> sin2(128, 0.0);
    for (usize i = 0; i < 128; ++i) {
        sin2[i] = keel::cos(keel::kTau * 0.1 * keel::as<f64>(i));
    }
    CHECK(sig::arburg(sin2.as_ref(), 2, bg) == sig::SignalStatus::kOk);
    CHECK_NEAR(bg.a[1], -2.0 * keel::cos(keel::kTau * 0.1), 1e-2);
    CHECK_NEAR(bg.a[2], 1.0, 1e-3);

    keel::Array<f64> ac_a;
    f64 ac_e = 0.0;
    CHECK(sig::arcov(sin2.as_ref(), 2, ac_a, ac_e) == sig::SignalStatus::kOk);
    CHECK_NEAR(ac_a[1], -2.0 * keel::cos(keel::kTau * 0.1), 1e-10);
    CHECK_NEAR(ac_a[2], 1.0, 1e-10);

    // poly2rc / rc2poly / poly2lsf / lsf2poly / rlevinson roundtrips
    keel::Array<f64> k_rc, a_rec, lsf, a_lsf;
    f64 r0 = 0.0;
    f64 ef = 0.0;
    const f64 a_true[3] = {1.0, -0.75, 0.5};
    CHECK(sig::poly2rc(a_true, 0.25, k_rc, r0) == sig::SignalStatus::kOk);
    CHECK(sig::rc2poly(k_rc.as_ref(), r0, a_rec, ef) == sig::SignalStatus::kOk);
    CHECK_NEAR(ef, 0.25, 1e-12);
    for (usize i = 0; i < 3; ++i) CHECK_NEAR(a_rec[i], a_true[i], 1e-12);

    CHECK(sig::poly2lsf(a_true, lsf) == sig::SignalStatus::kOk);
    CHECK(lsf.size() == 2);
    CHECK(sig::lsf2poly(lsf.as_ref(), a_lsf) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 3; ++i) CHECK_NEAR(a_lsf[i], a_true[i], 1e-9);

    sig::RlevinsonResult rlev;
    CHECK(sig::rlevinson(a_true, 0.25, rlev) == sig::SignalStatus::kOk);
    sig::LevinsonResult lev_back;
    CHECK(sig::levinson(rlev.r.as_ref(), 2, lev_back) == sig::SignalStatus::kOk);
    for (usize i = 0; i < 3; ++i) CHECK_NEAR(lev_back.a[i], a_true[i], 1e-11);

    // prony & stmcb recover impulse response of 1 / (1 - 0.75 z^-1 + 0.5 z^-2)
    sig::TfResult pr_res, st_res;
    CHECK(sig::prony(x.as_ref(), 0, 2, pr_res) == sig::SignalStatus::kOk);
    CHECK_NEAR(pr_res.b[0], 1.0, 1e-9);
    CHECK_NEAR(pr_res.a[1], -0.75, 1e-9);
    CHECK_NEAR(pr_res.a[2], 0.5, 1e-9);

    CHECK(sig::stmcb(x.as_ref(), {}, 0, 2, 5, {}, st_res) == sig::SignalStatus::kOk);
    CHECK_NEAR(st_res.a[1], -0.75, 1e-9);
    CHECK_NEAR(st_res.a[2], 0.5, 1e-9);

    // pmusic detects a single tone at 100 Hz (fs = 1000 Hz)
    keel::Array<f64> tone(256, 0.0);
    for (usize i = 0; i < 256; ++i) {
        tone[i] = keel::cos(keel::kTau * 100.0 * keel::as<f64>(i) / 1000.0);
    }
    f64 pm[129]{};
    f64 fm[129]{};
    CHECK(sig::pmusic(tone.as_ref(), 2, 256, 1000.0, pm, fm) == sig::SignalStatus::kOk);
    usize best_k = 0;
    for (usize k = 1; k < 129; ++k) {
        if (pm[k] > pm[best_k]) best_k = k;
    }
    CHECK_NEAR(fm[best_k], 100.0, 10.0);
}

void test_spectral_metrics_and_vibration() {
    // Pure tone at 200 Hz, fs = 1000 Hz
    const usize n = 1024;
    const f64 fs = 1000.0;
    keel::Array<f64> x(n, 0.0);
    for (usize i = 0; i < n; ++i) {
        x[i] = keel::sin(keel::kTau * 200.0 * keel::as<f64>(i) / fs);
    }

    f64 mf = 0.0;
    f64 medf = 0.0;
    f64 bp = 0.0;
    CHECK(sig::meanfreq_time(x.as_ref(), fs, mf) == sig::SignalStatus::kOk);
    CHECK(sig::medfreq_time(x.as_ref(), fs, medf) == sig::SignalStatus::kOk);
    CHECK(sig::bandpower_time(x.as_ref(), fs, 150.0, 250.0, bp) == sig::SignalStatus::kOk);
    CHECK_NEAR(mf, 200.0, 5.0);
    CHECK_NEAR(medf, 200.0, 5.0);
    CHECK_NEAR(bp, 0.5, 0.05);

    sig::ObwResult obw_r;
    CHECK(sig::obw_time(x.as_ref(), fs, 0.99, obw_r) == sig::SignalStatus::kOk);
    CHECK(obw_r.flo < 200.0 && obw_r.fhi > 200.0);

    // pspectrum power mode peak at 200 Hz
    sig::PspectrumOptions opts;
    opts.mode = sig::PspectrumMode::kPower;
    opts.is_norm_freq = false;
    opts.fs = fs;
    sig::PspectrumResult ps;
    CHECK(sig::pspectrum_real(x.as_ref(), n, 1, opts, ps) == sig::SignalStatus::kOk);
    CHECK(ps.n_freq == 4096 && ps.n_cols == 1);
    usize pk = 0;
    for (usize k = 1; k < ps.n_freq; ++k) {
        if (ps.p_col_major[k] > ps.p_col_major[pk]) pk = k;
    }
    CHECK_NEAR(ps.f[pk], 200.0, 2.0);

    // rainflow simple sequence
    const f64 rf_in[7] = {0.0, 2.0, -1.0, 3.0, -2.0, 1.0, 0.0};
    keel::Array<sig::RainflowCycle> cycles;
    CHECK(sig::rainflow(rf_in, cycles) == sig::SignalStatus::kOk);
    CHECK(cycles.size() > 0);
}

void test_waveforms_and_pulse_roi() {
    // sinc / square / sawtooth / chirp / diric
    const f64 t5[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    f64 out5[5]{};
    CHECK(sig::sinc(t5, out5) == sig::SignalStatus::kOk);
    CHECK_NEAR(out5[2], 1.0, 1e-12);
    CHECK_NEAR(out5[0], 0.0, 1e-12);

    // modulate + demod roundtrip (PM)
    const usize n = 256;
    keel::Array<f64> msg(n, 0.0);
    keel::Array<f64> mod_y(n, 0.0);
    keel::Array<f64> dem_x(n, 0.0);
    for (usize i = 0; i < n; ++i) {
        msg[i] = 0.3 * keel::sin(keel::kTau * 5.0 * keel::as<f64>(i) / 1000.0);
    }
    CHECK(sig::modulate(msg.as_ref(), n, 1, 100.0, 1000.0, sig::ModulationMethod::kPm, true, 1.0,
                        mod_y.as_mut_ref()) == sig::SignalStatus::kOk);
    CHECK(sig::demod(mod_y.as_ref(), n, 1, 100.0, 1000.0, sig::ModulationMethod::kPm, true, 1.0,
                     dem_x.as_mut_ref()) == sig::SignalStatus::kOk);
    CHECK_NEAR(dem_x[128], msg[128], 0.05);

    // Pulse metrics on clean square wave
    keel::Array<f64> pulse(100, 0.0);
    for (usize i = 20; i < 60; ++i) pulse[i] = 5.0;
    sig::StateLevels sl;
    CHECK(sig::statelevels(pulse.as_ref(), sl) == sig::SignalStatus::kOk);
    CHECK_NEAR(sl.low, 0.0, 0.1);
    CHECK_NEAR(sl.high, 5.0, 0.1);

    keel::Array<f64> pw;
    CHECK(sig::pulsewidth(pulse.as_ref(), 1.0, pw) == sig::SignalStatus::kOk);
    CHECK(pw.size() == 1);
    CHECK_NEAR(pw[0], 40.0, 0.5);

    // SigROI roundtrip
    const u8 mask[10] = {0, 1, 1, 1, 0, 0, 1, 1, 0, 0};
    keel::Array<sig::SigRoi> rois, merged;
    CHECK(sig::binmask2sigroi(mask, rois) == sig::SignalStatus::kOk);
    CHECK(rois.size() == 2);
    CHECK(rois[0].start == 2 && rois[0].end == 4);
    CHECK(rois[1].start == 7 && rois[1].end == 8);
    CHECK(sig::mergesigroi(rois.as_ref(), 2, merged) == sig::SignalStatus::kOk);
    CHECK(merged.size() == 1);
    CHECK(merged[0].start == 2 && merged[0].end == 8);
}

} // namespace

int main() {
    test_windows();
    test_conv();
    test_iir_and_sos();
    test_goertzel_and_unwrap();
    test_dct_fwht_hilbert();
    test_smoothing_and_findpeaks();
    test_spectral_and_stft();
    test_iir_design();
    test_multirate();
    test_fir_design();
    test_filter_analysis_and_poly();
    test_conversions();
    test_conv_xcorr_transforms_multirate();
    test_modeling_and_subspace();
    test_spectral_metrics_and_vibration();
    test_waveforms_and_pulse_roi();

    if (g_failures == 0) {
        std::printf("ALL keel::signal TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
