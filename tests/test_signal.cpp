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

} // namespace

int main() {
    test_windows();
    test_conv();
    test_iir_and_sos();
    test_goertzel_and_unwrap();
    test_dct_fwht_hilbert();
    test_smoothing_and_findpeaks();
    test_spectral_and_stft();

    if (g_failures == 0) {
        std::printf("ALL keel::signal TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "TOTAL FAILURES: %d\n", g_failures);
    return 1;
}
