#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/math.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::signal {

enum class SignalStatus : u8 {
    kOk = 0,
    kInvalidArgument,
    kZeroLeadingCoeff,
    kSingular,
};

struct BiquadState {
    f64 s1 = 0.0;
    f64 s2 = 0.0;
};

enum class FwhtOrdering : u8 {
    kSequency = 0,
    kHadamard,
    kDyadic,
};

enum class PeakWidthRef : u8 {
    kHalfProm = 0,
    kHalfHeight,
};

enum class PeakSort : u8 {
    kNone = 0,
    kAscend,
    kDescend,
};

enum class StftRange : u8 {
    kTwoSided = 0,
    kOneSided,
    kCentered,
};

// ============================================================================
// Window Functions
// ============================================================================
// All window functions populate `out` (`out.size()` samples; no-op when empty).

void rectwin(MutArrayRef<f64> out) noexcept;
void hann(MutArrayRef<f64> out) noexcept;
void hamming(MutArrayRef<f64> out) noexcept;
void blackman(MutArrayRef<f64> out) noexcept;
void bartlett(MutArrayRef<f64> out) noexcept;
void triang(MutArrayRef<f64> out) noexcept;
void kaiser(MutArrayRef<f64> out, f64 beta) noexcept;
void tukeywin(MutArrayRef<f64> out, f64 r) noexcept;
void flattopwin(MutArrayRef<f64> out) noexcept;
void parzenwin(MutArrayRef<f64> out) noexcept;
void nuttallwin(MutArrayRef<f64> out) noexcept;
void blackmanharris(MutArrayRef<f64> out) noexcept;
void bohmanwin(MutArrayRef<f64> out) noexcept;
void barthannwin(MutArrayRef<f64> out) noexcept;

// Gaussian window: returns kInvalidArgument if alpha <= 0.
[[nodiscard]] SignalStatus gausswin(MutArrayRef<f64> out, f64 alpha) noexcept;

// Dolph-Chebyshev window with `at_db` dB sidelobe attenuation (> 0).
[[nodiscard]] SignalStatus chebwin(MutArrayRef<f64> out, f64 at_db, Allocator* alloc = nullptr);

// Taylor window: requires nbar >= 2 and sll_db < 0 (peak sidelobe in dB).
[[nodiscard]] SignalStatus taylorwin(MutArrayRef<f64> out, i32 nbar, f64 sll_db,
                                     Allocator* alloc = nullptr);

// Discrete prolate spheroidal sequences (Slepian tapers).
// Computes `k_eff = min(ratios.size(), n)` tapers of length `n`.
// Requires `tapers_col_major.size() >= n * k_eff` (column-major `n x k_eff`).
[[nodiscard]] SignalStatus dpss(usize n, f64 nw, MutArrayRef<f64> tapers_col_major,
                                MutArrayRef<f64> ratios, Allocator* alloc = nullptr);

// ============================================================================
// Linear Convolution
// ============================================================================

// Direct 1D linear convolution.
// Requires `out.size() >= a.size() + b.size() - 1` when both inputs are non-empty.
[[nodiscard]] SignalStatus conv_1d(ArrayRef<f64> a, ArrayRef<f64> b, MutArrayRef<f64> out) noexcept;

// Direct 2D linear convolution (column-major `m x n` and `p x q`).
// Requires `a.size() >= m * n`, `b.size() >= p * q`, and
// `out.size() >= (m + p - 1) * (n + q - 1)` when all dimensions are non-zero.
[[nodiscard]] SignalStatus conv_2d(ArrayRef<f64> a, usize m, usize n, ArrayRef<f64> b, usize p,
                                   usize q, MutArrayRef<f64> out) noexcept;

// ============================================================================
// IIR, Biquad & Second-Order Sections (SOS) Filtering
// ============================================================================

// Direct-Form-II-transposed IIR filter (real).
// Assumes `a` is already normalized (`a[0] == 1`).
// Writes `x.size()` samples into `y` (`y.size() >= x.size()`).
// Optional initial state `zi` (empty = zero state) and optional final state `zf`
// (writes `min(zf.size(), max(b.size(), a.size()) - 1)` elements when non-empty).
[[nodiscard]] SignalStatus filter_df2t(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> x,
                                       MutArrayRef<f64> y, ArrayRef<f64> zi = {},
                                       MutArrayRef<f64> zf = {}, Allocator* alloc = nullptr);

// Direct-Form-II-transposed IIR filter (complex).
[[nodiscard]] SignalStatus filter_df2t_c64(ArrayRef<c64> b, ArrayRef<c64> a, ArrayRef<c64> x,
                                           MutArrayRef<c64> y, ArrayRef<c64> zi = {},
                                           MutArrayRef<c64> zf = {}, Allocator* alloc = nullptr);

// Single biquad section DF2T recurrence with zero initial state (`y.size() >= x.size()`).
[[nodiscard]] SignalStatus biquad_df2t(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2, ArrayRef<f64> x,
                                       MutArrayRef<f64> y) noexcept;

// Single biquad section DF2T recurrence with explicit initial state (`y.size() >= x.size()`).
[[nodiscard]] SignalStatus biquad_df2t_state(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2,
                                             ArrayRef<f64> x, MutArrayRef<f64> y, f64 s1_init,
                                             f64 s2_init,
                                             BiquadState* final_state = nullptr) noexcept;

// Steady-state unit-input initial conditions for one normalized biquad section.
[[nodiscard]] BiquadState biquad_zi(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2) noexcept;

// Apply an `L x 6` column-major SOS cascade (`sos.size() >= 6 * num_sections`) to `x`,
// writing `x.size()` samples to `y` (`y.size() >= x.size()`).
// Optional `zi` has length `>= 2 * num_sections` (interleaved `(s1, s2)` per section) or empty.
// Returns kZeroLeadingCoeff if any section has `a0 == 0`.
[[nodiscard]] SignalStatus sosfilt_col_major(ArrayRef<f64> sos, usize num_sections, ArrayRef<f64> x,
                                             MutArrayRef<f64> y, ArrayRef<f64> zi = {},
                                             Allocator* alloc = nullptr);

// Zero-phase forward-backward SOS cascade on `x`, writing `x.size()` samples to `y`
// using odd-reflection edge padding (`edge = min(6 * L, n - 1)`) and
// DC-gain-scaled steady-state initial conditions.
[[nodiscard]] SignalStatus sosfiltfilt_col_major(ArrayRef<f64> sos, usize num_sections,
                                                 ArrayRef<f64> x, MutArrayRef<f64> y,
                                                 Allocator* alloc = nullptr);

// ============================================================================
// Transforms: Goertzel, Unwrap, DCT/IDCT, FWHT/IFWHT, Hilbert & Envelope
// ============================================================================

// Single-bin Goertzel DFT at 0-based (possibly fractional) bin index `k`.
// Computes sum_{m=0}^{n-1} x[m] * exp(-2*pi*i * k * m / n).
[[nodiscard]] c64 goertzel_bin(ArrayRef<f64> x, f64 k_zero_based) noexcept;
[[nodiscard]] c64 goertzel_bin_c64(ArrayRef<c64> x, f64 k_zero_based) noexcept;

// Phase unwrapping: removes jumps exceeding `period / 2` (default `2*pi`).
// Safe for in-place (`out.data() == phase.data()`). Requires `out.size() >= phase.size()`.
[[nodiscard]] SignalStatus unwrap_phase(ArrayRef<f64> phase, MutArrayRef<f64> out,
                                        f64 period = kTau) noexcept;

// Orthonormal 1-D Discrete Cosine Transform (Types 1, 2, 3, 4; default Type 2).
// Types 2 & 3 use O(N log N) FFT acceleration for N >= 16.
[[nodiscard]] SignalStatus dct(ArrayRef<f64> x, MutArrayRef<f64> out, i32 type = 2,
                               Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus idct(ArrayRef<f64> x, MutArrayRef<f64> out, i32 type = 2,
                                Allocator* alloc = nullptr);

// Fast Walsh-Hadamard Transform (`out.size()` must be a power of 2; `x` is zero-padded/truncated).
// Forward divides by `out.size()`; inverse does not scale.
[[nodiscard]] SignalStatus fwht(ArrayRef<f64> x, MutArrayRef<f64> out,
                                FwhtOrdering ordering = FwhtOrdering::kSequency,
                                Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus ifwht(ArrayRef<f64> x, MutArrayRef<f64> out,
                                 FwhtOrdering ordering = FwhtOrdering::kSequency,
                                 Allocator* alloc = nullptr);

// Exact length-N analytic signal via FFT (supports arbitrary N, power-of-2 and non-power-of-2).
[[nodiscard]] SignalStatus hilbert(ArrayRef<f64> x, MutArrayRef<c64> out,
                                   Allocator* alloc = nullptr);

// Signal envelopes (upper = mean + amp, lower = mean - amp; `lower` may be empty if unused).
[[nodiscard]] SignalStatus envelope_analytic(ArrayRef<f64> x, MutArrayRef<f64> upper,
                                             MutArrayRef<f64> lower = {},
                                             Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus envelope_fir(ArrayRef<f64> x, usize n_taps, MutArrayRef<f64> upper,
                                        MutArrayRef<f64> lower = {}, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus envelope_rms(ArrayRef<f64> x, usize window_len, MutArrayRef<f64> upper,
                                        MutArrayRef<f64> lower = {}) noexcept;

// ============================================================================
// Smoothing, Gap-Filling & Peak Detection
// ============================================================================

// Order-k 1-D sliding median filter (even k leans left; `zeropad` folds out-of-range zeros).
[[nodiscard]] SignalStatus medfilt1(ArrayRef<f64> x, usize k, bool zeropad, MutArrayRef<f64> out,
                                    Allocator* alloc = nullptr);

// Forward-backward Burg AR gap interpolation across NaN runs (`maxlen == 0` unlimited, `order == 0` AIC).
[[nodiscard]] SignalStatus fillgaps(ArrayRef<f64> x, usize maxlen, usize order,
                                    MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus fillgaps_c64(ArrayRef<c64> x, usize maxlen, usize order,
                                        MutArrayRef<c64> out, Allocator* alloc = nullptr);

// Savitzky-Golay projection matrix B (`framelen x framelen`, row-major) and optional
// differentiation matrix G (`framelen x (order + 1)`, row-major; unweighted).
// `weights` may be empty (unweighted) or length `framelen` positive weights.
[[nodiscard]] SignalStatus sgolay(i32 order, i32 framelen, MutArrayRef<f64> b_rowmajor,
                                  MutArrayRef<f64> g_rowmajor = {}, ArrayRef<f64> weights = {},
                                  Allocator* alloc = nullptr);

// 1-D Savitzky-Golay filter (`x.size() >= framelen`; handles edges via asymmetric rows of B).
[[nodiscard]] SignalStatus sgolayfilt(ArrayRef<f64> x, i32 order, i32 framelen,
                                      MutArrayRef<f64> out, ArrayRef<f64> weights = {},
                                      Allocator* alloc = nullptr);

struct PeakOptions {
    f64 min_height = -Limits<f64>::infinity();
    f64 threshold = 0.0;
    bool use_min_prom = false;
    f64 min_prom = 0.0;
    bool use_width_filter = false;
    f64 min_width = 0.0;
    f64 max_width = Limits<f64>::infinity();
    bool use_min_dist = false;
    f64 min_dist = 0.0;
    i32 max_peaks = -1; // < 0 means unlimited
    PeakSort sort_mode = PeakSort::kNone;
    PeakWidthRef width_ref = PeakWidthRef::kHalfProm;
};

struct PeakResult {
    Array<f64> values;
    Array<f64> locations;
    Array<f64> widths;
    Array<f64> prominences;
};

// Strict local-maxima finder with prominence, width, distance, sorting, and NPeaks filtering.
// `locations` may be empty (defaults to 1-based sample indices `1..x.size()`).
[[nodiscard]] SignalStatus findpeaks(ArrayRef<f64> x, ArrayRef<f64> locations,
                                     const PeakOptions& opts, PeakResult& out,
                                     Allocator* alloc = nullptr);

// ============================================================================
// Spectral Estimation & Time-Frequency Analysis
// ============================================================================

// Periodogram PSD estimate. If `window` is empty, uses rectangular window.
// Real: one-sided (`nfft / 2 + 1` bins); complex: two-sided (`nfft` bins).
[[nodiscard]] SignalStatus periodogram(ArrayRef<f64> x, ArrayRef<f64> window, usize nfft, f64 fs,
                                       MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out,
                                       Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus periodogram_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize nfft,
                                           f64 fs, MutArrayRef<f64> pxx_out,
                                           MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);

// Welch overlapped-segment PSD estimate (requires non-empty `window`, `noverlap < window.size()`).
[[nodiscard]] SignalStatus pwelch(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                                  f64 fs, MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out,
                                  Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pwelch_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize noverlap,
                                      usize nfft, f64 fs, MutArrayRef<f64> pxx_out,
                                      MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);

// Cross-spectral Welch estimators (real inputs of equal length; one-sided `nfft / 2 + 1` bins).
[[nodiscard]] SignalStatus cpsd(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window,
                                usize noverlap, usize nfft, f64 fs, MutArrayRef<c64> pxy_out,
                                MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus mscohere(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window,
                                    usize noverlap, usize nfft, f64 fs, MutArrayRef<f64> cxy_out,
                                    MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus tfestimate(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> window,
                                      usize noverlap, usize nfft, f64 fs, MutArrayRef<c64> txy_out,
                                      MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);

// Parametric AR PSD estimators: Yule-Walker (`pyulear`) and Burg (`pburg`).
// One-sided normalized angular frequency grid `w in [0, pi]` (`nfft / 2 + 1` bins).
[[nodiscard]] SignalStatus pyulear(ArrayRef<f64> x, usize order, usize nfft,
                                   MutArrayRef<f64> pxx_out, MutArrayRef<f64> freq_out,
                                   Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pburg(ArrayRef<f64> x, usize order, usize nfft, MutArrayRef<f64> pxx_out,
                                 MutArrayRef<f64> freq_out, Allocator* alloc = nullptr);

// One-sided STFT spectrogram (`s_colmajor` is `(nfft/2+1) x n_segments`, column-major).
// `freq_out` has length `nfft/2+1` in `[0, pi]`; `time_out` has length `n_segments` (sample centers).
[[nodiscard]] SignalStatus spectrogram(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap,
                                       usize nfft, MutArrayRef<c64> s_colmajor,
                                       MutArrayRef<f64> freq_out, MutArrayRef<f64> time_out,
                                       Allocator* alloc = nullptr);

// Short-Time Fourier Transform (`s_colmajor` is `out_rows x n_frames`, column-major).
[[nodiscard]] SignalStatus stft(ArrayRef<f64> x, ArrayRef<f64> window, usize noverlap, usize nfft,
                                StftRange range, MutArrayRef<c64> s_colmajor,
                                Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus stft_c64(ArrayRef<c64> x, ArrayRef<f64> window, usize noverlap,
                                    usize nfft, StftRange range, MutArrayRef<c64> s_colmajor,
                                    Allocator* alloc = nullptr);

// Inverse STFT via weighted overlap-add (`x_out` length `(n_frames - 1) * (M - noverlap) + M`).
[[nodiscard]] SignalStatus istft(ArrayRef<c64> s_colmajor, usize in_rows, usize n_frames,
                                 ArrayRef<f64> window, usize noverlap, usize nfft, StftRange range,
                                 MutArrayRef<f64> x_out, Allocator* alloc = nullptr);

struct ColaResult {
    bool is_cola = false;
    f64 median = 0.0;
    f64 max_dev = 0.0;
};

// Constant Overlap-Add compliance check (`wola = true` checks sum(w^2), `false` checks sum(w)).
[[nodiscard]] SignalStatus iscola(ArrayRef<f64> window, usize noverlap, bool wola, ColaResult& out,
                                  Allocator* alloc = nullptr);

// ============================================================================
// Analog Prototypes, Frequency Transforms & IIR Filter Design
// ============================================================================

enum class FilterType : u8 {
    kLowpass = 0,
    kHighpass,
    kBandpass,
    kBandstop,
};

enum class KaiserBandType : u8 {
    kLow = 0,
    kHigh,
    kStop,
    kDc0,
    kDc1,
};

struct ZpkResult {
    Array<c64> z;
    Array<c64> p;
    f64 k = 1.0;
};

struct TfResult {
    Array<f64> b;
    Array<f64> a;
};

struct OrderResult {
    i32 order = 0;
    f64 wn[2] = {0.0, 0.0};
    usize wn_count = 0;
};

struct FirpmOrdResult {
    i32 order = 0;
    Array<f64> f;
    Array<f64> a;
    Array<f64> w;
};

struct KaiserOrdResult {
    i32 order = 0;
    Array<f64> wn;
    f64 beta = 0.0;
    KaiserBandType ftype = KaiserBandType::kLow;
};

// Analog lowpass prototypes (cutoff Omega = 1 rad/s).
[[nodiscard]] SignalStatus buttap(i32 n, ZpkResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus cheb1ap(i32 n, f64 rp_db, ZpkResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus cheb2ap(i32 n, f64 rs_db, ZpkResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus besselap(i32 n, ZpkResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus ellipap(i32 n, f64 rp_db, f64 rs_db, ZpkResult& out,
                                   Allocator* alloc = nullptr);

// Analog lowpass -> LP / HP / BP / BS frequency transformations.
[[nodiscard]] SignalStatus lp2lp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, ZpkResult& out,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus lp2hp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, ZpkResult& out,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus lp2bp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, f64 bw,
                                 ZpkResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus lp2bs(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, f64 bw,
                                 ZpkResult& out, Allocator* alloc = nullptr);

// Bilinear (Tustin) transform (`fp == 0` disables prewarping) & impulse invariance.
[[nodiscard]] SignalStatus bilinear(ArrayRef<f64> b, ArrayRef<f64> a, f64 fs, f64 fp,
                                    TfResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus impinvar(ArrayRef<f64> b, ArrayRef<f64> a, f64 fs, TfResult& out,
                                    Allocator* alloc = nullptr);

// Analog frequency response H(j*w) = b(j*w) / a(j*w).
[[nodiscard]] SignalStatus freqs(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> w,
                                 MutArrayRef<c64> h_out) noexcept;

// Top-level IIR filter designs (`wn` has length 1 for LP/HP, length 2 for BP/BS).
[[nodiscard]] SignalStatus butter(i32 n, ArrayRef<f64> wn, FilterType ftype, bool analog,
                                  TfResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus cheby1(i32 n, f64 rp_db, ArrayRef<f64> wn, FilterType ftype, bool analog,
                                  TfResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus cheby2(i32 n, f64 rs_db, ArrayRef<f64> wn, FilterType ftype, bool analog,
                                  TfResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus ellip(i32 n, f64 rp_db, f64 rs_db, ArrayRef<f64> wn, FilterType ftype,
                                 bool analog, TfResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus besself(i32 n, ArrayRef<f64> wn, FilterType ftype, bool analog,
                                   TfResult& out, Allocator* alloc = nullptr);

// IIR & FIR order estimators.
[[nodiscard]] SignalStatus buttord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db,
                                   bool analog, OrderResult& out) noexcept;
[[nodiscard]] SignalStatus cheb1ord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db,
                                    bool analog, OrderResult& out) noexcept;
[[nodiscard]] SignalStatus cheb2ord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db,
                                    bool analog, OrderResult& out) noexcept;
[[nodiscard]] SignalStatus ellipord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db,
                                    bool analog, OrderResult& out) noexcept;
[[nodiscard]] SignalStatus firpmord(ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> dev, f64 fs,
                                    FirpmOrdResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus kaiserord(ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> dev, f64 fs,
                                     KaiserOrdResult& out, Allocator* alloc = nullptr);

// ============================================================================
// Multirate & Fast Overlap-Add Filtering
// ============================================================================

[[nodiscard]] constexpr usize upfirdn_length(usize lx, usize lh, usize p, usize q) noexcept {
    if (lx == 0 || lh == 0 || p == 0 || q == 0) return 0;
    const usize conv_len = (lx - 1) * p + lh;
    return (conv_len + q - 1) / q;
}

// Polyphase upsample-by-p, FIR-convolve with `h`, downsample-by-q.
[[nodiscard]] SignalStatus upfirdn(ArrayRef<f64> x, ArrayRef<f64> h, usize p, usize q,
                                   MutArrayRef<f64> out, Allocator* alloc = nullptr);

// Overlap-add FFT FIR filtering (`nfft == 0` selects automatic power-of-2 block size).
[[nodiscard]] SignalStatus fftfilt(ArrayRef<f64> b, ArrayRef<f64> x, usize nfft,
                                   MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus fftfilt_c64(ArrayRef<c64> b, ArrayRef<c64> x, usize nfft,
                                       MutArrayRef<c64> out, Allocator* alloc = nullptr);

// ============================================================================
// FIR Filter Design (fir1 / fir2 / firls / firpm Parks-McClellan)
// ============================================================================

enum class FirFtype : u8 {
    kStandard = 0,
    kHilbert,
    kDifferentiator,
};

// Windowed-sinc single-cutoff lowpass/highpass FIR (Hamming window, length `order + 1`).
[[nodiscard]] SignalStatus fir1_single(i32 order, f64 wn, bool highpass,
                                       MutArrayRef<f64> out_b) noexcept;

// General multi-band windowed-sinc FIR (`window` empty => Hamming, odd-order Nyquist bump).
[[nodiscard]] SignalStatus fir1_multiband(i32 order, ArrayRef<f64> wn, bool first_band_pass,
                                          ArrayRef<f64> window, bool scale, Array<f64>& out_b,
                                          Allocator* alloc = nullptr);

// Frequency-sampling FIR filter design (`npt <= 0` default 512, `lap <= 0` default npt/25).
[[nodiscard]] SignalStatus fir2(i32 order, ArrayRef<f64> f, ArrayRef<f64> m, i32 npt, i32 lap,
                                ArrayRef<f64> window, Array<f64>& out_b,
                                Allocator* alloc = nullptr);

// Least-squares linear-phase FIR design (Types I-IV, `w` empty => unit weights).
[[nodiscard]] SignalStatus firls(i32 order, ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> w,
                                 FirFtype ftype, MutArrayRef<f64> out_b,
                                 Allocator* alloc = nullptr);

// Parks-McClellan equiripple FIR design via Remez exchange (Types I-IV).
// Writes length `order + 1` impulse response into `out_b` and peak ripple magnitude to `out_err`.
[[nodiscard]] SignalStatus firpm(i32 order, ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> w,
                                 FirFtype ftype, MutArrayRef<f64> out_b, f64& out_err,
                                 Allocator* alloc = nullptr);

// Zero-phase forward-backward IIR filter (`y.size() >= x.size()`) with odd-reflection
// edge padding (`min(3 * (max(b.size(), a.size()) - 1), x.size() - 1)`) and steady-state
// initial conditions. Returns kZeroLeadingCoeff if `a.empty()` or `a[0] == 0`.
[[nodiscard]] SignalStatus filtfilt(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> x,
                                    MutArrayRef<f64> y, Allocator* alloc = nullptr);

// Digital filter frequency response H(e^{j*w}) = B(e^{-j*w}) / A(e^{-j*w}).
// Evaluated at explicit radian frequencies `w` (`h_out.size() >= w.size()`).
[[nodiscard]] SignalStatus freqz(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> w,
                                 MutArrayRef<c64> h_out) noexcept;
[[nodiscard]] SignalStatus freqz_c64(ArrayRef<c64> b, ArrayRef<c64> a, ArrayRef<f64> w,
                                     MutArrayRef<c64> h_out) noexcept;

// Digital filter frequency response on `npts` equispaced grid points over [0, pi)
// (or [0, 2*pi) when `whole == true`). When `fs > 0`, `freq_out` is in Hz.
[[nodiscard]] SignalStatus freqz_grid(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, bool whole,
                                      f64 fs, MutArrayRef<c64> h_out,
                                      MutArrayRef<f64> freq_out) noexcept;
[[nodiscard]] SignalStatus freqz_grid_c64(ArrayRef<c64> b, ArrayRef<c64> a, usize npts, bool whole,
                                          f64 fs, MutArrayRef<c64> h_out,
                                          MutArrayRef<f64> freq_out) noexcept;

// Unwrapped phase response on `npts` points over [0, pi) (Hz if `fs > 0`).
[[nodiscard]] SignalStatus phasez(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, f64 fs,
                                  MutArrayRef<f64> phi_out, MutArrayRef<f64> freq_out) noexcept;
[[nodiscard]] SignalStatus phasez_c64(ArrayRef<c64> b, ArrayRef<c64> a, usize npts, f64 fs,
                                      MutArrayRef<f64> phi_out, MutArrayRef<f64> freq_out) noexcept;

// Exact group delay (in samples) via combined polynomial c = conv(b, reverse(a)).
[[nodiscard]] SignalStatus grpdelay(ArrayRef<f64> b, ArrayRef<f64> a, usize npts, f64 fs,
                                    MutArrayRef<f64> gd_out, MutArrayRef<f64> freq_out,
                                    Allocator* alloc = nullptr);

// Impulse response decay length (5e-5 threshold, capped at 8192 for unstable/marginally stable IIR).
[[nodiscard]] SignalStatus impzlength(ArrayRef<f64> b, ArrayRef<f64> a, usize& out_len,
                                      Allocator* alloc = nullptr);

// Impulse and unit-step responses (`n` samples; writes `h_out` / `s_out` and optional `t_out`).
[[nodiscard]] SignalStatus impz(ArrayRef<f64> b, ArrayRef<f64> a, usize n, MutArrayRef<f64> h_out,
                                MutArrayRef<f64> t_out = {}, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus stepz(ArrayRef<f64> b, ArrayRef<f64> a, usize n, MutArrayRef<f64> s_out,
                                 MutArrayRef<f64> t_out = {}, Allocator* alloc = nullptr);

// Phase delay -phi(w)/w (NaN at w == 0) and zero-phase response Hr(w).
[[nodiscard]] SignalStatus phasedelay(ArrayRef<f64> b, ArrayRef<f64> a, usize npts,
                                      MutArrayRef<f64> pd_out, MutArrayRef<f64> w_out) noexcept;
[[nodiscard]] SignalStatus zerophase(ArrayRef<f64> b, ArrayRef<f64> a, usize npts,
                                     MutArrayRef<f64> hr_out, MutArrayRef<f64> w_out) noexcept;

// Polynomial root radial scaling: b[i, k] = p[i, k] * factor(k), where `p` is `rows x cols`
// column-major and `scale` has length 1 (factor(k) = scale[0]^k) or `cols` (scale[k]^k).
[[nodiscard]] SignalStatus polyscale(ArrayRef<f64> p_col_major, usize rows, usize cols,
                                     ArrayRef<f64> scale, MutArrayRef<f64> out_col_major,
                                     Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus polyscale_c64(ArrayRef<c64> p_col_major, usize rows, usize cols,
                                         ArrayRef<c64> scale, MutArrayRef<c64> out_col_major,
                                         Allocator* alloc = nullptr);

// Minimum-phase polynomial stabilization: reflects roots with |r| > 1 to 1 / conj(r).
[[nodiscard]] SignalStatus polystab(ArrayRef<f64> a, Array<f64>& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus polystab_c64(ArrayRef<c64> a, Array<c64>& out,
                                        Allocator* alloc = nullptr);

// ============================================================================
// Filter Representation Conversions (ZPK / TF / SOS / State-Space)
// ============================================================================

struct SosResult {
    Array<f64> sos_col_major; // L x 6 column-major
    usize num_sections = 0;
    f64 gain = 1.0;
};

struct StateSpaceResult {
    Array<f64> a_col_major; // N x N column-major
    Array<f64> b;           // N x 1
    Array<f64> c;           // 1 x N
    f64 d = 0.0;
    usize order = 0;
};

// Convert zeros/poles/gain to L x 6 column-major SOS cascade.
// When `surplus_at_origin == true` (zp2sos), sections with no finite zeros receive [0 0 1];
// when `false` (tf2sos), surplus zeros stay left-aligned at infinity [1 0 0].
// Gain is returned in `out.gain` (unit leading section gain; caller scales row 0 if desired).
[[nodiscard]] SignalStatus zp2sos(ArrayRef<c64> zeros, ArrayRef<c64> poles, f64 gain,
                                  bool surplus_at_origin, SosResult& out,
                                  Allocator* alloc = nullptr);

// Convert transfer function b/a to L x 6 column-major SOS cascade + gain.
[[nodiscard]] SignalStatus tf2sos(ArrayRef<c64> b, ArrayRef<c64> a, SosResult& out,
                                  Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus tf2sos(ArrayRef<f64> b, ArrayRef<f64> a, SosResult& out,
                                  Allocator* alloc = nullptr);

// Convert L x 6 column-major SOS cascade + gain to transfer function (b, a).
[[nodiscard]] SignalStatus sos2tf(ArrayRef<f64> sos_col_major, usize num_sections, f64 gain,
                                  TfResult& out, Allocator* alloc = nullptr);

// Convert L x 6 column-major SOS cascade + gain to zero-pole-gain (z, p, k).
[[nodiscard]] SignalStatus sos2zp(ArrayRef<f64> sos_col_major, usize num_sections, f64 gain,
                                  ZpkResult& out, Allocator* alloc = nullptr);

// Controllable canonical state-space realization of proper SISO transfer function b/a (na >= 2).
[[nodiscard]] SignalStatus tf2ss(ArrayRef<f64> b, ArrayRef<f64> a, StateSpaceResult& out,
                                 Allocator* alloc = nullptr);

// SISO state-space to transfer function via Faddeev-LeVerrier (`a_col_major` is N x N).
[[nodiscard]] SignalStatus ss2tf(ArrayRef<f64> a_col_major, usize order, ArrayRef<f64> b,
                                 ArrayRef<f64> c, f64 d, TfResult& out,
                                 Allocator* alloc = nullptr);

// ============================================================================
// Convolution, Deconvolution & Correlation Extensions
// ============================================================================

enum class XcorrScale : u8 {
    kNone = 0,
    kBiased,
    kUnbiased,
    kCoeff,
};

// Automatic direct/FFT 1-D linear convolution (`out.size() >= a.size() + b.size() - 1`).
[[nodiscard]] SignalStatus conv_1d_auto(ArrayRef<f64> a, ArrayRef<f64> b, MutArrayRef<f64> out,
                                        Allocator* alloc = nullptr);

// Modulo-n circular convolution (`n == 0` uses linear length `x.size() + y.size() - 1`).
[[nodiscard]] SignalStatus cconv(ArrayRef<f64> x, ArrayRef<f64> y, usize n, MutArrayRef<f64> out,
                                 Allocator* alloc = nullptr);

// Polynomial deconvolution b = conv(a, q) + r.
// If `a.size() > b.size()`, writes q = [0] and r = b.
[[nodiscard]] SignalStatus deconv(ArrayRef<f64> b, ArrayRef<f64> a, Array<f64>& q_out,
                                  Array<f64>& r_out, Allocator* alloc = nullptr);

// Full cross-correlation (`len = 2 * max(nx, ny) - 1`, lags `-(max-1)..+(max-1)`).
// `lags_out` may be empty if lags are not needed.
[[nodiscard]] SignalStatus xcorr(ArrayRef<f64> x, ArrayRef<f64> y, MutArrayRef<f64> r_out,
                                 MutArrayRef<f64> lags_out = {}, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus xcorr_c64(ArrayRef<c64> x, ArrayRef<c64> y, MutArrayRef<c64> r_out,
                                     MutArrayRef<f64> lags_out = {}, Allocator* alloc = nullptr);

// Cross-covariance (mean-removed cross-correlation with optional `maxlag` and scaling).
// `maxlag < 0` uses full lag `max(nx, ny) - 1`.
[[nodiscard]] SignalStatus xcov(ArrayRef<f64> x, ArrayRef<f64> y, i32 maxlag, XcorrScale scale,
                                Array<f64>& c_out, Array<f64>& lags_out,
                                Allocator* alloc = nullptr);

// 2-D cross-correlation (column-major `ra x ca` and `rb x cb` -> `(ra+rb-1) x (ca+cb-1)`).
[[nodiscard]] SignalStatus xcorr2(ArrayRef<f64> a, usize ra, usize ca, ArrayRef<f64> b, usize rb,
                                  usize cb, MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus xcorr2_c64(ArrayRef<c64> a, usize ra, usize ca, ArrayRef<c64> b,
                                      usize rb, usize cb, MutArrayRef<c64> out) noexcept;

// Estimate integer sample delay between `x` and `y` (`max_lag <= 0` searches all lags).
[[nodiscard]] SignalStatus finddelay(ArrayRef<f64> x, ArrayRef<f64> y, i64 max_lag, i64& out_delay,
                                     Allocator* alloc = nullptr);

// ============================================================================
// Additional Transforms: DFT Matrix, DST, Cepstrum & Chirp Z-Transform
// ============================================================================

// N x N column-major DFT matrix W(k, n) = exp(-2*pi*i * k * n / N).
[[nodiscard]] SignalStatus dftmtx(usize n, MutArrayRef<c64> out_col_major) noexcept;

// Type-II Discrete Sine Transform and its inverse (`out.size() >= x.size()`).
[[nodiscard]] SignalStatus dst(ArrayRef<f64> x, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus idst(ArrayRef<f64> y, MutArrayRef<f64> out) noexcept;

// Real cepstrum and minimum-phase reconstruction (exact length N via Bluestein/radix-2 FFT).
[[nodiscard]] SignalStatus rceps(ArrayRef<f64> x, MutArrayRef<f64> out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus rceps_minphase(ArrayRef<f64> x, MutArrayRef<f64> cep_out,
                                          MutArrayRef<f64> minphase_out,
                                          Allocator* alloc = nullptr);

// Complex cepstrum (optional `nd_out` receives circular delay removed) and inverse complex cepstrum.
[[nodiscard]] SignalStatus cceps(ArrayRef<f64> x, MutArrayRef<f64> out, f64* nd_out = nullptr,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus icceps(ArrayRef<f64> c, MutArrayRef<f64> out,
                                  Allocator* alloc = nullptr);

// Chirp Z-Transform via Bluestein convolution: Y[k] = sum_{n=0}^{N-1} x[n] * a^{-n} * w^{n*k}.
// `czt_batch` processes `n_cols` contiguous column-major signals of length `n_rows`, sharing
// the precomputed chirp FFT across all columns (`out_col_major` is `m x n_cols`).
[[nodiscard]] SignalStatus czt(ArrayRef<c64> x, usize m, c64 w, c64 a, MutArrayRef<c64> out,
                               Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus czt_batch(ArrayRef<c64> x_col_major, usize n_rows, usize n_cols, usize m,
                                     c64 w, c64 a, MutArrayRef<c64> out_col_major,
                                     Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus czt_batch_real(ArrayRef<f64> x_col_major, usize n_rows, usize n_cols,
                                          usize m, c64 w, c64 a, MutArrayRef<c64> out_col_major,
                                          Allocator* alloc = nullptr);

// ============================================================================
// Multirate Extensions (decimate & resample)
// ============================================================================

// Anti-aliased integer decimation (order-8 Chebyshev I, Rp=0.05 dB, zero-phase filtfilt).
[[nodiscard]] SignalStatus decimate(ArrayRef<f64> x, usize factor, Array<f64>& out,
                                    Allocator* alloc = nullptr);

// Rational resampling by p/q using Kaiser-windowed ideal sinc anti-alias filter + polyphase upfirdn.
// `n_order <= 0` defaults to 10. Optional `b_out` receives the anti-alias FIR taps when non-null.
[[nodiscard]] SignalStatus resample(ArrayRef<f64> x, usize p, usize q, i32 n_order, f64 beta,
                                    Array<f64>& y_out, Array<f64>* b_out = nullptr,
                                    Allocator* alloc = nullptr);

// ============================================================================
// Parametric Signal Modeling & Subspace Pseudospectra
// ============================================================================

struct LevinsonResult {
    Array<f64> a;
    f64 e = 0.0;
    Array<f64> k;
};

struct RlevinsonResult {
    Array<f64> r;
    Array<f64> k;
};

// Levinson-Durbin recursion (`order < 0` uses `r.size() - 1`).
[[nodiscard]] SignalStatus levinson(ArrayRef<f64> r, i32 order, LevinsonResult& out,
                                    Allocator* alloc = nullptr);

// Step-down reverse Levinson: recover autocorrelation R and reflection coeffs k from (a, efinal).
[[nodiscard]] SignalStatus rlevinson(ArrayRef<f64> a, f64 efinal, RlevinsonResult& out,
                                     Allocator* alloc = nullptr);

// AR model estimation: Yule-Walker, Burg, Covariance, Modified Covariance.
[[nodiscard]] SignalStatus aryule(ArrayRef<f64> x, i32 order, LevinsonResult& out,
                                  Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus arburg(ArrayRef<f64> x, i32 order, LevinsonResult& out,
                                  Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus arcov(ArrayRef<f64> x, i32 order, Array<f64>& a_out, f64& e_out,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus armcov(ArrayRef<f64> x, i32 order, Array<f64>& a_out, f64& e_out,
                                  Allocator* alloc = nullptr);

// Autocorrelation / reflection / polynomial / inverse-sine / log-area-ratio conversions.
[[nodiscard]] SignalStatus schurrc(ArrayRef<f64> r, Array<f64>& k_out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus rc2ac(ArrayRef<f64> k, f64 r0, Array<f64>& r_out,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus poly2rc(ArrayRef<f64> a, f64 efinal, Array<f64>& k_out, f64& r0_out,
                                   Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus rc2poly(ArrayRef<f64> k, f64 r0, Array<f64>& a_out, f64& efinal_out,
                                   Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus is2rc(ArrayRef<f64> is_in, MutArrayRef<f64> k_out) noexcept;
[[nodiscard]] SignalStatus rc2is(ArrayRef<f64> k, MutArrayRef<f64> is_out) noexcept;
[[nodiscard]] SignalStatus lar2rc(ArrayRef<f64> g, MutArrayRef<f64> k_out) noexcept;
[[nodiscard]] SignalStatus rc2lar(ArrayRef<f64> k, MutArrayRef<f64> g_out) noexcept;

// Prony and Steiglitz-McBride IIR time-domain modeling.
// In `stmcb`, empty `x_opt` models unit-impulse response; empty `ai_opt` initializes via Prony.
[[nodiscard]] SignalStatus prony(ArrayRef<f64> h, i32 nb, i32 na, TfResult& out,
                                 Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus stmcb(ArrayRef<f64> y, ArrayRef<f64> x_opt, i32 nb, i32 na, i32 niter,
                                 ArrayRef<f64> ai_opt, TfResult& out, Allocator* alloc = nullptr);

// Autocorrelation data matrix X ((N + m) x (m + 1), column-major) with X' * X = R.
[[nodiscard]] SignalStatus corrmtx(ArrayRef<f64> x, i32 m, MutArrayRef<f64> out_col_major) noexcept;

// Line Spectral Frequencies <-> AR polynomial conversions.
[[nodiscard]] SignalStatus poly2lsf(ArrayRef<f64> a, Array<f64>& lsf_out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus lsf2poly(ArrayRef<f64> lsf, Array<f64>& a_out,
                                    Allocator* alloc = nullptr);

// Frequency-domain Levi equation-error LSQ rational fitting (s-domain & z-domain).
[[nodiscard]] SignalStatus invfreqs(ArrayRef<c64> h, ArrayRef<f64> w, i32 nb, i32 na, TfResult& out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus invfreqz(ArrayRef<c64> h, ArrayRef<f64> w, i32 nb, i32 na, TfResult& out,
                                    Allocator* alloc = nullptr);

// MUSIC & Eigenvector subspace pseudospectra (`p_out` & `f_out` length `nfft / 2 + 1`).
[[nodiscard]] SignalStatus pmusic(ArrayRef<f64> x, i32 p, i32 nfft, f64 fs, MutArrayRef<f64> p_out,
                                  MutArrayRef<f64> f_out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus peig(ArrayRef<f64> x, i32 p, i32 nfft, f64 fs, MutArrayRef<f64> p_out,
                                MutArrayRef<f64> f_out, Allocator* alloc = nullptr);

// ============================================================================
// Spectral Metrics, pspectrum, Instantaneous Frequency & Vibration Analysis
// ============================================================================

struct ObwResult {
    f64 bw = 0.0;
    f64 flo = 0.0;
    f64 fhi = 0.0;
    f64 power = 0.0;
};

enum class PspectrumMode : u8 {
    kPower = 0,
    kSpectrogram,
};

struct PspectrumOptions {
    PspectrumMode mode = PspectrumMode::kPower;
    f64 leakage = 0.5;
    f64 overlap_percent = -1.0;  // < 0 => auto from leakage
    f64 time_resolution = -1.0;  // <= 0 => auto
    f64 fmin = Limits<f64>::quiet_nan();
    f64 fmax = Limits<f64>::quiet_nan();
    bool two_sided = false;
    bool has_two_sided = false;
    bool is_norm_freq = true;
    f64 fs = kTau;
    f64 t0 = 0.0;
};

struct PspectrumResult {
    Array<f64> p_col_major; // n_freq x n_cols
    usize n_freq = 0;
    usize n_cols = 0;
    Array<f64> f;           // n_freq
    Array<f64> t;           // n_cols (empty in power mode)
};

struct RainflowCycle {
    f64 count = 0.0;
    f64 range = 0.0;
    f64 mean = 0.0;
    f64 start_idx = 0.0;
    f64 end_idx = 0.0;
};

// Bandpower from precomputed PSD (rectangular full-range or clipped trapezoidal over [f_lo, f_hi]).
[[nodiscard]] SignalStatus bandpower_psd(ArrayRef<f64> pxx, ArrayRef<f64> f, f64 f_lo, f64 f_hi,
                                         bool use_range, f64& out_power) noexcept;

// Time-series spectral measurements (`fs <= 0` => normalized radian frequency [0, pi]).
[[nodiscard]] SignalStatus bandpower_time(ArrayRef<f64> x, f64 fs, f64 f_lo, f64 f_hi,
                                          f64& out_power, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus meanfreq_time(ArrayRef<f64> x, f64 fs, f64& out_freq,
                                         Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus medfreq_time(ArrayRef<f64> x, f64 fs, f64& out_freq,
                                        Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus enbw(ArrayRef<f64> window, f64 fs, f64& out_bw) noexcept;
[[nodiscard]] SignalStatus obw_time(ArrayRef<f64> x, f64 fs, f64 pct, ObwResult& out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus powerbw_time(ArrayRef<f64> x, f64 fs, f64& out_bw,
                                        Allocator* alloc = nullptr);

// Spectral shape descriptors.
[[nodiscard]] SignalStatus spectral_crest(ArrayRef<f64> x, f64 fs, f64& out_val,
                                          Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus spectral_flatness(ArrayRef<f64> x, f64 fs, f64& out_val,
                                             Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus spectral_entropy(ArrayRef<f64> x, f64 fs, f64& out_val,
                                            Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus spectral_kurtosis(ArrayRef<f64> x, f64 fs, f64& out_val,
                                             Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus spectral_skewness(ArrayRef<f64> x, f64 fs, f64& out_val,
                                             Allocator* alloc = nullptr);

// Harmonic distortion / SNR family (Kaiser beta=38 periodogram).
[[nodiscard]] SignalStatus snr_time(ArrayRef<f64> x, f64 fs, f64& out_db,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus sinad_time(ArrayRef<f64> x, f64 fs, f64& out_db,
                                      Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus thd_time(ArrayRef<f64> x, f64 fs, f64& out_db,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus sfdr_time(ArrayRef<f64> x, f64 fs, f64& out_db,
                                     Allocator* alloc = nullptr);

// Zoom-CZT power spectrum & spectrogram (`x_col_major` is `n_samples x n_channels`).
[[nodiscard]] SignalStatus pspectrum_real(ArrayRef<f64> x_col_major, usize n_samples,
                                          usize n_channels, const PspectrumOptions& opts,
                                          PspectrumResult& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pspectrum_c64(ArrayRef<c64> x_col_major, usize n_samples,
                                         usize n_channels, const PspectrumOptions& opts,
                                         PspectrumResult& out, Allocator* alloc = nullptr);

// Instantaneous frequency (Hilbert & spectrogram first moment) and instantaneous bandwidth.
[[nodiscard]] SignalStatus instfreq_hilbert(ArrayRef<f64> x, f64 fs, f64 t0,
                                            MutArrayRef<f64> f_out, MutArrayRef<f64> t_out,
                                            Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus instfreq_from_spec(ArrayRef<f64> p_col_major, usize n_freq, usize n_time,
                                              ArrayRef<f64> f, MutArrayRef<f64> out_if) noexcept;
[[nodiscard]] SignalStatus instbw_from_spec(ArrayRef<f64> p_col_major, usize n_freq, usize n_time,
                                            ArrayRef<f64> f, MutArrayRef<f64> out_ibw) noexcept;

// Vibration analysis: envelope spectrum, tachometer RPM, ASTM E1049-85 rainflow, TSA.
[[nodiscard]] SignalStatus envspectrum(ArrayRef<f64> x, f64 fs, Array<f64>& es_out,
                                       Array<f64>& f_out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus tachorpm(ArrayRef<f64> x, f64 fs, bool has_threshold, f64 threshold,
                                    i32 ppr, Array<f64>& rpm_out, Array<f64>& time_out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus rainflow(ArrayRef<f64> x, Array<RainflowCycle>& cycles_out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus tsa(ArrayRef<f64> x, f64 fs, ArrayRef<f64> rpm, f64 fs_rpm,
                               i32 n_per_rev, Array<f64>& avg_out, Array<f64>& theta_out,
                               Allocator* alloc = nullptr);

// ============================================================================
// Waveform Generation & Analog Modulation
// ============================================================================

enum class PulstranKind : u8 {
    kRectpuls = 0,
    kTripuls,
    kGauspuls,
};

enum class ChirpMethod : u8 {
    kLinear = 0,
    kQuadratic,
    kLogarithmic,
};

enum class ChirpQuadShape : u8 {
    kDefault = 0,
    kConvex,
    kConcave,
};

enum class ModulationMethod : u8 {
    kAm = 0,
    kAmdsbTc,
    kAmssb,
    kFm,
    kPm,
};

[[nodiscard]] SignalStatus rectpuls(ArrayRef<f64> t, f64 w, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus tripuls(ArrayRef<f64> t, f64 w, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus gauspuls(ArrayRef<f64> t, f64 fc, f64 bw, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus pulstran_builtin(ArrayRef<f64> t, ArrayRef<f64> d, PulstranKind kind,
                                            f64 fc_or_w, f64 bw, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus square(ArrayRef<f64> t, f64 duty, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus sawtooth(ArrayRef<f64> t, f64 width, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus sinc(ArrayRef<f64> t, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus gmonopuls(ArrayRef<f64> t, f64 fc, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus diric(ArrayRef<f64> x, i32 n, MutArrayRef<f64> out) noexcept;
[[nodiscard]] SignalStatus chirp(ArrayRef<f64> t, f64 f0, f64 t1, f64 f1, ChirpMethod method,
                                 f64 phi_deg, ChirpQuadShape quad_shape,
                                 MutArrayRef<f64> out) noexcept;

// Multi-channel column-major (`len x cols`) analog modulation / demodulation / VCO.
[[nodiscard]] SignalStatus modulate(ArrayRef<f64> x_col_major, usize len, usize cols, f64 fc,
                                    f64 fs, ModulationMethod method, bool has_opt, f64 opt,
                                    MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus demod(ArrayRef<f64> y_col_major, usize len, usize cols, f64 fc, f64 fs,
                                 ModulationMethod method, bool has_opt, f64 opt,
                                 MutArrayRef<f64> out_col_major, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus vco(ArrayRef<f64> x_col_major, usize len, usize cols, f64 fc, f64 range1,
                               f64 fs, MutArrayRef<f64> out_col_major) noexcept;

// ============================================================================
// Pulse Transition Metrics & Signal Regions of Interest (ROI)
// ============================================================================

struct StateLevels {
    f64 low = 0.0;
    f64 high = 0.0;
};

struct RiseFallResult {
    Array<f64> dur;
    Array<f64> lt;
    Array<f64> ut;
    f64 ll = 0.0;
    f64 ul = 0.0;
};

struct SigRoi {
    i64 start = 0; // 1-based inclusive
    i64 end = 0;   // 1-based inclusive
};

[[nodiscard]] SignalStatus statelevels(ArrayRef<f64> x, StateLevels& out) noexcept;
[[nodiscard]] SignalStatus midcross(ArrayRef<f64> x, f64 fs, Array<f64>& out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus risetime_full(ArrayRef<f64> x, f64 fs, bool rising, RiseFallResult& out,
                                         Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus slewrate(ArrayRef<f64> x, f64 fs, Array<f64>& out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus overshoot(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus undershoot(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus settlingtime(ArrayRef<f64> x, f64 fs, f64 tol, Array<f64>& out,
                                        Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pulsewidth(ArrayRef<f64> x, f64 fs, Array<f64>& out,
                                      Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pulseperiod(ArrayRef<f64> x, f64 fs, Array<f64>& out,
                                       Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus pulsesep(ArrayRef<f64> x, f64 fs, Array<f64>& out,
                                    Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus dutycycle(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc = nullptr);

// Signal ROI utilities (1-based inclusive [start, end] pairs).
[[nodiscard]] SignalStatus binmask2sigroi(ArrayRef<u8> mask, Array<SigRoi>& out,
                                          Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus sigroi2binmask(ArrayRef<SigRoi> rois, i64 len_user, Array<u8>& out_mask,
                                          Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus extendsigroi(ArrayRef<SigRoi> rois, i64 l_pre, i64 l_post,
                                        Array<SigRoi>& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus shortensigroi(ArrayRef<SigRoi> rois, i64 l_pre, i64 l_post,
                                         Array<SigRoi>& out, Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus mergesigroi(ArrayRef<SigRoi> rois, i64 sep, Array<SigRoi>& out,
                                       Allocator* alloc = nullptr);
[[nodiscard]] SignalStatus removesigroi(ArrayRef<SigRoi> rois, i64 max_len, Array<SigRoi>& out,
                                        Allocator* alloc = nullptr);

} // namespace keel::signal
