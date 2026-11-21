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

} // namespace keel::signal
