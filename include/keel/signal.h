#pragma once

#include "keel/complex.h"
#include "keel/core.h"
#include "keel/math.h"
#include "keel/memory.h"

namespace keel::signal {

enum class SignalStatus : u8 {
    kOk = 0,
    kInvalidArgument,
    kZeroLeadingCoeff,
};

struct BiquadState {
    f64 s1 = 0.0;
    f64 s2 = 0.0;
};

// ============================================================================
// Window Functions
// ============================================================================
// All window functions write `n` samples into `out` (no-op when `n == 0`).

void rectwin(usize n, f64* out) noexcept;
void hann(usize n, f64* out) noexcept;
void hamming(usize n, f64* out) noexcept;
void blackman(usize n, f64* out) noexcept;
void bartlett(usize n, f64* out) noexcept;
void triang(usize n, f64* out) noexcept;
void kaiser(usize n, f64 beta, f64* out) noexcept;
void tukeywin(usize n, f64 r, f64* out) noexcept;
void flattopwin(usize n, f64* out) noexcept;
void parzenwin(usize n, f64* out) noexcept;
void nuttallwin(usize n, f64* out) noexcept;
void blackmanharris(usize n, f64* out) noexcept;
void bohmanwin(usize n, f64* out) noexcept;
void barthannwin(usize n, f64* out) noexcept;

// Gaussian window: returns kInvalidArgument if alpha <= 0.
[[nodiscard]] SignalStatus gausswin(usize n, f64 alpha, f64* out) noexcept;

// Dolph-Chebyshev window with `at_db` dB sidelobe attenuation (> 0).
[[nodiscard]] SignalStatus chebwin(usize n, f64 at_db, f64* out, Allocator* alloc = nullptr);

// Taylor window: requires nbar >= 2 and sll_db < 0 (peak sidelobe in dB).
[[nodiscard]] SignalStatus taylorwin(usize n, i32 nbar, f64 sll_db, f64* out,
                                     Allocator* alloc = nullptr);

// Discrete prolate spheroidal sequences (Slepian tapers).
// Writes `k_eff = min(k, n)` column-major tapers (`n * k_eff` doubles) into
// `tapers_col_major` and `k_eff` concentration ratios into `ratios`.
[[nodiscard]] SignalStatus dpss(usize n, f64 nw, usize k, f64* tapers_col_major, f64* ratios,
                                Allocator* alloc = nullptr);

// ============================================================================
// Linear Convolution
// ============================================================================

// Direct 1D linear convolution. Writes `na + nb - 1` samples to `out`.
// No-op if `na == 0 || nb == 0`.
void conv_1d(const f64* a, usize na, const f64* b, usize nb, f64* out) noexcept;

// Direct 2D linear convolution (column-major `m x n` and `p x q`).
// Writes `(m + p - 1) x (n + q - 1)` column-major doubles to `out`.
// No-op if any dimension is 0.
void conv_2d(const f64* a, usize m, usize n, const f64* b, usize p, usize q, f64* out) noexcept;

// ============================================================================
// IIR, Biquad & Second-Order Sections (SOS) Filtering
// ============================================================================

// Direct-Form-II-transposed IIR filter (real).
// Assumes `a` is already normalized (`a[0] == 1`).
// Writes `len` samples to `y`. Optional initial state `zi` (length `zi_len`)
// and final state `zf` (length `max(nb, na) - 1` when `max(nb, na) > 1`).
void filter_df2t(const f64* b, usize nb, const f64* a, usize na, const f64* x, usize len, f64* y,
                 const f64* zi = nullptr, usize zi_len = 0, f64* zf = nullptr,
                 Allocator* alloc = nullptr);

// Direct-Form-II-transposed IIR filter (complex).
void filter_df2t_c64(const c64* b, usize nb, const c64* a, usize na, const c64* x, usize len,
                     c64* y, const c64* zi = nullptr, usize zi_len = 0, c64* zf = nullptr,
                     Allocator* alloc = nullptr);

// Single biquad section DF2T recurrence with zero initial state.
void biquad_df2t(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2, const f64* x, f64* y, usize n) noexcept;

// Single biquad section DF2T recurrence with explicit initial state.
void biquad_df2t_state(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2, const f64* x, f64* y, usize n,
                       f64 s1_init, f64 s2_init, BiquadState* final_state = nullptr) noexcept;

// Steady-state unit-input initial conditions for one normalized biquad section.
[[nodiscard]] BiquadState biquad_zi(f64 b0, f64 b1, f64 b2, f64 a1, f64 a2) noexcept;

// Apply an `L x 6` column-major SOS cascade to a 1D signal `x` of length `n`.
// Optional `zi` has length `2 * L` (interleaved `(s1, s2)` per section).
// Returns kZeroLeadingCoeff if any section has `a0 == 0`.
[[nodiscard]] SignalStatus sosfilt_col_major(const f64* sos, usize num_sections, const f64* x,
                                             f64* y, usize n, const f64* zi = nullptr,
                                             Allocator* alloc = nullptr);

// Zero-phase forward-backward SOS cascade on a 1D signal `x` of length `n`
// using odd-reflection edge padding (`edge = min(6 * L, n - 1)`) and
// DC-gain-scaled steady-state initial conditions.
[[nodiscard]] SignalStatus sosfiltfilt_col_major(const f64* sos, usize num_sections, const f64* x,
                                                 f64* y, usize n, Allocator* alloc = nullptr);

// ============================================================================
// Spectral & Phase Utilities
// ============================================================================

// Single-bin Goertzel DFT at 0-based (possibly fractional) bin index `k`.
// Computes sum_{m=0}^{n-1} x[m] * exp(-2*pi*i * k * m / n).
[[nodiscard]] c64 goertzel_bin(const f64* x, usize n, f64 k_zero_based) noexcept;
[[nodiscard]] c64 goertzel_bin_c64(const c64* x, usize n, f64 k_zero_based) noexcept;

// Phase unwrapping: removes jumps exceeding `period / 2` (default `2*pi`).
// Safe for in-place (`out == phase`).
void unwrap_phase(const f64* phase, usize n, f64* out, f64 period = kTau) noexcept;

} // namespace keel::signal
