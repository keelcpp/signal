#pragma once

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
};

struct BiquadState {
    f64 s1 = 0.0;
    f64 s2 = 0.0;
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
// Spectral & Phase Utilities
// ============================================================================

// Single-bin Goertzel DFT at 0-based (possibly fractional) bin index `k`.
// Computes sum_{m=0}^{n-1} x[m] * exp(-2*pi*i * k * m / n).
[[nodiscard]] c64 goertzel_bin(ArrayRef<f64> x, f64 k_zero_based) noexcept;
[[nodiscard]] c64 goertzel_bin_c64(ArrayRef<c64> x, f64 k_zero_based) noexcept;

// Phase unwrapping: removes jumps exceeding `period / 2` (default `2*pi`).
// Safe for in-place (`out.data() == phase.data()`). Requires `out.size() >= phase.size()`.
[[nodiscard]] SignalStatus unwrap_phase(ArrayRef<f64> phase, MutArrayRef<f64> out,
                                        f64 period = kTau) noexcept;

} // namespace keel::signal
