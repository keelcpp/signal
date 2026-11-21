# keel::signal

Zero-STL C++20 digital signal processing kernels built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Window functions**: `rectwin`, `hann`, `hamming`, `blackman`, `bartlett`, `triang`, `kaiser`, `tukeywin`, `flattopwin`, `gausswin`, `chebwin`, `parzenwin`, `nuttallwin`, `taylorwin`, `blackmanharris`, `bohmanwin`, `barthannwin`, and Slepian tapers (`dpss`).
- **Linear convolution**: 1D (`conv_1d`) and 2D column-major (`conv_2d`).
- **IIR & SOS filtering**: Direct-Form-II-transposed real & complex IIR (`filter_df2t`, `filter_df2t_c64`), biquad kernels (`biquad_df2t`, `biquad_df2t_state`, `biquad_zi`), second-order-sections cascade (`sosfilt_col_major`), and zero-phase forward-backward SOS filtering (`sosfiltfilt_col_major`).
- **Transforms & envelopes**: single-bin Goertzel DFT (`goertzel_bin`, `goertzel_bin_c64`), phase unwrapping (`unwrap_phase`), orthonormal DCT/IDCT Types 1–4 (`dct`, `idct`, FFT-accelerated for Types 2/3), Fast Walsh-Hadamard Transform (`fwht`, `ifwht` in sequency/Hadamard/dyadic orderings), analytic Hilbert signal (`hilbert`), and analytic/FIR/RMS envelopes (`envelope_analytic`, `envelope_fir`, `envelope_rms`).
- **Smoothing, gap-filling & peaks**: sliding median (`medfilt1`), forward-backward Burg AR NaN interpolation (`fillgaps`, `fillgaps_c64`), Savitzky-Golay projection/differentiation matrices & filtering (`sgolay`, `sgolayfilt`), and topographic peak detection (`findpeaks`).
- **Spectral & time-frequency estimation**: `periodogram`, `periodogram_c64`, Welch PSD (`pwelch`, `pwelch_c64`), cross-spectral estimators (`cpsd`, `mscohere`, `tfestimate`), parametric AR PSDs (`pyulear`, `pburg`), `spectrogram`, `stft`/`stft_c64`, `istft`, and `iscola`.

## Design

- Strict zero-STL public headers and implementation files (`keel_enforce_guidelines` + `CheckHeaderPurity`).
- Compiles cleanly under `-fno-exceptions -fno-rtti`.
- All public symbols live in `namespace keel::signal` under `<keel/signal.h>`.

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
