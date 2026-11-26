"""Differential reference verification for keel::signal against NumPy, SciPy, and MATLAB R2025b."""

import math
import os
import subprocess
import sys
import numpy as np
import scipy.fft
import scipy.signal as ss
import scipy.signal.windows as sw
import scipy.special as sp


def find_exe():
    candidates = [
        os.path.join("build", "Release", "test_signal.exe"),
        os.path.join("build", "test_signal.exe"),
        os.path.join("build", "test_signal"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"test_signal binary not found in {candidates}")


def parse_dump(text):
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if ";" in v:
            arr = []
            for item in v.split(","):
                re, im = item.split(";")
                arr.append(complex(float(re), float(im)))
            data[k] = np.array(arr, dtype=np.complex128)
        else:
            arr = [float(x) for x in v.split(",") if x]
            data[k] = np.array(arr, dtype=np.float64)
    return data


def run_matlab_ref():
    script = r"""
format long g;
pvec = @(name, v) fprintf('%s=%s\n', name, strjoin(arrayfun(@(x) sprintf('%.17g', x), v(:).', 'UniformOutput', false), ','));
pcvec = @(name, v) fprintf('%s=%s\n', name, strjoin(arrayfun(@(x) sprintf('%.17g;%.17g', real(x), imag(x)), v(:).', 'UniformOutput', false), ','));

pvec('win_hann_16', hann(16));
pvec('win_hamming_15', hamming(15));
pvec('win_blackman_16', blackman(16));
pvec('win_bartlett_15', bartlett(15));
pvec('win_triang_16', triang(16));
pvec('win_triang_15', triang(15));
pvec('win_kaiser_16_5', kaiser(16, 5.0));
pvec('win_tukey_16', tukeywin(16, 0.5));
pvec('win_flattop_16', flattopwin(16));
pvec('win_gauss_15', gausswin(15, 2.5));
pvec('win_cheb_15', chebwin(15, 80.0));
pvec('win_parzen_16', parzenwin(16));
pvec('win_nuttall_16', nuttallwin(16));
pvec('win_bh_16', blackmanharris(16));
pvec('win_bohman_15', bohmanwin(15));
pvec('win_barthann_15', barthannwin(15));
pvec('win_taylor_15', taylorwin(15, 4, -30.0));
[e, v] = dpss(32, 3.0, 3);
pvec('win_dpss_ratios', v);
pvec('win_dpss_col0', e(:,1));

a = [1 2 3 4]; b = [1 -0.5 2];
pvec('conv_1d', conv(a, b));
pvec('cconv_2', cconv(a, b, 2));
pvec('cconv_3', cconv(a, b, 3));
pvec('cconv_5', cconv(a, b, 5));
[dq, dr] = deconv(conv(a, b), b);
pvec('deconv_q', dq); pvec('deconv_r', dr);
pvec('xcorr', xcorr(a, b));
pvec('xcov_coeff', xcov(a, a, 3, 'coeff'));
ma = reshape(1:6, 2, 3); mb = [1 2; -1 0.5];
pvec('conv_2d', conv2(ma, mb));
pvec('xcorr2', xcorr2(ma, mb));

bf = [0.2 0.4 0.2]; af = [1 -0.5 0.25];
x32 = sin(0.25*(0:31)) + 0.3*cos(1.1*(0:31));
[y, zf] = filter(bf, af, x32);
pvec('filter_df2t', y); pvec('filter_df2t_zf', zf);
pvec('filtfilt', filtfilt(bf, af, x32));
[hz, ~] = freqz(bf, af, 16);
pcvec('freqz_h', hz);
[gd, ~] = grpdelay(bf, af, 16);
pvec('grpdelay', gd);
pvec('impz', impz(bf, af, 16));
pvec('stepz', stepz(bf, af, 16));
pvec('polystab', polystab([1 -2.5 2 -0.5]));

x16 = sin(0.31*(0:15)) + 0.4*cos(0.9*(0:15));
pvec('dct1_16', dct(x16, 'Type', 1));
pvec('dct2_16', dct(x16, 'Type', 2));
pvec('dct3_16', dct(x16, 'Type', 3));
pvec('dct4_16', dct(x16, 'Type', 4));
fwin = [1 2 -1 4 0.5 -2.5 3 1.5];
pvec('fwht_seq', fwht(fwin, 8, 'sequency'));
pvec('fwht_had', fwht(fwin, 8, 'hadamard'));
pvec('fwht_dya', fwht(fwin, 8, 'dyadic'));
pcvec('hilbert_15', hilbert(x16(1:15)));
pvec('unwrap_nan', unwrap([0 3 NaN -3 -2 3.5]));
seq = [1 0.5 0.25 0.1 0];
pvec('cceps_5', cceps(seq));
[rce, rmp] = rceps(seq);
pvec('rceps_5', rce); pvec('rceps_mp_5', rmp);
cz_in = (1:6) - 0.5i*(0:5);
pcvec('czt_5', czt(cz_in, 5, 0.95*exp(-0.4i), 1.05*exp(0.2i)));

mx = [1 5 2 8 3 1 4];
pvec('medfilt1_zp', medfilt1(mx, 3));
pvec('sgolay_2_5', sgolay(2, 5).');
pvec('sgolayfilt', sgolayfilt(mx, 2, 5));
[pv, pl, ~, pp] = findpeaks([0 1 5 2 3 1 0 2 8 1 0]);
pvec('findpeaks_val', pv); pvec('findpeaks_loc', pl); pvec('findpeaks_prom', pp);

x64 = cos(2*pi*0.125*(0:63)) + 0.3*sin(2*pi*0.3*(0:63));
[p64, f64] = periodogram(x64, [], 64, 100);
pvec('per_64_p', p64); pvec('per_64_f', f64);
[p5, f5] = periodogram([1 2 3 4 5], [], 5, 10);
pvec('per_5_p', p5); pvec('per_5_f', f5);
[pwp, pwf] = pwelch(x64, hamming(16), 8, 16, 100);
pvec('pwelch_p', pwp); pvec('pwelch_f', pwf);
[py15, fy15] = pyulear(x64, 4, 15);
pvec('pyulear_15_p', py15); pvec('pyulear_15_f', fy15);
[pb16, fb16] = pburg(x64, 4, 16);
pvec('pburg_16_p', pb16); pvec('pburg_16_f', fb16);

[bb, ba] = butter(4, 0.35); pvec('butter_lp4_b', bb); pvec('butter_lp4_a', ba);
[bb, ba] = butter(2, [0.2 0.5]); pvec('butter_bp2_b', bb); pvec('butter_bp2_a', ba);
[bb, ba] = cheby1(3, 1, 0.35, 'high'); pvec('cheby1_hp3_b', bb); pvec('cheby1_hp3_a', ba);
[bb, ba] = cheby2(3, 40, 0.35); pvec('cheby2_lp3_b', bb); pvec('cheby2_lp3_a', ba);
[bb, ba] = ellip(3, 1, 40, 0.35); pvec('ellip_lp3_b', bb); pvec('ellip_lp3_a', ba);
[bb, ba] = besself(4, 10); pvec('besself_4_b', bb); pvec('besself_4_a', ba);
[~, bp9, ~] = besselap(9); pcvec('besselap_9_p', bp9);
[bb, ba] = bilinear([1 2], [1 3 2], 10); pvec('bilinear_b', bb); pvec('bilinear_a', ba);
[bb, ba] = impinvar([1 2], [1 3 2], 10); pvec('impinvar_b', bb); pvec('impinvar_a', ba);
[no, wno] = buttord(0.2, 0.3, 3, 40); pvec('buttord_lp', [no wno]);
[no, wno] = buttord([0.2 0.5], [0.1 0.6], 3, 40); pvec('buttord_bp', [no wno]);
[no, wno] = cheb1ord(0.2, 0.3, 3, 40); pvec('cheb1ord_lp', [no wno]);
[no, wno] = cheb2ord(0.2, 0.3, 3, 40); pvec('cheb2ord_lp', [no wno]);
[no, wno] = ellipord(0.2, 0.3, 3, 40); pvec('ellipord_lp', [no wno]);

pvec('fir1_lp20', fir1(20, 0.4));
pvec('fir1_hp20', fir1(20, 0.4, 'high'));
pvec('firls_20', firls(20, [0 0.3 0.4 1], [1 1 0 0]));
pvec('firpm_20', firpm(20, [0 0.3 0.4 1], [1 1 0 0]));
pvec('upfirdn_3_2', upfirdn([1 2 -1 3 0.5 -2], [0.2 0.5 0.3 0.1], 3, 2));
pvec('resample_3_2', resample(sin(0.2*(0:31)), 3, 2, 10, 5));

x64m = cos(0.4*(0:63)) + 0.25*sin(1.1*(0:63));
pvec('aryule_4_a', aryule(x64m, 4));
pvec('arburg_4_a', arburg(x64m, 4));
pvec('arcov_4_a', arcov(x64m, 4));
pvec('armcov_4_a', armcov(x64m, 4));
pvec('poly2rc_k', poly2rc([1 -0.75 0.5 -0.1]));
pvec('poly2lsf', poly2lsf([1 -0.75 0.5 -0.1]));
pvec('enbw_ham16', enbw(hamming(16), 1000));
tw = [-1 -0.5 -0.25 0 0.25 0.5];
pvec('rectpuls', rectpuls(tw, 1));
pvec('tripuls', tripuls(tw, 1));
pvec('sinc', sinc(tw));
pvec('diric_5', diric(tw, 5));
tc = 0:0.1:0.5;
pvec('chirp_lin', chirp(tc, 1, 1, 5, 'linear'));
pvec('chirp_qconv', chirp(tc, 1, 1, 5, 'quadratic', 0, 'convex'));
pvec('chirp_log', chirp(tc, 1, 1, 5, 'logarithmic'));
rfc = rainflow([0 2 -1 3 -2 1 0]);
pvec('rainflow', reshape(rfc(:,1:3).', 1, []));
ti = (0:14)/15; xe = (1 + 0.5*cos(2*pi*2*ti)).*cos(2*pi*5*ti);
[~, fe] = envspectrum(xe, 15);
pvec('envspec_15_f', fe);
"""
    proc = subprocess.run(
        ["matlab", "-batch", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return parse_dump(proc.stdout)


def main():
    exe = find_exe()
    proc = subprocess.run([exe, "--dump-ref"], capture_output=True, text=True, check=True)
    got = parse_dump(proc.stdout)

    checks = 0
    failures = 0

    def check(name, actual, expected, atol=1e-10, rtol=1e-9, label="ref"):
        nonlocal checks, failures
        checks += 1
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.shape != expected.shape:
            print(f"FAIL [{label}] {name}: shape {actual.shape} != {expected.shape}")
            failures += 1
            return
        if np.iscomplexobj(actual) or np.iscomplexobj(expected):
            ok = np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
            err = float(np.nanmax(np.abs(actual - expected))) if actual.size else 0.0
        else:
            ok = np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
            diff = np.abs(actual - expected)
            both_nan = np.isnan(actual) & np.isnan(expected)
            diff = np.where(both_nan, 0.0, diff)
            err = float(np.max(diff)) if actual.size else 0.0
        if not ok:
            print(f"FAIL [{label}] {name}: max_err={err:.3e} (atol={atol}, rtol={rtol})")
            print(f"  got={actual[:8]}")
            print(f"  ref={expected[:8]}")
            failures += 1

    # --- SciPy / NumPy checks ---
    check("win_hann_16", got["win_hann_16"], sw.hann(16, sym=True), label="scipy")
    check("win_hamming_15", got["win_hamming_15"], sw.hamming(15, sym=True), label="scipy")
    check("win_blackman_16", got["win_blackman_16"], sw.blackman(16, sym=True), label="scipy")
    check("win_bartlett_15", got["win_bartlett_15"], sw.bartlett(15, sym=True), label="scipy")
    check("win_triang_16", got["win_triang_16"], sw.triang(16, sym=True), label="scipy")
    check("win_kaiser_16_5", got["win_kaiser_16_5"], sw.kaiser(16, 5.0, sym=True), label="scipy")
    # Exact scaled-Bessel reference for large beta=800 (where naive I0 overflows in SciPy/MATLAB)
    alpha_k = 2.0 * np.arange(15) / 14.0 - 1.0
    arg_k = 800.0 * np.sqrt(np.maximum(0.0, 1.0 - alpha_k * alpha_k))
    kaiser_800_exact = (sp.i0e(arg_k) / sp.i0e(800.0)) * np.exp(arg_k - 800.0)
    check("win_kaiser_15_800", got["win_kaiser_15_800"], kaiser_800_exact, atol=1e-12, label="scipy-i0e")
    check("win_tukey_16", got["win_tukey_16"], sw.tukey(16, 0.5, sym=True), label="scipy")
    check("win_flattop_16", got["win_flattop_16"], sw.flattop(16, sym=True), atol=1e-6, label="scipy")
    check("win_parzen_16", got["win_parzen_16"], sw.parzen(16, sym=True), label="scipy")
    check("win_nuttall_16", got["win_nuttall_16"], sw.nuttall(16, sym=True), atol=1e-6, label="scipy")
    check("win_bh_16", got["win_bh_16"], sw.blackmanharris(16, sym=True), label="scipy")
    check("win_bohman_15", got["win_bohman_15"], sw.bohman(15, sym=True), label="scipy")
    check("win_barthann_15", got["win_barthann_15"], sw.barthann(15, sym=True), label="scipy")
    check("win_cheb_15", got["win_cheb_15"], sw.chebwin(15, 80.0, sym=True), label="scipy")
    check("win_taylor_15", got["win_taylor_15"], sw.taylor(15, nbar=4, sll=30.0, norm=False, sym=True), atol=1e-10, label="scipy")
    dpss_t, dpss_r = sw.dpss(32, 3.0, Kmax=3, return_ratios=True)
    check("win_dpss_ratios", got["win_dpss_ratios"], dpss_r, atol=1e-10, label="scipy")
    check("win_dpss_col0", np.abs(got["win_dpss_col0"]), np.abs(dpss_t[0]), atol=1e-10, label="scipy")

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, -0.5, 2.0])
    check("conv_1d", got["conv_1d"], np.convolve(a, b), label="numpy")
    x16 = np.sin(0.31 * np.arange(16)) + 0.4 * np.cos(0.9 * np.arange(16))
    for t in (1, 2, 3, 4):
        check(f"dct{t}_16", got[f"dct{t}_16"], scipy.fft.dct(x16, type=t, norm="ortho"), label="scipy")
    check("hilbert_15", got["hilbert_15"], ss.hilbert(x16[:15]), label="scipy")

    bf, af = np.array([0.2, 0.4, 0.2]), np.array([1.0, -0.5, 0.25])
    x32 = np.sin(0.25 * np.arange(32)) + 0.3 * np.cos(1.1 * np.arange(32))
    check("filter_df2t", got["filter_df2t"], ss.lfilter(bf, af, x32), label="scipy")
    check("filtfilt", got["filtfilt"], ss.filtfilt(bf, af, x32, padlen=6), atol=1e-9, label="scipy")
    _, hz = ss.freqz(bf, af, worN=16, whole=False)
    check("freqz_h", got["freqz_h"], hz, label="scipy")
    _, gd = ss.group_delay((bf, af), w=np.linspace(0, np.pi, 16, endpoint=False))
    check("grpdelay", got["grpdelay"], gd, atol=1e-9, label="scipy")

    b_lp, a_lp = ss.butter(4, 0.35)
    check("butter_lp4_b", got["butter_lp4_b"], b_lp, label="scipy")
    check("butter_lp4_a", got["butter_lp4_a"], a_lp, label="scipy")
    b_bp, a_bp = ss.butter(2, [0.2, 0.5], btype="bandpass")
    check("butter_bp2_b", got["butter_bp2_b"], b_bp, label="scipy")
    check("butter_bp2_a", got["butter_bp2_a"], a_bp, label="scipy")
    b_c1, a_c1 = ss.cheby1(3, 1.0, 0.35, btype="highpass")
    check("cheby1_hp3_b", got["cheby1_hp3_b"], b_c1, label="scipy")
    check("cheby1_hp3_a", got["cheby1_hp3_a"], a_c1, label="scipy")
    b_c2, a_c2 = ss.cheby2(3, 40.0, 0.35)
    check("cheby2_lp3_b", got["cheby2_lp3_b"], b_c2, label="scipy")
    check("cheby2_lp3_a", got["cheby2_lp3_a"], a_c2, label="scipy")
    b_el, a_el = ss.ellip(3, 1.0, 40.0, 0.35)
    check("ellip_lp3_b", got["ellip_lp3_b"], b_el, atol=1e-9, label="scipy")
    check("ellip_lp3_a", got["ellip_lp3_a"], a_el, atol=1e-9, label="scipy")
    b_bi, a_bi = ss.bilinear([1.0, 2.0], [1.0, 3.0, 2.0], fs=10.0)
    check("bilinear_b", got["bilinear_b"], b_bi, label="scipy")
    check("bilinear_a", got["bilinear_a"], a_bi, label="scipy")

    check("fir1_lp20", got["fir1_lp20"], ss.firwin(21, 0.4, window="hamming"), label="scipy")
    check("fir1_hp20", got["fir1_hp20"], ss.firwin(21, 0.4, window="hamming", pass_zero=False), label="scipy")
    check("firls_20", got["firls_20"], ss.firls(21, [0, 0.3, 0.4, 1.0], [1, 1, 0, 0]), atol=1e-9, label="scipy")
    check("upfirdn_3_2", got["upfirdn_3_2"], ss.upfirdn([0.2, 0.5, 0.3, 0.1], [1, 2, -1, 3, 0.5, -2], 3, 2), label="scipy")
    x64 = np.cos(2 * np.pi * 0.125 * np.arange(64)) + 0.3 * np.sin(2 * np.pi * 0.3 * np.arange(64))
    f64, p64 = ss.periodogram(x64, fs=100.0, nfft=64, detrend=False)
    check("per_64_p", got["per_64_p"], p64, label="scipy")
    check("per_64_f", got["per_64_f"], f64, label="scipy")
    f5, p5 = ss.periodogram([1.0, 2.0, 3.0, 4.0, 5.0], fs=10.0, nfft=5, detrend=False)
    check("per_5_p", got["per_5_p"], p5, label="scipy")
    check("per_5_f", got["per_5_f"], f5, label="scipy")
    # Exact FFT analytic envelope spectrum on odd N=15
    ti15 = np.arange(15) / 15.0
    xe15 = (1.0 + 0.5 * np.cos(2 * np.pi * 2.0 * ti15)) * np.cos(2 * np.pi * 5.0 * ti15)
    env15 = np.abs(ss.hilbert(xe15))
    env15 -= np.mean(env15)
    z15 = np.fft.rfft(env15)
    es15_ref = np.abs(z15) * (2.0 / 15.0)
    es15_ref[0] *= 0.5
    check("envspec_15_es", got["envspec_15_es"], es15_ref, atol=1e-12, label="scipy-analytic")
    check("sinc", got["sinc"], np.sinc(np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5])), label="numpy")

    # --- MATLAB R2025b checks ---
    ml = run_matlab_ref()
    # Root sets may be ordered differently; compare sorted for besselap_9_p
    for key, ref_val in ml.items():
        if key not in got:
            print(f"FAIL [matlab] missing key in C++ dump: {key}")
            failures += 1
            continue
        if key == "besselap_9_p":
            s_got = np.sort_complex(got[key])
            s_ref = np.sort_complex(ref_val)
            check(key, s_got, s_ref, atol=1e-9, rtol=1e-8, label="matlab")
        elif key == "win_dpss_col0":
            # Sign convention may flip by -1 depending on eigenvector orientation
            sign = 1.0 if np.dot(got[key], ref_val) >= 0 else -1.0
            check(key, sign * got[key], ref_val, atol=1e-10, label="matlab")
        elif key == "firpm_20":
            check(key, got[key], ref_val, atol=1e-3, rtol=1e-2, label="matlab")
        elif key == "envspec_15_es":
            check(key, got[key], ref_val, atol=1e-10, label="matlab")
        elif key in ("win_flattop_16", "win_nuttall_16"):
            check(key, got[key], ref_val, atol=1e-6, label="matlab")
        elif key in ("polystab", "poly2lsf"):
            check(key, got[key], ref_val, atol=1e-8, rtol=1e-8, label="matlab")
        elif key == "resample_3_2":
            check(key, got[key], ref_val, atol=1e-9, label="matlab")
        elif key in ("filtfilt",):
            check(key, got[key], ref_val, atol=1e-9, label="matlab")
        elif key in ("ellip_lp3_b", "ellip_lp3_a", "cheby2_lp3_b", "cheby2_lp3_a", "impinvar_b", "impinvar_a"):
            check(key, got[key], ref_val, atol=1e-9, rtol=1e-8, label="matlab")
        elif key in ("besself_4_b", "besself_4_a"):
            check(key, got[key], ref_val, atol=1e-8, rtol=1e-8, label="matlab")
        else:
            check(key, got[key], ref_val, atol=1e-10, rtol=1e-9, label="matlab")

    print(f"Reference checks completed: {checks - failures}/{checks} PASSED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
