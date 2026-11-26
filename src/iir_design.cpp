#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::signal {

namespace {

struct KeyedRoot {
    f64 key = 0.0;
    usize orig_idx = 0;
    c64 up;
};

Array<c64> order_roots_canonical(Array<c64> v, bool by_abs_real, bool key_desc, bool neg_first,
                                 Allocator* alloc) {
    Array<KeyedRoot> uppers(alloc);
    Array<c64> reals(alloc);
    Array<u8> used(v.size(), u8{0}, alloc);
    for (usize i = 0; i < v.size(); ++i) {
        if (used[i] != 0 || v[i].imag() == 0.0) {
            continue;
        }
        for (usize j = i + 1; j < v.size(); ++j) {
            if (used[j] != 0) {
                continue;
            }
            if (keel::abs(v[j] - keel::conj(v[i])) <= 1e-10 * (1.0 + keel::abs(v[i]))) {
                used[i] = 1;
                used[j] = 1;
                const c64 up = (v[i].imag() > 0.0) ? v[i] : v[j];
                const f64 key = by_abs_real ? keel::abs(up.real()) : up.imag();
                uppers.push_back(KeyedRoot{key, i, up});
                break;
            }
        }
    }
    for (usize i = 0; i < v.size(); ++i) {
        if (used[i] == 0) {
            reals.push_back(v[i]);
        }
    }
    keel::sort(uppers, [key_desc](const KeyedRoot& a, const KeyedRoot& b) noexcept {
        if (a.key != b.key) {
            return key_desc ? (a.key > b.key) : (a.key < b.key);
        }
        return a.orig_idx < b.orig_idx;
    });
    Array<c64> out(alloc);
    out.reserve(v.size());
    for (usize i = 0; i < uppers.size(); ++i) {
        const c64 up = uppers[i].up;
        const c64 lo = keel::conj(up);
        out.push_back(neg_first ? lo : up);
        out.push_back(neg_first ? up : lo);
    }
    for (usize i = 0; i < reals.size(); ++i) {
        out.push_back(reals[i]);
    }
    return out;
}

// Complete elliptic integral K(m) via arithmetic-geometric mean (m = k^2).
f64 ellip_km(f64 m) noexcept {
    if (m >= 1.0) return Limits<f64>::infinity();
    if (m <= 0.0) return 0.5 * kPi;
    f64 a = 1.0;
    f64 b = keel::sqrt(1.0 - m);
    for (i32 i = 0; i < 64; ++i) {
        const f64 an = 0.5 * (a + b);
        const f64 bn = keel::sqrt(a * b);
        if (keel::abs(an - bn) < 1e-17 * an) {
            a = an;
            break;
        }
        a = an;
        b = bn;
    }
    return kPi / (2.0 * a);
}

struct SnCnDn {
    f64 sn = 0.0;
    f64 cn = 1.0;
    f64 dn = 1.0;
};

SnCnDn ellipj_real(f64 u, f64 k) noexcept {
    const f64 m = k * k;
    if (m == 0.0) {
        return {keel::sin(u), keel::cos(u), 1.0};
    }
    if (m == 1.0) {
        const f64 t = keel::tanh(u);
        const f64 s = 1.0 / keel::cosh(u);
        return {t, s, s};
    }
    constexpr i32 kMax = 16;
    f64 a[kMax + 1]{};
    f64 c[kMax + 1]{};
    a[0] = 1.0;
    f64 b = keel::sqrt(1.0 - m);
    c[0] = keel::sqrt(m);
    i32 n = 0;
    for (i32 i = 1; i <= kMax; ++i) {
        const f64 an = 0.5 * (a[i - 1] + b);
        const f64 bn = keel::sqrt(a[i - 1] * b);
        const f64 cn = 0.5 * (a[i - 1] - b);
        a[i] = an;
        c[i] = cn;
        b = bn;
        n = i;
        if (keel::abs(cn) < 1e-15 * keel::abs(an)) {
            break;
        }
    }
    f64 phi = keel::ldexp(a[n] * u, n);
    for (i32 i = n; i >= 1; --i) {
        phi = 0.5 * (phi + keel::asin((c[i] / a[i]) * keel::sin(phi)));
    }
    const f64 sn = keel::sin(phi);
    return {sn, keel::cos(phi), keel::sqrt(1.0 - m * sn * sn)};
}

f64 asn_real(f64 z, f64 k) noexcept {
    if (z == 0.0) return 0.0;
    const f64 sign = (z < 0.0) ? -1.0 : 1.0;
    z = keel::abs(z);
    if (z >= 1.0) {
        return sign * ellip_km(k * k);
    }
    if (k >= 1.0 - 1e-12) {
        return sign * keel::atanh(z);
    }
    const f64 k_val = ellip_km(k * k);
    f64 lo = 0.0;
    f64 hi = k_val;
    for (i32 it = 0; it < 200; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        const SnCnDn j = ellipj_real(mid, k);
        if (j.sn < z) lo = mid;
        else hi = mid;
        if ((hi - lo) < 1e-15 * (1.0 + k_val)) break;
    }
    return sign * 0.5 * (lo + hi);
}

f64 ellipdeg(i32 n, f64 k1) noexcept {
    const f64 k1_val = ellip_km(k1 * k1);
    const f64 k1p_val = ellip_km(1.0 - k1 * k1);
    const f64 target = k1p_val / k1_val / as<f64>(n);
    f64 lo = 1e-15;
    f64 hi = 1.0 - 1e-15;
    for (i32 it = 0; it < 200; ++it) {
        const f64 mid = 0.5 * (lo + hi);
        const f64 kmid = ellip_km(mid * mid);
        const f64 kpmid = ellip_km(1.0 - mid * mid);
        if (kpmid / kmid > target) lo = mid;
        else hi = mid;
        if (hi - lo < 1e-15 * (1.0 + keel::abs(mid))) break;
    }
    return 0.5 * (lo + hi);
}

c64 cd_complex(c64 uc, f64 k) noexcept {
    const f64 x = uc.real();
    const f64 y = uc.imag();
    const f64 kp = keel::sqrt(1.0 - k * k);
    const SnCnDn xv = ellipj_real(x, k);
    const SnCnDn yv = ellipj_real(y, kp);
    const f64 d = yv.cn * yv.cn + k * k * xv.sn * xv.sn * yv.sn * yv.sn;
    if (keel::abs(d) < 1e-300) {
        return c64(0.0, 0.0);
    }
    const c64 cn(xv.cn * yv.cn / d, -xv.sn * xv.dn * yv.sn * yv.dn / d);
    const c64 dn(xv.dn * yv.cn * yv.dn / d, -k * k * xv.sn * xv.cn * yv.sn / d);
    return cn / dn;
}

Array<c64> poly_mul_c64(ArrayRef<c64> a1, ArrayRef<c64> b1, Allocator* alloc) {
    Array<c64> r(a1.size() + b1.size() - 1, c64(0.0, 0.0), alloc);
    for (usize i = 0; i < a1.size(); ++i) {
        for (usize j = 0; j < b1.size(); ++j) {
            r[i + j] += a1[i] * b1[j];
        }
    }
    return r;
}

Array<c64> poly_mul_lin_c64(ArrayRef<c64> a1, c64 alpha_term, Allocator* alloc) {
    const c64 lin[2] = {c64(1.0, 0.0), -alpha_term};
    return poly_mul_c64(a1, ArrayRef<c64>(lin, 2), alloc);
}

Array<c64> taylor_about(ArrayRef<f64> c_hl, c64 p, Allocator* alloc) {
    Array<c64> c(c_hl.size(), c64(0.0, 0.0), alloc);
    for (usize i = 0; i < c_hl.size(); ++i) {
        c[i] = c64(c_hl[i], 0.0);
    }
    Array<c64> t(alloc);
    t.reserve(c_hl.size());
    while (!c.empty()) {
        if (c.size() == 1) {
            t.push_back(c[0]);
            break;
        }
        Array<c64> q(c.size() - 1, c64(0.0, 0.0), alloc);
        q[0] = c[0];
        for (usize i = 1; i + 1 < c.size(); ++i) {
            q[i] = c[i] + p * q[i - 1];
        }
        t.push_back(c[c.size() - 1] + p * q[q.size() - 1]);
        c = keel::move(q);
    }
    return t;
}

inline f64 prewarp_digital(f64 wn_norm) noexcept {
    return 2.0 * keel::tan(0.5 * kPi * wn_norm);
}

SignalStatus finish_iir_design(const ZpkResult& proto, ArrayRef<f64> wn, FilterType ftype,
                               bool analog, TfResult& out, Allocator* alloc) {
    const bool is_band = (ftype == FilterType::kBandpass || ftype == FilterType::kBandstop);
    if (is_band ? (wn.size() != 2) : (wn.size() != 1)) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < wn.size(); ++i) {
        if (!keel::is_finite(wn[i]) || !(wn[i] > 0.0) || (!analog && !(wn[i] < 1.0))) {
            return SignalStatus::kInvalidArgument;
        }
    }
    if (is_band && !(wn[0] < wn[1])) {
        return SignalStatus::kInvalidArgument;
    }

    f64 wo_analog[2] = {0.0, 0.0};
    for (usize i = 0; i < wn.size(); ++i) {
        wo_analog[i] = analog ? wn[i] : prewarp_digital(wn[i]);
    }

    ZpkResult xform;
    SignalStatus st = SignalStatus::kOk;
    if (ftype == FilterType::kLowpass) {
        st = lp2lp(proto.z.as_ref(), proto.p.as_ref(), proto.k, wo_analog[0], xform, alloc);
    } else if (ftype == FilterType::kHighpass) {
        st = lp2hp(proto.z.as_ref(), proto.p.as_ref(), proto.k, wo_analog[0], xform, alloc);
    } else {
        const f64 wo = keel::sqrt(wo_analog[0] * wo_analog[1]);
        const f64 bw = wo_analog[1] - wo_analog[0];
        if (ftype == FilterType::kBandpass) {
            st = lp2bp(proto.z.as_ref(), proto.p.as_ref(), proto.k, wo, bw, xform, alloc);
        } else {
            st = lp2bs(proto.z.as_ref(), proto.p.as_ref(), proto.k, wo, bw, xform, alloc);
        }
    }
    if (st != SignalStatus::kOk) {
        return st;
    }

    Array<f64> b = keel::poly::from_roots(xform.z.as_ref(), alloc);
    Array<f64> a = keel::poly::from_roots(xform.p.as_ref(), alloc);
    for (usize i = 0; i < b.size(); ++i) {
        b[i] *= xform.k;
    }
    if (b.size() < a.size()) {
        Array<f64> b_pad(a.size(), 0.0, alloc);
        const usize off = a.size() - b.size();
        for (usize i = 0; i < b.size(); ++i) {
            b_pad[off + i] = b[i];
        }
        b = keel::move(b_pad);
    }

    if (analog) {
        out.b = keel::move(b);
        out.a = keel::move(a);
        return SignalStatus::kOk;
    }
    return bilinear(b.as_ref(), a.as_ref(), 1.0, 0.0, out, alloc);
}

inline f64 prewarp_wp(f64 w, bool analog) noexcept {
    return analog ? w : keel::tan(0.5 * kPi * w);
}

inline f64 dewarp_wp(f64 omega, bool analog) noexcept {
    return analog ? omega : ((2.0 / kPi) * keel::atan(omega));
}

struct OrdNorm {
    f64 ratio = 0.0;
    f64 wo_stopband = 0.0;
    f64 wo_low = 0.0;
    f64 wo_high = 0.0;
    FilterType ftype = FilterType::kLowpass;
};

SignalStatus normalise_ord(ArrayRef<f64> wp, ArrayRef<f64> ws, bool analog, OrdNorm& n,
                           bool tighten_bs = true) noexcept {
    if (wp.size() != ws.size() || (wp.size() != 1 && wp.size() != 2)) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < wp.size(); ++i) {
        if (!keel::is_finite(wp[i]) || !keel::is_finite(ws[i]) || !(wp[i] > 0.0) || !(ws[i] > 0.0) ||
            (!analog && (!(wp[i] < 1.0) || !(ws[i] < 1.0)))) {
            return SignalStatus::kInvalidArgument;
        }
    }
    if (wp.size() == 1) {
        const f64 op = prewarp_wp(wp[0], analog);
        const f64 os = prewarp_wp(ws[0], analog);
        if (op == os) {
            return SignalStatus::kInvalidArgument;
        }
        if (op < os) {
            n.ftype = FilterType::kLowpass;
            n.ratio = os / op;
        } else {
            n.ftype = FilterType::kHighpass;
            n.ratio = op / os;
        }
        n.wo_stopband = os;
        return SignalStatus::kOk;
    }
    if (!(wp[0] < wp[1]) || !(ws[0] < ws[1])) {
        return SignalStatus::kInvalidArgument;
    }
    f64 op1 = prewarp_wp(wp[0], analog);
    f64 op2 = prewarp_wp(wp[1], analog);
    const f64 os1 = prewarp_wp(ws[0], analog);
    const f64 os2 = prewarp_wp(ws[1], analog);
    const bool bp = (os1 < op1 && op2 < os2);
    const bool bs = (op1 < os1 && os2 < op2);
    if (!bp && !bs) {
        return SignalStatus::kInvalidArgument;
    }
    n.ftype = bp ? FilterType::kBandpass : FilterType::kBandstop;
    if (bp) {
        const f64 wo = keel::sqrt(op1 * op2);
        const f64 bw = op2 - op1;
        n.wo_low = op1;
        n.wo_high = op2;
        const f64 m1 = keel::abs((os1 * os1 - wo * wo) / (bw * os1));
        const f64 m2 = keel::abs((os2 * os2 - wo * wo) / (bw * os2));
        n.ratio = keel::min(m1, m2);
    } else {
        if (tighten_bs) {
            // Tighten the looser passband edge so geometric center matches stopband center.
            const f64 os_prod = os1 * os2;
            if (op1 * op2 < os_prod) {
                op1 = os_prod / op2;
            } else {
                op2 = os_prod / op1;
            }
        }
        const f64 wo = keel::sqrt(op1 * op2);
        const f64 bw = op2 - op1;
        n.wo_low = op1;
        n.wo_high = op2;
        const f64 m1 = keel::abs((bw * os1) / (os1 * os1 - wo * wo));
        const f64 m2 = keel::abs((bw * os2) / (os2 * os2 - wo * wo));
        n.ratio = keel::min(m1, m2);
    }
    if (!(n.ratio > 1.0)) {
        return SignalStatus::kInvalidArgument;
    }
    return SignalStatus::kOk;
}

f64 remlpord(f64 f1, f64 f2, f64 d1, f64 d2) noexcept {
    static constexpr f64 kAA[3][3] = {
        {-4.278e-01, -4.761e-01, 0.0},
        {-5.941e-01,  7.114e-02, 0.0},
        {-2.660e-03,  5.309e-03, 0.0},
    };
    const f64 ld1 = keel::log10(d1);
    const f64 ld2 = keel::log10(d2);
    const f64 v[3] = {1.0, ld1, ld1 * ld1};
    const f64 w[3] = {1.0, ld2, ld2 * ld2};
    f64 d = 0.0;
    for (i32 i = 0; i < 3; ++i) {
        for (i32 j = 0; j < 3; ++j) {
            d += v[i] * kAA[i][j] * w[j];
        }
    }
    const f64 fk = 11.01217 + 0.51244 * (ld1 - ld2);
    const f64 df = keel::abs(f2 - f1);
    return d / df - fk * df + 1.0;
}

f64 kaiser_beta(f64 atten) noexcept {
    if (atten > 50.0) return 0.1102 * (atten - 8.7);
    if (atten >= 21.0) return 0.5842 * keel::pow(atten - 21.0, 0.4) + 0.07886 * (atten - 21.0);
    return 0.0;
}

struct KaisLpRes {
    f64 l = 0.0;
    f64 beta = 0.0;
};

KaisLpRes kaislpord(f64 f1, f64 f2, f64 d1, f64 d2) noexcept {
    const f64 delta = keel::min(d1, d2);
    const f64 atten = -20.0 * keel::log10(delta);
    const f64 d = (atten - 7.95) / (kTau * 2.285);
    const f64 l = d / keel::abs(f2 - f1) + 1.0;
    return {l, kaiser_beta(atten)};
}

} // namespace

// ============================================================================
// Analog Prototypes
// ============================================================================

SignalStatus buttap(i32 n, ZpkResult& out, Allocator* alloc) {
    if (n < 0) return SignalStatus::kInvalidArgument;
    out.z = Array<c64>(alloc);
    Array<c64> p(as<usize>(n), c64(0.0, 0.0), alloc);
    for (i32 k = 1; k <= n; ++k) {
        const f64 theta = kPi * (2.0 * as<f64>(k) - 1.0) / (2.0 * as<f64>(n)) + 0.5 * kPi;
        p[as<usize>(k - 1)] = keel::polar(1.0, theta);
    }
    out.p = order_roots_canonical(keel::move(p), false, true, false, alloc);
    out.k = 1.0;
    return SignalStatus::kOk;
}

SignalStatus cheb1ap(i32 n, f64 rp_db, ZpkResult& out, Allocator* alloc) {
    if (n <= 0 || !keel::is_finite(rp_db) || !(rp_db > 0.0)) return SignalStatus::kInvalidArgument;
    const f64 eps = keel::sqrt(keel::pow(10.0, rp_db / 10.0) - 1.0);
    const f64 mu = keel::asinh(1.0 / eps) / as<f64>(n);
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(as<usize>(n), c64(0.0, 0.0), alloc);
    for (i32 k = 1; k <= n; ++k) {
        const f64 theta = kPi * (2.0 * as<f64>(k) - 1.0) / (2.0 * as<f64>(n));
        out.p[as<usize>(k - 1)] = c64(-keel::sinh(mu) * keel::sin(theta),
                                      keel::cosh(mu) * keel::cos(theta));
    }
    c64 kc(1.0, 0.0);
    for (usize i = 0; i < out.p.size(); ++i) {
        kc *= -out.p[i];
    }
    f64 k = kc.real();
    if ((n % 2) == 0) {
        k /= keel::sqrt(1.0 + eps * eps);
    }
    out.k = k;
    return SignalStatus::kOk;
}

SignalStatus cheb2ap(i32 n, f64 rs_db, ZpkResult& out, Allocator* alloc) {
    if (n <= 0 || !keel::is_finite(rs_db) || !(rs_db > 0.0)) return SignalStatus::kInvalidArgument;
    const f64 delta = 1.0 / keel::sqrt(keel::pow(10.0, rs_db / 10.0) - 1.0);
    const f64 mu = keel::asinh(1.0 / delta) / as<f64>(n);

    out.p = Array<c64>(as<usize>(n), c64(0.0, 0.0), alloc);
    Array<c64> z(alloc);
    for (i32 k = 1; k <= n; ++k) {
        const f64 theta = kPi * (2.0 * as<f64>(k) - 1.0) / (2.0 * as<f64>(n));
        const c64 p1(-keel::sinh(mu) * keel::sin(theta), keel::cosh(mu) * keel::cos(theta));
        out.p[as<usize>(k - 1)] = c64(1.0, 0.0) / p1;
    }
    for (i32 k = 1; k <= n; ++k) {
        const f64 theta = kPi * (2.0 * as<f64>(k) - 1.0) / (2.0 * as<f64>(n));
        const f64 c = keel::cos(theta);
        if (keel::abs(c) < 1e-12) continue;
        z.push_back(c64(0.0, 1.0 / c));
    }
    out.z = order_roots_canonical(keel::move(z), false, false, false, alloc);

    c64 num(1.0, 0.0);
    c64 den(1.0, 0.0);
    for (usize i = 0; i < out.p.size(); ++i) num *= -out.p[i];
    for (usize i = 0; i < out.z.size(); ++i) den *= -out.z[i];
    out.k = (keel::abs(den) > 0.0) ? (num / den).real() : num.real();
    return SignalStatus::kOk;
}

SignalStatus besselap(i32 n, ZpkResult& out, Allocator* alloc) {
    static const c64 kB1[1] = {c64(-1.0, 0.0)};
    static const c64 kB2[2] = {c64(-0.866025403784438, 0.5), c64(-0.866025403784438, -0.5)};
    static const c64 kB3[3] = {
        c64(-0.745640385848077, 0.711366624972835),
        c64(-0.745640385848077, -0.711366624972835),
        c64(-0.941600026533207, 0.0),
    };
    static const c64 kB4[4] = {
        c64(-0.657211171671882, 0.830161435843259),
        c64(-0.657211171671882, -0.830161435843259),
        c64(-0.904758796788245, 0.270918733003839),
        c64(-0.904758796788245, -0.270918733003839),
    };
    static const c64 kB5[5] = {
        c64(-0.590575944611919, 0.907206756409664),
        c64(-0.590575944611919, -0.907206756409664),
        c64(-0.851553619368782, 0.442717463809432),
        c64(-0.851553619368782, -0.442717463809432),
        c64(-0.926174717377150, 0.0),
    };
    static const c64 kB6[6] = {
        c64(-0.538552681578156, 0.961864022747488),
        c64(-0.538552681578156, -0.961864022747488),
        c64(-0.799654185832948, 0.562171734879845),
        c64(-0.799654185832948, -0.562171734879845),
        c64(-0.909390683404036, 0.185696439038819),
        c64(-0.909390683404036, -0.185696439038819),
    };
    static const c64 kB7[7] = {
        c64(-0.496691725667232, 1.002508508454420),
        c64(-0.496691725667232, -1.002508508454420),
        c64(-0.752735543409167, 0.650469630987972),
        c64(-0.752735543409167, -0.650469630987972),
        c64(-0.880253434201683, 0.321665276230762),
        c64(-0.880253434201683, -0.321665276230762),
        c64(-0.919487155649029, 0.0),
    };
    static const c64 kB8[8] = {
        c64(-0.462174041253212, 1.034388681126410),
        c64(-0.462174041253212, -1.034388681126410),
        c64(-0.711138180848463, 0.718651731365058),
        c64(-0.711138180848463, -0.718651731365058),
        c64(-0.847396946514931, 0.425901753827867),
        c64(-0.847396946514931, -0.425901753827867),
        c64(-0.909286970144471, 0.141238678192833),
        c64(-0.909286970144471, -0.141238678192833),
    };
    static const ArrayRef<c64> kTables[9] = {
        {}, ArrayRef<c64>(kB1, 1), ArrayRef<c64>(kB2, 2), ArrayRef<c64>(kB3, 3),
        ArrayRef<c64>(kB4, 4), ArrayRef<c64>(kB5, 5), ArrayRef<c64>(kB6, 6),
        ArrayRef<c64>(kB7, 7), ArrayRef<c64>(kB8, 8),
    };
    if (n < 0 || n > 25) {
        return SignalStatus::kInvalidArgument;
    }
    out.z = Array<c64>(alloc);
    if (n <= 8) {
        out.p = Array<c64>::from(kTables[n], alloc);
        out.k = 1.0;
        return SignalStatus::kOk;
    }
    // Reverse Bessel polynomial recurrence B_k(s) = (2k-1)*B_{k-1}(s) + s^2 * B_{k-2}(s)
    // in ascending powers of s, then roots scaled so prod(-p_i) = 1 (MATLAB besselap convention).
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> bm2(1, 1.0, scratch.allocator());
    Array<f64> bm1(2, 1.0, scratch.allocator());
    for (i32 k = 2; k <= n; ++k) {
        Array<f64> bk(as<usize>(k + 1), 0.0, scratch.allocator());
        const f64 c = as<f64>(2 * k - 1);
        for (usize i = 0; i < bm1.size(); ++i) bk[i] += c * bm1[i];
        for (usize i = 0; i < bm2.size(); ++i) bk[i + 2] += bm2[i];
        bm2 = keel::move(bm1);
        bm1 = keel::move(bk);
    }
    Array<f64> desc(as<usize>(n + 1), 0.0, scratch.allocator());
    for (usize i = 0; i <= as<usize>(n); ++i) {
        desc[i] = bm1[as<usize>(n) - i];
    }
    Array<c64> poles = keel::poly::roots(desc.as_ref(), scratch.allocator());
    const f64 scale = keel::pow(bm1[0], -1.0 / as<f64>(n));
    for (usize i = 0; i < poles.size(); ++i) {
        poles[i] *= scale;
    }
    out.p = order_roots_canonical(keel::move(poles), false, true, false, alloc);
    out.k = 1.0;
    return SignalStatus::kOk;
}

SignalStatus ellipap(i32 n, f64 rp_db, f64 rs_db, ZpkResult& out, Allocator* alloc) {
    if (n < 0 || !keel::is_finite(rp_db) || !keel::is_finite(rs_db) ||
        !(rp_db > 0.0) || !(rs_db > 0.0) || !(rs_db > rp_db)) {
        return SignalStatus::kInvalidArgument;
    }
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(alloc);
    if (n == 0) {
        out.k = keel::pow(10.0, -rp_db / 20.0);
        return SignalStatus::kOk;
    }

    const f64 eps2 = keel::pow(10.0, rp_db / 10.0) - 1.0;
    const f64 eps = keel::sqrt(eps2);
    const f64 astop = keel::pow(10.0, rs_db / 10.0) - 1.0;
    const f64 k1 = eps / keel::sqrt(astop);
    const f64 k = ellipdeg(n, k1);
    const f64 cap_k = ellip_km(k * k);
    const f64 k1p = keel::sqrt(1.0 - k1 * k1);
    const f64 cap_k1 = ellip_km(k1 * k1);
    const f64 v0 = asn_real(1.0 / keel::sqrt(1.0 + eps2), k1p) * cap_k / (as<f64>(n) * cap_k1);

    const i32 l = n / 2;
    Array<c64> zeros(alloc);
    Array<c64> poles(alloc);
    zeros.reserve(as<usize>(2 * l));
    poles.reserve(as<usize>(n));

    for (i32 i = 1; i <= l; ++i) {
        const f64 ui_abs = (2.0 * as<f64>(i) - 1.0) / as<f64>(n) * cap_k;
        const SnCnDn zv = ellipj_real(ui_abs, k);
        const f64 zeta = zv.cn / zv.dn;
        const f64 zim = 1.0 / (k * zeta);
        zeros.push_back(c64(0.0, zim));
        zeros.push_back(c64(0.0, -zim));

        const c64 cd_p = cd_complex(c64(ui_abs, -v0), k);
        const c64 pole = c64(0.0, 1.0) * cd_p;
        poles.push_back(pole);
        poles.push_back(keel::conj(pole));
    }
    if ((n % 2) == 1) {
        const f64 kp = keel::sqrt(1.0 - k * k);
        const SnCnDn vv = ellipj_real(v0, kp);
        poles.push_back(c64(-vv.sn / vv.cn, 0.0));
    }

    out.z = order_roots_canonical(keel::move(zeros), false, true, true, alloc);
    out.p = order_roots_canonical(keel::move(poles), true, true, true, alloc);

    c64 num(1.0, 0.0);
    c64 den(1.0, 0.0);
    for (usize i = 0; i < out.p.size(); ++i) num *= -out.p[i];
    for (usize i = 0; i < out.z.size(); ++i) den *= -out.z[i];
    f64 gain = (keel::abs(den) > 0.0) ? (num / den).real() : num.real();
    if ((n % 2) == 0) {
        gain /= keel::sqrt(1.0 + eps2);
    }
    out.k = gain;
    return SignalStatus::kOk;
}

// ============================================================================
// Frequency Transformations
// ============================================================================

SignalStatus lp2lp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, ZpkResult& out,
                   Allocator* alloc) {
    out.z = Array<c64>(z.size(), c64(0.0, 0.0), alloc);
    out.p = Array<c64>(p.size(), c64(0.0, 0.0), alloc);
    for (usize i = 0; i < z.size(); ++i) out.z[i] = z[i] * wo;
    for (usize i = 0; i < p.size(); ++i) out.p[i] = p[i] * wo;
    const i32 deg_diff = as<i32>(p.size()) - as<i32>(z.size());
    out.k = k * keel::pow(wo, as<f64>(deg_diff));
    return SignalStatus::kOk;
}

SignalStatus lp2hp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, ZpkResult& out,
                   Allocator* alloc) {
    const i32 m = as<i32>(z.size());
    const i32 n = as<i32>(p.size());
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(p.size(), c64(0.0, 0.0), alloc);
    out.z.reserve(keel::max(z.size(), p.size()));

    c64 pz(1.0, 0.0);
    c64 pp(1.0, 0.0);
    for (usize i = 0; i < z.size(); ++i) {
        pz *= -z[i];
        out.z.push_back(c64(wo, 0.0) / z[i]);
    }
    for (usize i = 0; i < p.size(); ++i) {
        pp *= -p[i];
        out.p[i] = c64(wo, 0.0) / p[i];
    }
    for (i32 i = 0; i < n - m; ++i) {
        out.z.push_back(c64(0.0, 0.0));
    }
    out.k = k * (pz / pp).real();
    return SignalStatus::kOk;
}

SignalStatus lp2bp(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, f64 bw, ZpkResult& out,
                   Allocator* alloc) {
    const i32 m = as<i32>(z.size());
    const i32 n = as<i32>(p.size());
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(alloc);
    out.z.reserve(2 * z.size() + as<usize>(keel::max(0, n - m)));
    out.p.reserve(2 * p.size());

    auto push_pair = [&](c64 x, Array<c64>& dst) {
        const c64 b = bw * x;
        const c64 disc = keel::sqrt(b * b - 4.0 * wo * wo);
        dst.push_back((b + disc) * 0.5);
        dst.push_back((b - disc) * 0.5);
    };
    for (usize i = 0; i < z.size(); ++i) push_pair(z[i], out.z);
    for (usize i = 0; i < p.size(); ++i) push_pair(p[i], out.p);
    for (i32 i = 0; i < n - m; ++i) out.z.push_back(c64(0.0, 0.0));
    out.k = k * keel::pow(bw, as<f64>(n - m));
    return SignalStatus::kOk;
}

SignalStatus lp2bs(ArrayRef<c64> z, ArrayRef<c64> p, f64 k, f64 wo, f64 bw, ZpkResult& out,
                   Allocator* alloc) {
    const i32 m = as<i32>(z.size());
    const i32 n = as<i32>(p.size());
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(alloc);
    out.z.reserve(2 * z.size() + 2 * as<usize>(keel::max(0, n - m)));
    out.p.reserve(2 * p.size());

    auto push_pair = [&](c64 x, Array<c64>& dst) {
        const c64 a = x;
        const c64 b(-bw, 0.0);
        const c64 c = x * (wo * wo);
        const c64 disc = keel::sqrt(b * b - 4.0 * a * c);
        dst.push_back((-b + disc) / (2.0 * a));
        dst.push_back((-b - disc) / (2.0 * a));
    };
    c64 pz(1.0, 0.0);
    c64 pp(1.0, 0.0);
    for (usize i = 0; i < z.size(); ++i) {
        pz *= -z[i];
        push_pair(z[i], out.z);
    }
    for (usize i = 0; i < p.size(); ++i) {
        pp *= -p[i];
        push_pair(p[i], out.p);
    }
    for (i32 i = 0; i < n - m; ++i) {
        out.z.push_back(c64(0.0, wo));
        out.z.push_back(c64(0.0, -wo));
    }
    out.k = k * (pz / pp).real();
    return SignalStatus::kOk;
}

// ============================================================================
// Bilinear, Impinvar & Freqs
// ============================================================================

SignalStatus bilinear(ArrayRef<f64> b, ArrayRef<f64> a, f64 fs, f64 fp, TfResult& out,
                      Allocator* alloc) {
    if (!keel::is_finite(fs) || !(fs > 0.0) || !keel::is_finite(fp) || fp < 0.0 ||
        fp >= 0.5 * fs) {
        return SignalStatus::kInvalidArgument;
    }
    if (b.empty() || a.empty()) {
        out.b = Array<f64>(alloc);
        out.a = Array<f64>(alloc);
        return SignalStatus::kOk;
    }
    const i32 nb = as<i32>(b.size()) - 1;
    const i32 na = as<i32>(a.size()) - 1;

    f64 fs_eff = fs;
    if (fp > 0.0) {
        const f64 wp = kTau * fp;
        fs_eff = wp / (2.0 * keel::tan(wp / (2.0 * fs)));
    }
    const f64 k_scale = 2.0 * fs_eff;
    const i32 m = keel::max(na, nb);

    auto poly_pow = [alloc](f64 alpha, i32 p) {
        Array<f64> r(1, 1.0, alloc);
        for (i32 i = 0; i < p; ++i) {
            Array<f64> nr(r.size() + 1, 0.0, alloc);
            for (usize j = 0; j < r.size(); ++j) {
                nr[j] += r[j];
                nr[j + 1] += alpha * r[j];
            }
            r = keel::move(nr);
        }
        return r;
    };

    auto subst = [&](ArrayRef<f64> c, i32 deg, i32 order) {
        Array<f64> r(as<usize>(order + 1), 0.0, alloc);
        Array<f64> kpow(as<usize>(deg + 1), 1.0, alloc);
        for (i32 i = 1; i <= deg; ++i) {
            kpow[as<usize>(i)] = kpow[as<usize>(i - 1)] * k_scale;
        }
        for (i32 i = 0; i <= deg; ++i) {
            const Array<f64> pa = poly_pow(-1.0, i);
            const Array<f64> pb = poly_pow(1.0, order - i);
            Array<f64> cv(pa.size() + pb.size() - 1, 0.0, alloc);
            for (usize j = 0; j < pa.size(); ++j) {
                for (usize k = 0; k < pb.size(); ++k) {
                    cv[j + k] += pa[j] * pb[k];
                }
            }
            const f64 sc = c[as<usize>(deg - i)] * kpow[as<usize>(i)];
            for (usize j = 0; j < cv.size(); ++j) {
                r[j] += sc * cv[j];
            }
        }
        return r;
    };

    out.b = subst(b, nb, m);
    out.a = subst(a, na, m);
    const f64 a0 = out.a.empty() ? 1.0 : out.a[0];
    if (keel::abs(a0) > 1e-300) {
        for (usize i = 0; i < out.b.size(); ++i) out.b[i] /= a0;
        for (usize i = 0; i < out.a.size(); ++i) out.a[i] /= a0;
    }
    return SignalStatus::kOk;
}

SignalStatus impinvar(ArrayRef<f64> b, ArrayRef<f64> a, f64 fs, TfResult& out, Allocator* alloc) {
    if (a.size() < 2 || b.size() >= a.size() || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    const i32 n = as<i32>(a.size()) - 1;
    const f64 t_step = 1.0 / fs;

    const Array<c64> pv = keel::poly::roots(a, alloc);
    if (as<i32>(pv.size()) != n) {
        return SignalStatus::kInvalidArgument;
    }

    constexpr f64 kPtol = 1e-3;
    auto same_pole = [&](c64 q, c64 p) noexcept {
        return keel::abs(q - p) <= kPtol * keel::max(1.0, keel::abs(p));
    };
    Array<c64> distinct(alloc);
    Array<i32> mult(alloc);
    {
        Array<u8> used(as<usize>(n), u8{0}, alloc);
        for (i32 k = 0; k < n; ++k) {
            if (used[as<usize>(k)] != 0) continue;
            c64 sum = pv[as<usize>(k)];
            i32 m = 1;
            used[as<usize>(k)] = 1;
            for (i32 j = k + 1; j < n; ++j) {
                if (used[as<usize>(j)] == 0 && same_pole(pv[as<usize>(j)], pv[as<usize>(k)])) {
                    used[as<usize>(j)] = 1;
                    sum += pv[as<usize>(j)];
                    ++m;
                }
            }
            c64 p = sum / as<f64>(m);
            if (m >= 2) {
                for (i32 it = 0; it < 12; ++it) {
                    const Array<c64> tv = taylor_about(a, p, alloc);
                    if (as<usize>(m) >= tv.size()) break;
                    const c64 den = as<f64>(m) * tv[as<usize>(m)];
                    if (keel::abs(den) < 1e-300) break;
                    const c64 step = tv[as<usize>(m - 1)] / den;
                    p -= step;
                    if (keel::abs(step) <= 1e-15 * keel::max(1.0, keel::abs(p))) break;
                }
            }
            distinct.push_back(p);
            mult.push_back(m);
        }
    }

    i32 max_m = 1;
    for (usize i = 0; i < mult.size(); ++i) {
        if (mult[i] > max_m) max_m = mult[i];
    }

    Array<c64> alpha(alloc);
    alpha.reserve(as<usize>(n));
    for (usize di = 0; di < distinct.size(); ++di) {
        const c64 a_alpha = keel::exp(distinct[di] * t_step);
        for (i32 t = 0; t < mult[di]; ++t) {
            alpha.push_back(a_alpha);
        }
    }
    Array<c64> ad(1, c64(1.0, 0.0), alloc);
    for (usize i = 0; i < alpha.size(); ++i) {
        ad = poly_mul_lin_c64(ad.as_ref(), alpha[i], alloc);
    }

    // Eulerian numerators N_m(w).
    Array<Array<c64>> npoly(alloc);
    npoly.reserve(as<usize>(max_m + 1));
    for (i32 m = 0; m <= max_m; ++m) {
        npoly.push_back(Array<c64>(alloc));
    }
    npoly[1].push_back(c64(1.0, 0.0));
    for (i32 m = 1; m < max_m; ++m) {
        const Array<c64>& nm = npoly[as<usize>(m)];
        Array<c64> s(nm.size() + 1, c64(0.0, 0.0), alloc);
        for (usize i = 1; i < nm.size(); ++i) {
            const c64 d = as<f64>(i) * nm[i];
            s[i - 1] += d;
            s[i] -= d;
        }
        for (usize i = 0; i < nm.size(); ++i) {
            s[i] += as<f64>(m) * nm[i];
        }
        Array<c64> nm1(s.size() + 1, c64(0.0, 0.0), alloc);
        for (usize i = 0; i < s.size(); ++i) {
            nm1[i + 1] = s[i];
        }
        npoly[as<usize>(m + 1)] = keel::move(nm1);
    }

    Array<c64> bd(as<usize>(n), c64(0.0, 0.0), alloc);
    for (usize di = 0; di < distinct.size(); ++di) {
        const c64 p = distinct[di];
        const i32 m_ord = mult[di];
        const c64 a_alpha = keel::exp(p * t_step);

        const Array<c64> ta = taylor_about(a, p, alloc);
        const Array<c64> tb = taylor_about(b, p, alloc);
        auto atil = [&](i32 idx) noexcept -> c64 {
            const usize k = as<usize>(m_ord + idx);
            return (k < ta.size()) ? ta[k] : c64(0.0, 0.0);
        };
        Array<c64> cc(as<usize>(m_ord), c64(0.0, 0.0), alloc);
        for (i32 j = 0; j < m_ord; ++j) {
            c64 acc = (as<usize>(j) < tb.size()) ? tb[as<usize>(j)] : c64(0.0, 0.0);
            for (i32 i = 1; i <= j; ++i) {
                acc -= atil(i) * cc[as<usize>(j - i)];
            }
            cc[as<usize>(j)] = acc / atil(0);
        }

        Array<c64> cp(1, c64(1.0, 0.0), alloc);
        for (usize dj = 0; dj < distinct.size(); ++dj) {
            if (dj == di) continue;
            const c64 aj = keel::exp(distinct[dj] * t_step);
            for (i32 t = 0; t < mult[dj]; ++t) {
                cp = poly_mul_lin_c64(cp.as_ref(), aj, alloc);
            }
        }

        for (i32 m = 1; m <= m_ord; ++m) {
            const c64 rm = cc[as<usize>(m_ord - m)];
            f64 fact = 1.0;
            for (i32 q = 2; q < m; ++q) fact *= as<f64>(q);
            const c64 sc = rm * keel::pow(t_step, as<f64>(m)) / fact;

            const Array<c64>& nm = npoly[as<usize>(m)];
            Array<c64> nz(nm.size(), c64(0.0, 0.0), alloc);
            c64 apw(1.0, 0.0);
            for (usize i = 0; i < nm.size(); ++i) {
                nz[i] = nm[i] * apw;
                apw *= a_alpha;
            }
            Array<c64> pw(1, c64(1.0, 0.0), alloc);
            for (i32 t = 0; t < m_ord - m; ++t) {
                pw = poly_mul_lin_c64(pw.as_ref(), a_alpha, alloc);
            }
            const Array<c64> nz_pw = poly_mul_c64(nz.as_ref(), pw.as_ref(), alloc);
            const Array<c64> contrib = poly_mul_c64(nz_pw.as_ref(), cp.as_ref(), alloc);
            for (usize i = 0; i < contrib.size() && i < bd.size(); ++i) {
                bd[i] += sc * contrib[i];
            }
        }
    }

    out.a = Array<f64>(as<usize>(n + 1), 0.0, alloc);
    out.b = Array<f64>(as<usize>(n), 0.0, alloc);
    for (usize i = 0; i <= as<usize>(n); ++i) out.a[i] = ad[i].real();
    for (usize i = 0; i < as<usize>(n); ++i) out.b[i] = bd[i].real();
    return SignalStatus::kOk;
}

SignalStatus freqs(ArrayRef<f64> b, ArrayRef<f64> a, ArrayRef<f64> w,
                   MutArrayRef<c64> h_out) noexcept {
    if (h_out.size() < w.size()) {
        return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < w.size(); ++i) {
        const c64 s(0.0, w[i]);
        c64 num(0.0, 0.0);
        c64 den(0.0, 0.0);
        for (usize k = 0; k < b.size(); ++k) num = num * s + b[k];
        for (usize k = 0; k < a.size(); ++k) den = den * s + a[k];
        h_out[i] = (keel::abs(den) > 0.0) ? (num / den) : c64(0.0, 0.0);
    }
    return SignalStatus::kOk;
}

// ============================================================================
// Top-Level IIR Designs
// ============================================================================

SignalStatus butter(i32 n, ArrayRef<f64> wn, FilterType ftype, bool analog, TfResult& out,
                    Allocator* alloc) {
    if (n <= 0) return SignalStatus::kInvalidArgument;
    ZpkResult proto;
    const SignalStatus st = buttap(n, proto, alloc);
    if (st != SignalStatus::kOk) return st;
    return finish_iir_design(proto, wn, ftype, analog, out, alloc);
}

SignalStatus cheby1(i32 n, f64 rp_db, ArrayRef<f64> wn, FilterType ftype, bool analog,
                    TfResult& out, Allocator* alloc) {
    if (n <= 0) return SignalStatus::kInvalidArgument;
    ZpkResult proto;
    const SignalStatus st = cheb1ap(n, rp_db, proto, alloc);
    if (st != SignalStatus::kOk) return st;
    return finish_iir_design(proto, wn, ftype, analog, out, alloc);
}

SignalStatus cheby2(i32 n, f64 rs_db, ArrayRef<f64> wn, FilterType ftype, bool analog,
                    TfResult& out, Allocator* alloc) {
    if (n <= 0) return SignalStatus::kInvalidArgument;
    ZpkResult proto;
    const SignalStatus st = cheb2ap(n, rs_db, proto, alloc);
    if (st != SignalStatus::kOk) return st;
    return finish_iir_design(proto, wn, ftype, analog, out, alloc);
}

SignalStatus ellip(i32 n, f64 rp_db, f64 rs_db, ArrayRef<f64> wn, FilterType ftype, bool analog,
                   TfResult& out, Allocator* alloc) {
    if (n <= 0) return SignalStatus::kInvalidArgument;
    ZpkResult proto;
    const SignalStatus st = ellipap(n, rp_db, rs_db, proto, alloc);
    if (st != SignalStatus::kOk) return st;
    return finish_iir_design(proto, wn, ftype, analog, out, alloc);
}

SignalStatus besself(i32 n, ArrayRef<f64> wn, FilterType ftype, bool analog, TfResult& out,
                     Allocator* alloc) {
    if (n <= 0) return SignalStatus::kInvalidArgument;
    ZpkResult proto;
    const SignalStatus st = besselap(n, proto, alloc);
    if (st != SignalStatus::kOk) return st;
    return finish_iir_design(proto, wn, ftype, analog, out, alloc);
}

// ============================================================================
// Order Estimators
// ============================================================================

SignalStatus buttord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db, bool analog,
                     OrderResult& out) noexcept {
    if (!keel::is_finite(rp_db) || !keel::is_finite(rs_db) || !(rp_db > 0.0) || !(rs_db > rp_db)) {
        return SignalStatus::kInvalidArgument;
    }
    OrdNorm norm;
    const SignalStatus st = normalise_ord(wp, ws, analog, norm);
    if (st != SignalStatus::kOk) return st;
    const f64 gs_lin = keel::pow(10.0, rs_db / 10.0) - 1.0;
    const f64 gp_lin = keel::pow(10.0, rp_db / 10.0) - 1.0;
    if (!(gp_lin > 0.0) || !(gs_lin > gp_lin)) return SignalStatus::kInvalidArgument;

    const i32 ord =
        keel::max<i32>(1, as<i32>(keel::ceil(0.5 * keel::log10(gs_lin / gp_lin) / keel::log10(norm.ratio))));
    const f64 r = keel::pow(gs_lin, 1.0 / (2.0 * as<f64>(ord)));
    out.order = ord;
    if (norm.ftype == FilterType::kLowpass || norm.ftype == FilterType::kHighpass) {
        const f64 wn_a = (norm.ftype == FilterType::kLowpass) ? (norm.wo_stopband / r)
                                                              : (norm.wo_stopband * r);
        out.wn[0] = dewarp_wp(wn_a, analog);
        out.wn_count = 1;
    } else {
        const f64 w0_norm = norm.ratio / r;
        const f64 wo2 = norm.wo_low * norm.wo_high;
        const f64 bw = norm.wo_high - norm.wo_low;
        const f64 bw_eff = (norm.ftype == FilterType::kBandpass) ? (w0_norm * bw) : (bw / w0_norm);
        const f64 w2 = 0.5 * (bw_eff + keel::sqrt(bw_eff * bw_eff + 4.0 * wo2));
        const f64 w1 = wo2 / w2;
        out.wn[0] = dewarp_wp(w1, analog);
        out.wn[1] = dewarp_wp(w2, analog);
        out.wn_count = 2;
    }
    return SignalStatus::kOk;
}

SignalStatus cheb1ord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db, bool analog,
                      OrderResult& out) noexcept {
    if (!keel::is_finite(rp_db) || !keel::is_finite(rs_db) || !(rp_db > 0.0) || !(rs_db > rp_db)) {
        return SignalStatus::kInvalidArgument;
    }
    OrdNorm norm;
    const SignalStatus st = normalise_ord(wp, ws, analog, norm);
    if (st != SignalStatus::kOk) return st;
    const f64 gs_lin = keel::pow(10.0, rs_db / 10.0) - 1.0;
    const f64 gp_lin = keel::pow(10.0, rp_db / 10.0) - 1.0;
    if (!(gp_lin > 0.0) || !(gs_lin > gp_lin)) return SignalStatus::kInvalidArgument;

    out.order =
        keel::max<i32>(1, as<i32>(keel::ceil(keel::acosh(keel::sqrt(gs_lin / gp_lin)) / keel::acosh(norm.ratio))));
    out.wn_count = wp.size();
    for (usize i = 0; i < wp.size(); ++i) out.wn[i] = wp[i];
    return SignalStatus::kOk;
}

SignalStatus cheb2ord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db, bool analog,
                      OrderResult& out) noexcept {
    if (!keel::is_finite(rp_db) || !keel::is_finite(rs_db) || !(rp_db > 0.0) || !(rs_db > rp_db)) {
        return SignalStatus::kInvalidArgument;
    }
    OrdNorm norm;
    const SignalStatus st = normalise_ord(wp, ws, analog, norm);
    if (st != SignalStatus::kOk) return st;
    const f64 gs_lin = keel::pow(10.0, rs_db / 10.0) - 1.0;
    const f64 gp_lin = keel::pow(10.0, rp_db / 10.0) - 1.0;
    if (!(gp_lin > 0.0) || !(gs_lin > gp_lin)) return SignalStatus::kInvalidArgument;

    out.order =
        keel::max<i32>(1, as<i32>(keel::ceil(keel::acosh(keel::sqrt(gs_lin / gp_lin)) / keel::acosh(norm.ratio))));
    out.wn_count = ws.size();
    for (usize i = 0; i < ws.size(); ++i) out.wn[i] = ws[i];
    return SignalStatus::kOk;
}

SignalStatus ellipord(ArrayRef<f64> wp, ArrayRef<f64> ws, f64 rp_db, f64 rs_db, bool analog,
                      OrderResult& out) noexcept {
    if (!keel::is_finite(rp_db) || !keel::is_finite(rs_db) || !(rp_db > 0.0) || !(rs_db > rp_db) ||
        wp.size() != ws.size() || (wp.size() != 1 && wp.size() != 2)) {
        return SignalStatus::kInvalidArgument;
    }
    OrdNorm norm;
    const SignalStatus st = normalise_ord(wp, ws, analog, norm, /*tighten_bs=*/false);
    if (st != SignalStatus::kOk) return st;
    const usize num_w = wp.size();
    const f64 wa_min = norm.ratio;

    const f64 eps = keel::sqrt(keel::pow(10.0, 0.1 * rp_db) - 1.0);
    const f64 k1 = eps / keel::sqrt(keel::pow(10.0, 0.1 * rs_db) - 1.0);
    const f64 k = 1.0 / wa_min;

    const f64 k0 = ellip_km(k * k);
    const f64 k1c = ellip_km(1.0 - k * k);
    const f64 k1k1 = ellip_km(k1 * k1);
    const f64 k1k1c = ellip_km(1.0 - k1 * k1);

    out.order = keel::max<i32>(1, as<i32>(keel::ceil(k0 * k1k1c / (k1c * k1k1))));
    out.wn_count = num_w;
    for (usize i = 0; i < num_w; ++i) {
        out.wn[i] = wp[i];
    }
    return SignalStatus::kOk;
}

SignalStatus firpmord(ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> dev, f64 fs,
                      FirpmOrdResult& out, Allocator* alloc) {
    const usize mf = f.size();
    const usize nbands = a.size();
    if (nbands < 2 || nbands != dev.size() || mf != 2 * (nbands - 1) ||
        !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> fcuts(mf, 0.0, alloc);
    Array<f64> devs(nbands, 0.0, alloc);
    for (usize i = 0; i < mf; ++i) {
        if (!keel::is_finite(f[i]) || f[i] < 0.0) return SignalStatus::kInvalidArgument;
        fcuts[i] = f[i] / fs;
        if (fcuts[i] > 0.5) return SignalStatus::kInvalidArgument;
        if (i > 0 && !(fcuts[i] > fcuts[i - 1])) return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nbands; ++i) {
        if (!keel::is_finite(a[i]) || !keel::is_finite(dev[i]) || !(dev[i] > 0.0)) {
            return SignalStatus::kInvalidArgument;
        }
        const f64 base = ((a[i] == 0.0) ? 1.0 : 0.0) + a[i];
        devs[i] = (base != 0.0) ? (dev[i] / base) : dev[i];
    }

    const usize n_trans = mf / 2;
    usize n_min = 0;
    f64 min_w = keel::abs(fcuts[1] - fcuts[0]);
    for (usize i = 1; i < n_trans; ++i) {
        const f64 w = keel::abs(fcuts[2 * i + 1] - fcuts[2 * i]);
        if (w < min_w) {
            min_w = w;
            n_min = i;
        }
    }

    f64 l = 0.0;
    if (nbands == 2) {
        l = remlpord(fcuts[2 * n_min], fcuts[2 * n_min + 1], devs[0], devs[1]);
    } else {
        for (usize i = 1; i + 1 < nbands; ++i) {
            const f64 l1 = remlpord(fcuts[2 * (i - 1)], fcuts[2 * (i - 1) + 1], devs[i], devs[i - 1]);
            const f64 l2 = remlpord(fcuts[2 * i], fcuts[2 * i + 1], devs[i], devs[i + 1]);
            if (l1 > l) l = l1;
            if (l2 > l) l = l2;
        }
    }

    i32 ord = as<i32>(keel::ceil(l)) - 1;
    if (a[nbands - 1] != 0.0 && (ord % 2) != 0) {
        ++ord;
    }
    out.order = ord;

    out.f = Array<f64>(mf + 2, 0.0, alloc);
    out.f[0] = 0.0;
    for (usize i = 0; i < mf; ++i) out.f[i + 1] = 2.0 * fcuts[i];
    out.f[mf + 1] = 1.0;

    out.a = Array<f64>(2 * nbands, 0.0, alloc);
    for (usize k = 0; k < nbands; ++k) {
        out.a[2 * k] = a[k];
        out.a[2 * k + 1] = a[k];
    }

    out.w = Array<f64>(nbands, 0.0, alloc);
    f64 max_dev = devs[0];
    for (usize i = 1; i < nbands; ++i) {
        if (devs[i] > max_dev) max_dev = devs[i];
    }
    for (usize i = 0; i < nbands; ++i) {
        out.w[i] = (devs[i] != 0.0) ? (max_dev / devs[i]) : 0.0;
    }
    return SignalStatus::kOk;
}

SignalStatus kaiserord(ArrayRef<f64> f, ArrayRef<f64> a, ArrayRef<f64> dev, f64 fs,
                       KaiserOrdResult& out, Allocator* alloc) {
    const usize mf = f.size();
    const usize nbands = a.size();
    if (nbands < 2 || nbands != dev.size() || mf != 2 * (nbands - 1) ||
        !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> fcuts(mf, 0.0, alloc);
    Array<f64> devs(nbands, 0.0, alloc);
    for (usize i = 0; i < mf; ++i) {
        if (!keel::is_finite(f[i]) || f[i] < 0.0) return SignalStatus::kInvalidArgument;
        fcuts[i] = f[i] / fs;
        if (fcuts[i] >= 0.5) return SignalStatus::kInvalidArgument;
        if (i > 0 && !(fcuts[i] > fcuts[i - 1])) return SignalStatus::kInvalidArgument;
    }
    for (usize i = 0; i < nbands; ++i) {
        if (!keel::is_finite(a[i]) || !keel::is_finite(dev[i]) || !(dev[i] > 0.0)) {
            return SignalStatus::kInvalidArgument;
        }
        const f64 base = ((a[i] == 0.0) ? 1.0 : 0.0) + a[i];
        devs[i] = (base != 0.0) ? (dev[i] / base) : dev[i];
    }

    const usize n_trans = mf / 2;
    usize n_min = 0;
    f64 min_w = keel::abs(fcuts[1] - fcuts[0]);
    for (usize i = 1; i < n_trans; ++i) {
        const f64 w = keel::abs(fcuts[2 * i + 1] - fcuts[2 * i]);
        if (w < min_w) {
            min_w = w;
            n_min = i;
        }
    }

    f64 l = 0.0;
    f64 beta = 0.0;
    if (nbands == 2) {
        const KaisLpRes r = kaislpord(fcuts[2 * n_min], fcuts[2 * n_min + 1], devs[0], devs[1]);
        l = r.l;
        beta = r.beta;
    } else {
        for (usize i = 1; i + 1 < nbands; ++i) {
            const KaisLpRes r1 =
                kaislpord(fcuts[2 * (i - 1)], fcuts[2 * (i - 1) + 1], devs[i], devs[i - 1]);
            const KaisLpRes r2 =
                kaislpord(fcuts[2 * i], fcuts[2 * i + 1], devs[i], devs[i + 1]);
            if (r1.l > l) { l = r1.l; beta = r1.beta; }
            if (r2.l > l) { l = r2.l; beta = r2.beta; }
        }
    }

    i32 ord = as<i32>(keel::ceil(l)) - 1;
    if ((ord % 2) != 0 && a[nbands - 1] != 0.0) {
        ++ord;
    }

    out.order = ord;
    out.beta = beta;
    out.wn = Array<f64>(n_trans, 0.0, alloc);
    for (usize i = 0; i < n_trans; ++i) {
        out.wn[i] = fcuts[2 * i] + fcuts[2 * i + 1];
    }

    if (nbands == 2 && a[0] == 0.0) out.ftype = KaiserBandType::kHigh;
    else if (nbands == 3 && a[1] == 0.0) out.ftype = KaiserBandType::kStop;
    else if (nbands >= 3 && a[0] == 0.0) out.ftype = KaiserBandType::kDc0;
    else if (nbands >= 3 && a[0] == 1.0) out.ftype = KaiserBandType::kDc1;
    else out.ftype = KaiserBandType::kLow;
    return SignalStatus::kOk;
}

} // namespace keel::signal
