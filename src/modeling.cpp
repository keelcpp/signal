#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::signal {

namespace {

Array<f64> biased_autocorr(ArrayRef<f64> x, i32 max_lag, Allocator* alloc) {
    const usize n = x.size();
    Array<f64> r(as<usize>(max_lag + 1), 0.0, alloc);
    if (n == 0) return r;
    const f64 inv_n = 1.0 / as<f64>(n);
    for (i32 k = 0; k <= max_lag; ++k) {
        const usize uk = as<usize>(k);
        f64 s = 0.0;
        for (usize i = 0; i + uk < n; ++i) s += x[i] * x[i + uk];
        r[uk] = s * inv_n;
    }
    return r;
}

void levinson_core(ArrayRef<f64> r, i32 p, LevinsonResult& res, Allocator* alloc) {
    res.a = Array<f64>(as<usize>(p + 1), 0.0, alloc);
    res.k = Array<f64>(as<usize>(p), 0.0, alloc);
    res.a[0] = 1.0;
    if (p == 0) {
        res.e = r.empty() ? 0.0 : r[0];
        return;
    }
    f64 e = r[0];
    if (!keel::is_finite(e) || e <= 0.0) {
        res.e = 0.0;
        return;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> a_prev(as<usize>(p + 1), 0.0, scratch.allocator());
    Array<f64> a_new(as<usize>(p + 1), 0.0, scratch.allocator());
    a_prev[0] = 1.0;

    for (i32 i = 1; i <= p; ++i) {
        f64 num = r[as<usize>(i)];
        for (i32 j = 1; j < i; ++j) {
            num += a_prev[as<usize>(j)] * r[as<usize>(i - j)];
        }
        const f64 ki = (e != 0.0) ? (-num / e) : 0.0;
        res.k[as<usize>(i - 1)] = ki;

        a_new[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            a_new[as<usize>(j)] = a_prev[as<usize>(j)] + ki * a_prev[as<usize>(i - j)];
        }
        a_new[as<usize>(i)] = ki;
        for (i32 j = 0; j <= i; ++j) a_prev[as<usize>(j)] = a_new[as<usize>(j)];
        e *= (1.0 - ki * ki);
    }
    for (i32 j = 0; j <= p; ++j) res.a[as<usize>(j)] = a_prev[as<usize>(j)];
    res.e = e;
}

Array<f64> solve_linear_system(Array<f64>& a, Array<f64>& b, i32 p, Allocator* alloc) {
    const usize up = as<usize>(p);
    for (i32 k = 0; k < p; ++k) {
        const usize uk = as<usize>(k);
        i32 piv = k;
        f64 piv_val = keel::abs(a[uk * up + uk]);
        for (i32 i = k + 1; i < p; ++i) {
            const f64 v = keel::abs(a[as<usize>(i) * up + uk]);
            if (v > piv_val) {
                piv_val = v;
                piv = i;
            }
        }
        if (piv_val < 1e-300) continue;
        if (piv != k) {
            const usize upiv = as<usize>(piv);
            for (i32 j = k; j < p; ++j) {
                keel::swap(a[uk * up + as<usize>(j)], a[upiv * up + as<usize>(j)]);
            }
            keel::swap(b[uk], b[upiv]);
        }
        const f64 diag = a[uk * up + uk];
        for (i32 i = k + 1; i < p; ++i) {
            const usize ui = as<usize>(i);
            const f64 f = a[ui * up + uk] / diag;
            for (i32 j = k; j < p; ++j) {
                a[ui * up + as<usize>(j)] -= f * a[uk * up + as<usize>(j)];
            }
            b[ui] -= f * b[uk];
        }
    }
    Array<f64> x(up, 0.0, alloc);
    for (i32 i = p - 1; i >= 0; --i) {
        const usize ui = as<usize>(i);
        f64 s = b[ui];
        for (i32 j = i + 1; j < p; ++j) {
            s -= a[ui * up + as<usize>(j)] * x[as<usize>(j)];
        }
        const f64 diag = a[ui * up + ui];
        x[ui] = (keel::abs(diag) > 1e-300) ? (s / diag) : 0.0;
    }
    return x;
}

Array<f64> cov_matrix_forward(ArrayRef<f64> x, i32 p, Allocator* alloc) {
    const i32 n = as<i32>(x.size());
    const usize dim = as<usize>(p + 1);
    Array<f64> r(dim * dim, 0.0, alloc);
    for (i32 t = p; t < n; ++t) {
        for (i32 i = 0; i <= p; ++i) {
            const f64 xi = x[as<usize>(t - i)];
            for (i32 j = 0; j <= p; ++j) {
                r[as<usize>(i) * dim + as<usize>(j)] += xi * x[as<usize>(t - j)];
            }
        }
    }
    return r;
}

Array<f64> cov_matrix_forward_backward(ArrayRef<f64> x, i32 p, Allocator* alloc) {
    const i32 n = as<i32>(x.size());
    const usize dim = as<usize>(p + 1);
    Array<f64> r = cov_matrix_forward(x, p, alloc);
    for (i32 t = 0; t + p < n; ++t) {
        for (i32 i = 0; i <= p; ++i) {
            const f64 xi = x[as<usize>(t + i)];
            for (i32 j = 0; j <= p; ++j) {
                r[as<usize>(i) * dim + as<usize>(j)] += xi * x[as<usize>(t + j)];
            }
        }
    }
    return r;
}

void solve_ar_cov(ArrayRef<f64> r, i32 p, Array<f64>& a_out, f64& e_out, Allocator* alloc) {
    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize up = as<usize>(p);
    const usize dim = up + 1;
    Array<f64> a_mat(up * up, 0.0, scratch.allocator());
    Array<f64> b_vec(up, 0.0, scratch.allocator());
    for (i32 i = 0; i < p; ++i) {
        const usize ui = as<usize>(i);
        for (i32 j = 0; j < p; ++j) {
            a_mat[ui * up + as<usize>(j)] = r[(ui + 1) * dim + (as<usize>(j) + 1)];
        }
        b_vec[ui] = -r[(ui + 1) * dim + 0];
    }
    const Array<f64> a_tail = solve_linear_system(a_mat, b_vec, p, scratch.allocator());
    a_out = Array<f64>(dim, 0.0, alloc);
    a_out[0] = 1.0;
    for (i32 k = 0; k < p; ++k) a_out[as<usize>(k + 1)] = a_tail[as<usize>(k)];
    f64 e = r[0];
    for (i32 k = 1; k <= p; ++k) e += a_out[as<usize>(k)] * r[as<usize>(k)];
    e_out = e;
}

SignalStatus invfreq_lsq(ArrayRef<c64> h, ArrayRef<c64> x, i32 nb, i32 na, bool descending,
                         TfResult& out, Allocator* alloc) {
    if (nb < 0 || na < 0 || h.size() != x.size()) {
        return SignalStatus::kInvalidArgument;
    }
    const i32 k_cnt = as<i32>(h.size());
    const i32 nv = nb + 1 + na;
    if (k_cnt * 2 < nv) {
        return SignalStatus::kInvalidArgument;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const usize unv = as<usize>(nv);
    const usize rows = as<usize>(2 * k_cnt);
    Array<f64> ar(rows * unv, 0.0, scratch.allocator());
    Array<f64> rhs(rows, 0.0, scratch.allocator());

    const i32 m_deg = keel::max(nb, na);
    Array<c64> xpow(as<usize>(m_deg + 1), c64(0.0, 0.0), scratch.allocator());
    Array<c64> row(unv, c64(0.0, 0.0), scratch.allocator());

    for (i32 k = 0; k < k_cnt; ++k) {
        const usize uk = as<usize>(k);
        xpow[0] = c64(1.0, 0.0);
        for (i32 i = 1; i <= m_deg; ++i) {
            xpow[as<usize>(i)] = xpow[as<usize>(i - 1)] * x[uk];
        }
        c64 rhs_c = -h[uk];
        if (descending) {
            for (i32 i = 0; i <= nb; ++i) row[as<usize>(i)] = -xpow[as<usize>(nb - i)];
            for (i32 j = 1; j <= na; ++j) row[as<usize>(nb + j)] = h[uk] * xpow[as<usize>(na - j)];
            rhs_c = -h[uk] * xpow[as<usize>(na)];
        } else {
            for (i32 i = 0; i <= nb; ++i) row[as<usize>(i)] = -xpow[as<usize>(i)];
            for (i32 j = 1; j <= na; ++j) row[as<usize>(nb + j)] = h[uk] * xpow[as<usize>(j)];
        }

        for (i32 c = 0; c < nv; ++c) {
            ar[(2 * uk) * unv + as<usize>(c)] = row[as<usize>(c)].real();
            ar[(2 * uk + 1) * unv + as<usize>(c)] = row[as<usize>(c)].imag();
        }
        rhs[2 * uk] = rhs_c.real();
        rhs[2 * uk + 1] = rhs_c.imag();
    }

    Array<f64> ata(unv * unv, 0.0, scratch.allocator());
    Array<f64> atb(unv, 0.0, scratch.allocator());
    for (i32 i = 0; i < nv; ++i) {
        const usize ui = as<usize>(i);
        for (i32 j = 0; j < nv; ++j) {
            const usize uj = as<usize>(j);
            f64 s = 0.0;
            for (usize r = 0; r < rows; ++r) s += ar[r * unv + ui] * ar[r * unv + uj];
            ata[ui * unv + uj] = s;
        }
        f64 s = 0.0;
        for (usize r = 0; r < rows; ++r) s += ar[r * unv + ui] * rhs[r];
        atb[ui] = s;
    }

    const Array<f64> theta = solve_linear_system(ata, atb, nv, scratch.allocator());
    out.b = Array<f64>(as<usize>(nb + 1), 0.0, alloc);
    out.a = Array<f64>(as<usize>(na + 1), 0.0, alloc);
    for (i32 i = 0; i <= nb; ++i) out.b[as<usize>(i)] = theta[as<usize>(i)];
    out.a[0] = 1.0;
    for (i32 j = 1; j <= na; ++j) out.a[as<usize>(j)] = theta[as<usize>(nb + j)];
    return SignalStatus::kOk;
}

void jacobi_eig_sym(Array<f64>& a, Array<f64>& v, usize d) noexcept {
    for (usize i = 0; i < d * d; ++i) v[i] = 0.0;
    for (usize i = 0; i < d; ++i) v[i * d + i] = 1.0;
    for (i32 sweep = 0; sweep < 60; ++sweep) {
        f64 off = 0.0;
        for (usize p = 0; p < d; ++p) {
            for (usize q = p + 1; q < d; ++q) {
                off += a[p * d + q] * a[p * d + q];
            }
        }
        if (off < 1e-18) break;
        for (usize p = 0; p + 1 < d; ++p) {
            for (usize q = p + 1; q < d; ++q) {
                const f64 apq = a[p * d + q];
                if (keel::abs(apq) < 1e-300) continue;
                const f64 app = a[p * d + p];
                const f64 aqq = a[q * d + q];
                const f64 theta = (aqq - app) / (2.0 * apq);
                const f64 sgn = (theta >= 0.0) ? 1.0 : -1.0;
                const f64 t = (theta == 0.0)
                                  ? 1.0
                                  : (sgn / (keel::abs(theta) + keel::sqrt(theta * theta + 1.0)));
                const f64 c = 1.0 / keel::sqrt(t * t + 1.0);
                const f64 s = t * c;
                a[p * d + p] = app - t * apq;
                a[q * d + q] = aqq + t * apq;
                a[p * d + q] = 0.0;
                a[q * d + p] = 0.0;
                for (usize r = 0; r < d; ++r) {
                    if (r == p || r == q) continue;
                    const f64 arp = a[r * d + p];
                    const f64 arq = a[r * d + q];
                    a[r * d + p] = c * arp - s * arq;
                    a[p * d + r] = a[r * d + p];
                    a[r * d + q] = c * arq + s * arp;
                    a[q * d + r] = a[r * d + q];
                }
                for (usize r = 0; r < d; ++r) {
                    const f64 vrp = v[r * d + p];
                    const f64 vrq = v[r * d + q];
                    v[r * d + p] = c * vrp - s * vrq;
                    v[r * d + q] = c * vrq + s * vrp;
                }
            }
        }
    }
}

SignalStatus pseudospectrum_impl(ArrayRef<f64> x, i32 p, i32 nfft, f64 fs, bool eig_weight,
                                 MutArrayRef<f64> p_out, MutArrayRef<f64> f_out,
                                 Allocator* alloc) {
    if (x.empty() || !keel::is_finite(fs) || !(fs > 0.0)) {
        return SignalStatus::kInvalidArgument;
    }
    if (p < 1) p = 1;
    if (nfft < 2) nfft = 256;
    const usize nf = as<usize>(nfft / 2 + 1);
    if (p_out.size() < nf || (!f_out.empty() && f_out.size() < nf)) {
        return SignalStatus::kInvalidArgument;
    }

    const i32 m = 2 * p;
    const usize um = as<usize>(m);
    const usize rows = x.size() + as<usize>(m - 1);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> xc(rows * um, 0.0, scratch.allocator());
    const SignalStatus st = corrmtx(x, m - 1, xc.as_mut_ref());
    if (st != SignalStatus::kOk) return st;

    Array<f64> r_mat(um * um, 0.0, scratch.allocator());
    for (usize i = 0; i < um; ++i) {
        for (usize j = 0; j < um; ++j) {
            f64 s = 0.0;
            for (usize r = 0; r < rows; ++r) {
                s += xc[i * rows + r] * xc[j * rows + r];
            }
            r_mat[i * um + j] = s;
        }
    }

    Array<f64> v_mat(um * um, 0.0, scratch.allocator());
    jacobi_eig_sym(r_mat, v_mat, um);
    Array<f64> lam(um, 0.0, scratch.allocator());
    for (usize i = 0; i < um; ++i) lam[i] = r_mat[i * um + i];

    Array<usize> ord(um, usize{0}, scratch.allocator());
    for (usize i = 0; i < um; ++i) ord[i] = i;
    keel::sort(ord, [&lam](usize a, usize b) noexcept { return lam[a] < lam[b]; });
    const usize n_noise = um - as<usize>(p);

    const f64 inv_nfft = 1.0 / as<f64>(nfft);
    for (usize k = 0; k < nf; ++k) {
        const f64 w = kTau * as<f64>(k) * inv_nfft;
        f64 sum = 0.0;
        for (usize t = 0; t < n_noise; ++t) {
            const usize col = ord[t];
            f64 re = 0.0;
            f64 im = 0.0;
            for (usize idx = 0; idx < um; ++idx) {
                const f64 vv = v_mat[idx * um + col];
                const f64 ang = w * as<f64>(idx);
                re += vv * keel::cos(ang);
                im += vv * keel::sin(ang);
            }
            f64 proj = re * re + im * im;
            if (eig_weight) {
                proj /= (lam[col] > 1e-300) ? lam[col] : 1e-300;
            }
            sum += proj;
        }
        p_out[k] = (sum > 0.0) ? (1.0 / sum) : Limits<f64>::infinity();
        if (!f_out.empty()) {
            f_out[k] = fs * as<f64>(k) * inv_nfft;
        }
    }
    return SignalStatus::kOk;
}

} // namespace

SignalStatus levinson(ArrayRef<f64> r, i32 order, LevinsonResult& out, Allocator* alloc) {
    i32 n = order;
    if (n < 0) n = as<i32>(r.size()) - 1;
    if (n < 0) n = 0;
    if (as<usize>(n + 1) <= r.size()) {
        levinson_core(r, n, out, alloc);
    } else {
        ArenaTemp scratch = keel::get_scratch(alloc);
        Array<f64> rpad(as<usize>(n + 1), 0.0, scratch.allocator());
        for (usize i = 0; i < r.size(); ++i) rpad[i] = r[i];
        levinson_core(rpad.as_ref(), n, out, alloc);
    }
    return SignalStatus::kOk;
}

SignalStatus rlevinson(ArrayRef<f64> a, f64 efinal, RlevinsonResult& out, Allocator* alloc) {
    if (a.empty()) return SignalStatus::kInvalidArgument;
    if (!keel::is_finite(a[0]) || a[0] == 0.0) return SignalStatus::kZeroLeadingCoeff;
    const i32 p = as<i32>(a.size()) - 1;
    out.r = Array<f64>(as<usize>(p + 1), 0.0, alloc);
    out.k = Array<f64>(as<usize>(p), 0.0, alloc);
    if (p == 0) {
        out.r[0] = efinal;
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cur(a.size(), 0.0, scratch.allocator());
    const f64 inv_a0 = 1.0 / a[0];
    for (usize i = 0; i < a.size(); ++i) cur[i] = a[i] * inv_a0;
    f64 e = efinal;
    for (i32 i = p; i >= 1; --i) {
        const f64 ki = cur[as<usize>(i)];
        out.k[as<usize>(i - 1)] = ki;
        const f64 denom = 1.0 - ki * ki;
        if (keel::abs(denom) < 1e-300) break;
        Array<f64> prev(as<usize>(i + 1), 0.0, scratch.allocator());
        prev[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            prev[as<usize>(j)] = (cur[as<usize>(j)] - ki * cur[as<usize>(i - j)]) / denom;
        }
        cur = keel::move(prev);
        e /= denom;
    }

    return rc2ac(out.k.as_ref(), e, out.r, alloc);
}

SignalStatus aryule(ArrayRef<f64> x, i32 order, LevinsonResult& out, Allocator* alloc) {
    if (order < 0) return SignalStatus::kInvalidArgument;
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<f64> r = biased_autocorr(x, order, scratch.allocator());
    levinson_core(r.as_ref(), order, out, alloc);
    return SignalStatus::kOk;
}

SignalStatus arburg(ArrayRef<f64> x, i32 order, LevinsonResult& out, Allocator* alloc) {
    if (order < 0) return SignalStatus::kInvalidArgument;
    const usize n = x.size();
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> ef(x, scratch.allocator());
    Array<f64> eb(x, scratch.allocator());
    out.a = Array<f64>(as<usize>(order + 1), 0.0, alloc);
    out.a[0] = 1.0;
    out.k = Array<f64>(as<usize>(order), 0.0, alloc);

    f64 e = 0.0;
    for (usize i = 0; i < n; ++i) e += x[i] * x[i];
    e = (n > 0) ? (e / as<f64>(n)) : 0.0;

    Array<f64> a_new(as<usize>(order + 1), 0.0, scratch.allocator());
    for (i32 i = 0; i < order; ++i) {
        f64 num = 0.0;
        f64 den = 0.0;
        for (usize t = as<usize>(i + 1); t < n; ++t) {
            num += ef[t] * eb[t - 1];
            den += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }
        const f64 ki = (den > 0.0) ? (-2.0 * num / den) : 0.0;
        out.k[as<usize>(i)] = ki;

        for (usize t = n; t > as<usize>(i + 1); --t) {
            const usize idx = t - 1;
            const f64 f = ef[idx];
            const f64 b = eb[idx - 1];
            ef[idx] = f + ki * b;
            eb[idx] = b + ki * f;
        }

        a_new[0] = 1.0;
        for (i32 j = 1; j <= i; ++j) {
            a_new[as<usize>(j)] = out.a[as<usize>(j)] + ki * out.a[as<usize>(i + 1 - j)];
        }
        a_new[as<usize>(i + 1)] = ki;
        for (i32 j = 0; j <= i + 1; ++j) out.a[as<usize>(j)] = a_new[as<usize>(j)];
        e *= (1.0 - ki * ki);
    }
    out.e = e;
    return SignalStatus::kOk;
}

SignalStatus arcov(ArrayRef<f64> x, i32 order, Array<f64>& a_out, f64& e_out, Allocator* alloc) {
    if (order < 0) return SignalStatus::kInvalidArgument;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> r = cov_matrix_forward(x, order, scratch.allocator());
    const i32 n = as<i32>(x.size());
    const f64 norm = as<f64>(keel::max(n - order, i32{1}));
    for (usize i = 0; i < r.size(); ++i) r[i] /= norm;
    solve_ar_cov(r.as_ref(), order, a_out, e_out, alloc);
    return SignalStatus::kOk;
}

SignalStatus armcov(ArrayRef<f64> x, i32 order, Array<f64>& a_out, f64& e_out, Allocator* alloc) {
    if (order < 0) return SignalStatus::kInvalidArgument;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> r = cov_matrix_forward_backward(x, order, scratch.allocator());
    const i32 n = as<i32>(x.size());
    const f64 norm = as<f64>(keel::max(2 * (n - order), i32{1}));
    for (usize i = 0; i < r.size(); ++i) r[i] /= norm;
    solve_ar_cov(r.as_ref(), order, a_out, e_out, alloc);
    return SignalStatus::kOk;
}

SignalStatus schurrc(ArrayRef<f64> r, Array<f64>& k_out, Allocator* alloc) {
    const i32 p = as<i32>(r.size()) - 1;
    if (p <= 0) {
        k_out = Array<f64>(alloc);
        return SignalStatus::kOk;
    }
    k_out = Array<f64>(as<usize>(p), 0.0, alloc);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> a_prev(as<usize>(p + 1), 0.0, scratch.allocator());
    Array<f64> a_new(as<usize>(p + 1), 0.0, scratch.allocator());
    a_prev[0] = 1.0;
    f64 e = r[0];
    for (i32 i = 1; i <= p; ++i) {
        f64 num = r[as<usize>(i)];
        for (i32 j = 1; j < i; ++j) {
            num += a_prev[as<usize>(j)] * r[as<usize>(i - j)];
        }
        const f64 ki = (e != 0.0) ? (-num / e) : 0.0;
        k_out[as<usize>(i - 1)] = ki;
        a_new[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            a_new[as<usize>(j)] = a_prev[as<usize>(j)] + ki * a_prev[as<usize>(i - j)];
        }
        a_new[as<usize>(i)] = ki;
        for (i32 j = 0; j <= i; ++j) a_prev[as<usize>(j)] = a_new[as<usize>(j)];
        e *= (1.0 - ki * ki);
    }
    return SignalStatus::kOk;
}

SignalStatus rc2ac(ArrayRef<f64> k, f64 r0, Array<f64>& r_out, Allocator* alloc) {
    const i32 p = as<i32>(k.size());
    r_out = Array<f64>(as<usize>(p + 1), 0.0, alloc);
    r_out[0] = r0;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> a(as<usize>(p + 1), 0.0, scratch.allocator());
    Array<f64> a_new(as<usize>(p + 1), 0.0, scratch.allocator());
    a[0] = 1.0;
    f64 e = r0;
    for (i32 i = 1; i <= p; ++i) {
        const f64 ki = k[as<usize>(i - 1)];
        f64 s = 0.0;
        for (i32 j = 1; j < i; ++j) s += a[as<usize>(j)] * r_out[as<usize>(i - j)];
        r_out[as<usize>(i)] = -ki * e - s;
        a_new[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            a_new[as<usize>(j)] = a[as<usize>(j)] + ki * a[as<usize>(i - j)];
        }
        a_new[as<usize>(i)] = ki;
        for (i32 j = 0; j <= i; ++j) a[as<usize>(j)] = a_new[as<usize>(j)];
        e *= (1.0 - ki * ki);
    }
    return SignalStatus::kOk;
}

SignalStatus poly2rc(ArrayRef<f64> a, f64 efinal, Array<f64>& k_out, f64& r0_out,
                     Allocator* alloc) {
    if (a.empty()) return SignalStatus::kInvalidArgument;
    if (!keel::is_finite(a[0]) || a[0] == 0.0) return SignalStatus::kZeroLeadingCoeff;
    const i32 p = as<i32>(a.size()) - 1;
    k_out = Array<f64>(as<usize>(p), 0.0, alloc);
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> cur(a.size(), 0.0, scratch.allocator());
    const f64 inv_a0 = 1.0 / a[0];
    for (usize i = 0; i < a.size(); ++i) cur[i] = a[i] * inv_a0;
    for (i32 i = p; i >= 1; --i) {
        const f64 ki = cur[as<usize>(i)];
        k_out[as<usize>(i - 1)] = ki;
        const f64 denom = 1.0 - ki * ki;
        if (keel::abs(denom) < 1e-300) break;
        Array<f64> prev(as<usize>(i + 1), 0.0, scratch.allocator());
        prev[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            prev[as<usize>(j)] = (cur[as<usize>(j)] - ki * cur[as<usize>(i - j)]) / denom;
        }
        cur = keel::move(prev);
    }
    f64 prod = 1.0;
    for (usize i = 0; i < k_out.size(); ++i) prod *= (1.0 - k_out[i] * k_out[i]);
    r0_out = (prod != 0.0) ? (efinal / prod) : 0.0;
    return SignalStatus::kOk;
}

SignalStatus rc2poly(ArrayRef<f64> k, f64 r0, Array<f64>& a_out, f64& efinal_out,
                     Allocator* alloc) {
    const i32 p = as<i32>(k.size());
    a_out = Array<f64>(as<usize>(p + 1), 0.0, alloc);
    a_out[0] = 1.0;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> a_new(as<usize>(p + 1), 0.0, scratch.allocator());
    for (i32 i = 1; i <= p; ++i) {
        const f64 ki = k[as<usize>(i - 1)];
        a_new[0] = 1.0;
        for (i32 j = 1; j < i; ++j) {
            a_new[as<usize>(j)] = a_out[as<usize>(j)] + ki * a_out[as<usize>(i - j)];
        }
        a_new[as<usize>(i)] = ki;
        for (i32 j = 0; j <= i; ++j) a_out[as<usize>(j)] = a_new[as<usize>(j)];
    }
    f64 prod = 1.0;
    for (usize i = 0; i < k.size(); ++i) prod *= (1.0 - k[i] * k[i]);
    efinal_out = r0 * prod;
    return SignalStatus::kOk;
}

SignalStatus is2rc(ArrayRef<f64> is_in, MutArrayRef<f64> k_out) noexcept {
    if (k_out.size() < is_in.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < is_in.size(); ++i) k_out[i] = keel::sin(kHalfPi * is_in[i]);
    return SignalStatus::kOk;
}

SignalStatus rc2is(ArrayRef<f64> k, MutArrayRef<f64> is_out) noexcept {
    if (is_out.size() < k.size()) return SignalStatus::kInvalidArgument;
    constexpr f64 kTwoOverPi = 2.0 / kPi;
    for (usize i = 0; i < k.size(); ++i) {
        if (keel::is_nan(k[i])) {
            is_out[i] = Limits<f64>::quiet_nan();
            continue;
        }
        const f64 c = keel::max(-1.0, keel::min(1.0, k[i]));
        is_out[i] = kTwoOverPi * keel::asin(c);
    }
    return SignalStatus::kOk;
}

SignalStatus lar2rc(ArrayRef<f64> g, MutArrayRef<f64> k_out) noexcept {
    if (k_out.size() < g.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < g.size(); ++i) {
        k_out[i] = keel::tanh(0.5 * g[i]);
    }
    return SignalStatus::kOk;
}

SignalStatus rc2lar(ArrayRef<f64> k, MutArrayRef<f64> g_out) noexcept {
    if (g_out.size() < k.size()) return SignalStatus::kInvalidArgument;
    for (usize i = 0; i < k.size(); ++i) {
        const f64 ki = k[i];
        if (keel::is_nan(ki)) {
            g_out[i] = Limits<f64>::quiet_nan();
        } else if (ki >= 1.0) {
            g_out[i] = Limits<f64>::infinity();
        } else if (ki <= -1.0) {
            g_out[i] = -Limits<f64>::infinity();
        } else {
            g_out[i] = keel::log((1.0 + ki) / (1.0 - ki));
        }
    }
    return SignalStatus::kOk;
}

SignalStatus prony(ArrayRef<f64> h, i32 nb, i32 na, TfResult& out, Allocator* alloc) {
    if (nb < 0 || na < 0) return SignalStatus::kInvalidArgument;
    const i32 n = as<i32>(h.size());
    if (na == 0) {
        out.b = Array<f64>(as<usize>(nb + 1), 0.0, alloc);
        for (i32 i = 0; i <= nb && i < n; ++i) out.b[as<usize>(i)] = h[as<usize>(i)];
        out.a = Array<f64>(1, 1.0, alloc);
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    const i32 rows = keel::max(n - 1 - nb, i32{1});
    const usize urows = as<usize>(rows);
    const usize una = as<usize>(na);
    Array<f64> m_mat(urows * una, 0.0, scratch.allocator());
    Array<f64> rhs(urows, 0.0, scratch.allocator());
    for (i32 i = 0; i < rows; ++i) {
        for (i32 j = 0; j < na; ++j) {
            const i32 idx = nb + i - j;
            m_mat[as<usize>(i) * una + as<usize>(j)] =
                (idx >= 0 && idx < n) ? h[as<usize>(idx)] : 0.0;
        }
        const i32 rhs_idx = nb + 1 + i;
        rhs[as<usize>(i)] = (rhs_idx < n) ? -h[as<usize>(rhs_idx)] : 0.0;
    }

    Array<f64> mtm(una * una, 0.0, scratch.allocator());
    Array<f64> mtb(una, 0.0, scratch.allocator());
    for (i32 i = 0; i < na; ++i) {
        const usize ui = as<usize>(i);
        for (i32 j = 0; j < na; ++j) {
            const usize uj = as<usize>(j);
            f64 s = 0.0;
            for (i32 k = 0; k < rows; ++k) {
                s += m_mat[as<usize>(k) * una + ui] * m_mat[as<usize>(k) * una + uj];
            }
            mtm[ui * una + uj] = s;
        }
        f64 s = 0.0;
        for (i32 k = 0; k < rows; ++k) s += m_mat[as<usize>(k) * una + ui] * rhs[as<usize>(k)];
        mtb[ui] = s;
    }

    const Array<f64> a_tail = solve_linear_system(mtm, mtb, na, scratch.allocator());
    out.a = Array<f64>(una + 1, 0.0, alloc);
    out.a[0] = 1.0;
    for (i32 k = 0; k < na; ++k) out.a[as<usize>(k + 1)] = a_tail[as<usize>(k)];

    out.b = Array<f64>(as<usize>(nb + 1), 0.0, alloc);
    for (i32 i = 0; i <= nb; ++i) {
        f64 s = 0.0;
        const i32 lim = keel::min(i, na);
        for (i32 k = 0; k <= lim; ++k) {
            if (i - k < n) s += out.a[as<usize>(k)] * h[as<usize>(i - k)];
        }
        out.b[as<usize>(i)] = s;
    }
    return SignalStatus::kOk;
}

SignalStatus stmcb(ArrayRef<f64> y, ArrayRef<f64> x_opt, i32 nb, i32 na, i32 niter,
                   ArrayRef<f64> ai_opt, TfResult& out, Allocator* alloc) {
    if (nb < 0 || na < 0) return SignalStatus::kInvalidArgument;
    const i32 n = as<i32>(y.size());
    if (!x_opt.empty() && x_opt.size() != y.size()) return SignalStatus::kInvalidArgument;
    if (!ai_opt.empty()) {
        if (ai_opt.size() != as<usize>(na + 1) && ai_opt.size() != as<usize>(na)) {
            return SignalStatus::kInvalidArgument;
        }
        if (ai_opt.size() == as<usize>(na + 1) && ai_opt[0] == 0.0) {
            return SignalStatus::kZeroLeadingCoeff;
        }
    }
    if (niter < 1) niter = 5;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> xv(as<usize>(n), 0.0, scratch.allocator());
    if (!x_opt.empty()) {
        for (usize i = 0; i < as<usize>(n); ++i) xv[i] = x_opt[i];
    } else if (n > 0) {
        xv[0] = 1.0;
    }

    out.b = Array<f64>(as<usize>(nb + 1), 0.0, alloc);
    out.a = Array<f64>(as<usize>(na + 1), 0.0, alloc);
    out.a[0] = 1.0;

    if (ai_opt.empty()) {
        TfResult init;
        const SignalStatus st = prony(y, x_opt.empty() ? nb : 0, na, init, scratch.allocator());
        if (st != SignalStatus::kOk) return st;
        for (usize k = 0; k < init.b.size() && k < out.b.size(); ++k) out.b[k] = init.b[k];
        for (usize k = 0; k < init.a.size() && k < out.a.size(); ++k) out.a[k] = init.a[k];
        out.a[0] = 1.0;
    } else if (ai_opt.size() == as<usize>(na + 1)) {
        const f64 a0 = ai_opt[0];
        for (i32 k = 0; k <= na; ++k) out.a[as<usize>(k)] = ai_opt[as<usize>(k)] / a0;
        out.a[0] = 1.0;
    } else {
        for (i32 k = 0; k < na; ++k) out.a[as<usize>(k + 1)] = ai_opt[as<usize>(k)];
    }

    const i32 p_tot = nb + 1 + na;
    const usize up = as<usize>(p_tot);
    Array<f64> e_buf(as<usize>(n), 0.0, scratch.allocator());
    Array<f64> g_buf(as<usize>(n), 0.0, scratch.allocator());
    Array<f64> mtm(up * up, 0.0, scratch.allocator());
    Array<f64> mtg(up, 0.0, scratch.allocator());

    for (i32 it = 0; it < niter; ++it) {
        for (i32 t = 0; t < n; ++t) {
            f64 se = xv[as<usize>(t)];
            f64 sg = y[as<usize>(t)];
            const i32 lim = keel::min(na, t);
            for (i32 i = 1; i <= lim; ++i) {
                const f64 ai = out.a[as<usize>(i)];
                se -= ai * e_buf[as<usize>(t - i)];
                sg -= ai * g_buf[as<usize>(t - i)];
            }
            e_buf[as<usize>(t)] = se;
            g_buf[as<usize>(t)] = sg;
        }
        auto mcol = [&](i32 t, i32 col) noexcept -> f64 {
            if (col <= nb) {
                return (t - col >= 0) ? e_buf[as<usize>(t - col)] : 0.0;
            }
            const i32 s = col - nb;
            return (t - s >= 0) ? -g_buf[as<usize>(t - s)] : 0.0;
        };
        for (usize k = 0; k < up * up; ++k) mtm[k] = 0.0;
        for (usize k = 0; k < up; ++k) mtg[k] = 0.0;
        for (i32 t = 0; t < n; ++t) {
            const f64 gt = g_buf[as<usize>(t)];
            for (i32 i = 0; i < p_tot; ++i) {
                const f64 mi = mcol(t, i);
                if (mi == 0.0) continue;
                const usize ui = as<usize>(i);
                mtg[ui] += mi * gt;
                for (i32 j = 0; j < p_tot; ++j) {
                    mtm[ui * up + as<usize>(j)] += mi * mcol(t, j);
                }
            }
        }
        const Array<f64> c = solve_linear_system(mtm, mtg, p_tot, scratch.allocator());
        for (i32 k = 0; k <= nb; ++k) out.b[as<usize>(k)] = c[as<usize>(k)];
        for (i32 k = 0; k < na; ++k) out.a[as<usize>(k + 1)] = c[as<usize>(nb + 1 + k)];
    }
    return SignalStatus::kOk;
}

SignalStatus corrmtx(ArrayRef<f64> x, i32 m, MutArrayRef<f64> out_col_major) noexcept {
    if (m < 0) return SignalStatus::kInvalidArgument;
    const i32 n = as<i32>(x.size());
    const usize rows = as<usize>(n) + as<usize>(m);
    const usize cols = as<usize>(m) + 1;
    if (out_col_major.size() / cols < rows) return SignalStatus::kInvalidArgument;
    for (usize k = 0; k < rows * cols; ++k) out_col_major[k] = 0.0;
    for (usize j = 0; j < cols; ++j) {
        for (usize i = 0; i < rows; ++i) {
            const i32 idx = as<i32>(i) - as<i32>(j);
            if (idx >= 0 && idx < n) {
                out_col_major[j * rows + i] = x[as<usize>(idx)];
            }
        }
    }
    return SignalStatus::kOk;
}

SignalStatus poly2lsf(ArrayRef<f64> a, Array<f64>& lsf_out, Allocator* alloc) {
    const i32 n = as<i32>(a.size()) - 1;
    lsf_out = Array<f64>(alloc);
    if (n <= 0) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> p_poly(as<usize>(n + 2), 0.0, scratch.allocator());
    Array<f64> q_poly(as<usize>(n + 2), 0.0, scratch.allocator());
    for (i32 i = 0; i <= n; ++i) {
        p_poly[as<usize>(i)] = a[as<usize>(i)];
        q_poly[as<usize>(i)] = a[as<usize>(i)];
    }
    for (i32 i = 1; i <= n + 1; ++i) {
        const f64 rev = a[as<usize>(n + 1 - i)];
        p_poly[as<usize>(i)] += rev;
        q_poly[as<usize>(i)] -= rev;
    }

    const ArrayRef<f64> polys[2] = {p_poly.as_ref(), q_poly.as_ref()};
    for (usize pi = 0; pi < 2; ++pi) {
        const Array<c64> rts = keel::poly::roots(polys[pi], scratch.allocator());
        for (usize i = 0; i < rts.size(); ++i) {
            const f64 ang = keel::atan2(rts[i].imag(), rts[i].real());
            if (ang > 1e-9 && ang < kPi - 1e-9) {
                lsf_out.push_back(ang);
            }
        }
    }
    keel::sort(lsf_out);
    if (as<i32>(lsf_out.size()) > n) lsf_out.resize(as<usize>(n));
    return SignalStatus::kOk;
}

SignalStatus lsf2poly(ArrayRef<f64> lsf, Array<f64>& a_out, Allocator* alloc) {
    const i32 m = as<i32>(lsf.size());
    if (m == 0) {
        a_out = Array<f64>(1, 1.0, alloc);
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> p_poly(scratch.allocator());
    Array<f64> q_poly(scratch.allocator());
    if (m % 2 == 1) {
        p_poly.push_back(1.0);
        q_poly.push_back(1.0);
        q_poly.push_back(0.0);
        q_poly.push_back(-1.0);
    } else {
        p_poly.push_back(1.0);
        p_poly.push_back(1.0);
        q_poly.push_back(1.0);
        q_poly.push_back(-1.0);
    }
    for (i32 i = 0; i < m; ++i) {
        const f64 quad[3] = {1.0, -2.0 * keel::cos(lsf[as<usize>(i)]), 1.0};
        if (i % 2 == 0) {
            p_poly = keel::poly::mul(p_poly.as_ref(), ArrayRef<f64>(quad, 3), scratch.allocator());
        } else {
            q_poly = keel::poly::mul(q_poly.as_ref(), ArrayRef<f64>(quad, 3), scratch.allocator());
        }
    }
    const usize out_len = (p_poly.size() > 1) ? (p_poly.size() - 1) : p_poly.size();
    a_out = Array<f64>(out_len, 0.0, alloc);
    for (usize i = 0; i < out_len; ++i) {
        a_out[i] = 0.5 * (p_poly[i] + q_poly[i]);
    }
    return SignalStatus::kOk;
}

SignalStatus invfreqs(ArrayRef<c64> h, ArrayRef<f64> w, i32 nb, i32 na, TfResult& out,
                      Allocator* alloc) {
    if (h.size() != w.size()) return SignalStatus::kInvalidArgument;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> x(w.size(), c64(0.0, 0.0), scratch.allocator());
    for (usize k = 0; k < w.size(); ++k) x[k] = c64(0.0, w[k]);
    return invfreq_lsq(h, x.as_ref(), nb, na, /*descending=*/true, out, alloc);
}

SignalStatus invfreqz(ArrayRef<c64> h, ArrayRef<f64> w, i32 nb, i32 na, TfResult& out,
                      Allocator* alloc) {
    if (h.size() != w.size()) return SignalStatus::kInvalidArgument;
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> x(w.size(), c64(0.0, 0.0), scratch.allocator());
    for (usize k = 0; k < w.size(); ++k) x[k] = c64(keel::cos(-w[k]), keel::sin(-w[k]));
    return invfreq_lsq(h, x.as_ref(), nb, na, /*descending=*/false, out, alloc);
}

SignalStatus pmusic(ArrayRef<f64> x, i32 p, i32 nfft, f64 fs, MutArrayRef<f64> p_out,
                    MutArrayRef<f64> f_out, Allocator* alloc) {
    return pseudospectrum_impl(x, p, nfft, fs, /*eig_weight=*/false, p_out, f_out, alloc);
}

SignalStatus peig(ArrayRef<f64> x, i32 p, i32 nfft, f64 fs, MutArrayRef<f64> p_out,
                  MutArrayRef<f64> f_out, Allocator* alloc) {
    return pseudospectrum_impl(x, p, nfft, fs, /*eig_weight=*/true, p_out, f_out, alloc);
}

} // namespace keel::signal
