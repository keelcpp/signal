#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"
#include "keel/poly.h"

namespace keel::signal {

namespace {

constexpr f64 kImagThresh = 1e-10;

inline bool is_real_root(c64 r) noexcept {
    return keel::abs(r.imag()) <= kImagThresh;
}

struct RootPair {
    c64 a;
    c64 b;
    bool is_pair = false;
};

struct QuadCoeffs {
    f64 b1 = 0.0;
    f64 b2 = 0.0;
};

bool pop_pole_pair(Array<c64>& pool, RootPair& out) noexcept {
    if (pool.empty()) return false;
    usize pick_idx = 0;
    f64 best_abs = keel::abs(pool[0]);
    f64 best_re = keel::abs(pool[0].real());
    for (usize i = 1; i < pool.size(); ++i) {
        const f64 a = keel::abs(pool[i]);
        const f64 re = keel::abs(pool[i].real());
        if (a < best_abs || (a == best_abs && re < best_re)) {
            pick_idx = i;
            best_abs = a;
            best_re = re;
        }
    }
    const c64 pick = pool[pick_idx];
    if (is_real_root(pick)) {
        usize mate_idx = pool.size();
        f64 best_dist = Limits<f64>::infinity();
        for (usize i = 0; i < pool.size(); ++i) {
            if (i == pick_idx || !is_real_root(pool[i])) continue;
            const f64 d = keel::abs(pool[i] - pick);
            if (d < best_dist) {
                best_dist = d;
                mate_idx = i;
            }
        }
        if (mate_idx < pool.size()) {
            out = {c64(pick.real(), 0.0), c64(pool[mate_idx].real(), 0.0), true};
            const usize hi = keel::max(pick_idx, mate_idx);
            const usize lo = keel::min(pick_idx, mate_idx);
            pool.erase(hi);
            pool.erase(lo);
        } else {
            out = {c64(pick.real(), 0.0), c64(0.0, 0.0), false};
            pool.erase(pick_idx);
        }
    } else {
        const c64 conj_p = keel::conj(pick);
        usize mate_idx = pool.size();
        f64 best_dist = Limits<f64>::infinity();
        for (usize i = 0; i < pool.size(); ++i) {
            if (i == pick_idx) continue;
            const f64 d = keel::abs(pool[i] - conj_p);
            if (d < best_dist) {
                best_dist = d;
                mate_idx = i;
            }
        }
        if (mate_idx >= pool.size()) {
            return false;
        }
        out = {pick, pool[mate_idx], true};
        const usize hi = keel::max(pick_idx, mate_idx);
        const usize lo = keel::min(pick_idx, mate_idx);
        pool.erase(hi);
        pool.erase(lo);
    }
    return true;
}

bool pop_closest_zero_pair(Array<c64>& zeros, c64 target, RootPair& out) noexcept {
    if (zeros.empty()) return false;
    usize pick_idx = 0;
    f64 best_dist = keel::abs(zeros[0] - target);
    for (usize i = 1; i < zeros.size(); ++i) {
        const f64 d = keel::abs(zeros[i] - target);
        if (d < best_dist) {
            best_dist = d;
            pick_idx = i;
        }
    }
    const c64 pick = zeros[pick_idx];
    if (is_real_root(pick)) {
        usize mate_idx = zeros.size();
        f64 best_mate = Limits<f64>::infinity();
        for (usize i = 0; i < zeros.size(); ++i) {
            if (i == pick_idx || !is_real_root(zeros[i])) continue;
            const f64 d = keel::abs(zeros[i] - target);
            if (d < best_mate) {
                best_mate = d;
                mate_idx = i;
            }
        }
        if (mate_idx < zeros.size()) {
            out = {c64(pick.real(), 0.0), c64(zeros[mate_idx].real(), 0.0), true};
            const usize hi = keel::max(pick_idx, mate_idx);
            const usize lo = keel::min(pick_idx, mate_idx);
            zeros.erase(hi);
            zeros.erase(lo);
        } else {
            out = {c64(pick.real(), 0.0), c64(0.0, 0.0), false};
            zeros.erase(pick_idx);
        }
    } else {
        const c64 conj_p = keel::conj(pick);
        usize mate_idx = zeros.size();
        f64 best_dist2 = Limits<f64>::infinity();
        for (usize i = 0; i < zeros.size(); ++i) {
            if (i == pick_idx) continue;
            const f64 d = keel::abs(zeros[i] - conj_p);
            if (d < best_dist2) {
                best_dist2 = d;
                mate_idx = i;
            }
        }
        if (mate_idx >= zeros.size()) {
            return false;
        }
        out = {pick, zeros[mate_idx], true};
        const usize hi = keel::max(pick_idx, mate_idx);
        const usize lo = keel::min(pick_idx, mate_idx);
        zeros.erase(hi);
        zeros.erase(lo);
    }
    return true;
}

QuadCoeffs pair_to_quad(const RootPair& p) noexcept {
    if (!p.is_pair) {
        return {-p.a.real(), 0.0};
    }
    if (!is_real_root(p.a) || !is_real_root(p.b)) {
        return {-2.0 * p.a.real(), p.a.real() * p.a.real() + p.a.imag() * p.a.imag()};
    }
    return {-(p.a.real() + p.b.real()), p.a.real() * p.b.real()};
}

inline c64 pair_centroid(const RootPair& p) noexcept {
    if (!p.is_pair) return p.a;
    return (p.a + p.b) * 0.5;
}

void append_section_roots(const f64* coeffs, usize n, Array<c64>& out, Allocator* scratch_alloc) {
    usize trimmed_len = n;
    while (trimmed_len > 1 && keel::abs(coeffs[trimmed_len - 1]) < 1e-15) {
        --trimmed_len;
    }
    if (trimmed_len < 2) return;
    Array<c64> r = keel::poly::roots(ArrayRef<f64>(coeffs, trimmed_len), scratch_alloc);
    for (usize i = 0; i < r.size(); ++i) {
        out.push_back(r[i]);
    }
}

} // namespace

SignalStatus zp2sos(ArrayRef<c64> zeros_in, ArrayRef<c64> poles_in, f64 gain,
                    bool surplus_at_origin, SosResult& out, Allocator* alloc) {
    if (poles_in.empty()) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<c64> zeros(zeros_in, scratch.allocator());
    Array<c64> poles(poles_in, scratch.allocator());

    const usize num_sec = (poles.size() + 1) / 2;
    Array<f64> b1s(num_sec, 0.0, scratch.allocator());
    Array<f64> b2s(num_sec, 0.0, scratch.allocator());
    Array<f64> a1s(num_sec, 0.0, scratch.allocator());
    Array<f64> a2s(num_sec, 0.0, scratch.allocator());
    Array<u8> no_zeros(num_sec, u8{0}, scratch.allocator());

    for (usize s = 0; s < num_sec; ++s) {
        RootPair pole_pair;
        if (!pop_pole_pair(poles, pole_pair)) {
            return SignalStatus::kInvalidArgument;
        }
        const QuadCoeffs pol_q = pair_to_quad(pole_pair);
        a1s[s] = pol_q.b1;
        a2s[s] = pol_q.b2;

        if (!zeros.empty()) {
            RootPair zero_pair;
            if (!pop_closest_zero_pair(zeros, pair_centroid(pole_pair), zero_pair)) {
                return SignalStatus::kInvalidArgument;
            }
            const QuadCoeffs zer_q = pair_to_quad(zero_pair);
            b1s[s] = zer_q.b1;
            b2s[s] = zer_q.b2;
            no_zeros[s] = 0;
        } else {
            b1s[s] = 0.0;
            b2s[s] = 0.0;
            no_zeros[s] = 1;
        }
    }
    if (!zeros.empty()) {
        return SignalStatus::kInvalidArgument;
    }

    out.sos_col_major = Array<f64>(num_sec * 6, 0.0, alloc);
    out.num_sections = num_sec;
    out.gain = gain;
    f64* p = out.sos_col_major.data();
    for (usize r = 0; r < num_sec; ++r) {
        if (no_zeros[r] != 0 && surplus_at_origin) {
            p[0 * num_sec + r] = 0.0;
            p[1 * num_sec + r] = 0.0;
            p[2 * num_sec + r] = 1.0;
        } else {
            p[0 * num_sec + r] = 1.0;
            p[1 * num_sec + r] = b1s[r];
            p[2 * num_sec + r] = b2s[r];
        }
        p[3 * num_sec + r] = 1.0;
        p[4 * num_sec + r] = a1s[r];
        p[5 * num_sec + r] = a2s[r];
    }
    return SignalStatus::kOk;
}

SignalStatus tf2sos(ArrayRef<c64> b, ArrayRef<c64> a, SosResult& out, Allocator* alloc) {
    if (a.empty() || a[0] == c64(0.0, 0.0)) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (b.empty()) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 gain = (b[0] / a[0]).real();
    const Array<c64> zeros = keel::poly::roots(b, scratch.allocator());
    const Array<c64> poles = keel::poly::roots(a, scratch.allocator());
    return zp2sos(zeros.as_ref(), poles.as_ref(), gain, /*surplus_at_origin=*/false, out, alloc);
}

SignalStatus tf2sos(ArrayRef<f64> b, ArrayRef<f64> a, SosResult& out, Allocator* alloc) {
    if (a.empty() || a[0] == 0.0) {
        return SignalStatus::kZeroLeadingCoeff;
    }
    if (b.empty()) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    const f64 gain = b[0] / a[0];
    const Array<c64> zeros = keel::poly::roots(b, scratch.allocator());
    const Array<c64> poles = keel::poly::roots(a, scratch.allocator());
    return zp2sos(zeros.as_ref(), poles.as_ref(), gain, /*surplus_at_origin=*/false, out, alloc);
}

SignalStatus sos2tf(ArrayRef<f64> sos_col_major, usize num_sections, f64 gain, TfResult& out,
                    Allocator* alloc) {
    if (num_sections == 0 || sos_col_major.size() < 6 * num_sections) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> b(1, 1.0, scratch.allocator());
    Array<f64> a(1, 1.0, scratch.allocator());
    for (usize i = 0; i < num_sections; ++i) {
        const f64 bs[3] = {sos_col_major[0 * num_sections + i],
                           sos_col_major[1 * num_sections + i],
                           sos_col_major[2 * num_sections + i]};
        const f64 as_sec[3] = {sos_col_major[3 * num_sections + i],
                               sos_col_major[4 * num_sections + i],
                               sos_col_major[5 * num_sections + i]};
        b = keel::poly::mul(b.as_ref(), ArrayRef<f64>(bs, 3), scratch.allocator());
        a = keel::poly::mul(a.as_ref(), ArrayRef<f64>(as_sec, 3), scratch.allocator());
    }
    if (gain != 1.0) {
        for (usize i = 0; i < b.size(); ++i) b[i] *= gain;
    }
    out.b = Array<f64>(b.as_ref(), alloc);
    out.a = Array<f64>(a.as_ref(), alloc);
    return SignalStatus::kOk;
}

SignalStatus sos2zp(ArrayRef<f64> sos_col_major, usize num_sections, f64 gain, ZpkResult& out,
                    Allocator* alloc) {
    if (num_sections == 0 || sos_col_major.size() < 6 * num_sections) {
        return SignalStatus::kInvalidArgument;
    }
    ArenaTemp scratch = keel::get_scratch(alloc);
    out.z = Array<c64>(alloc);
    out.p = Array<c64>(alloc);
    f64 total_gain = gain;
    for (usize i = 0; i < num_sections; ++i) {
        const f64 b_arr[3] = {sos_col_major[0 * num_sections + i],
                              sos_col_major[1 * num_sections + i],
                              sos_col_major[2 * num_sections + i]};
        const f64 a_arr[3] = {sos_col_major[3 * num_sections + i],
                              sos_col_major[4 * num_sections + i],
                              sos_col_major[5 * num_sections + i]};
        append_section_roots(b_arr, 3, out.z, scratch.allocator());
        append_section_roots(a_arr, 3, out.p, scratch.allocator());
        if (keel::abs(a_arr[0]) > 1e-15) {
            total_gain *= b_arr[0] / a_arr[0];
        }
    }
    out.k = total_gain;
    return SignalStatus::kOk;
}

SignalStatus tf2ss(ArrayRef<f64> b, ArrayRef<f64> a, StateSpaceResult& out, Allocator* alloc) {
    const usize na = a.size();
    if (na == 0) return SignalStatus::kInvalidArgument;
    const f64 a0 = a[0];
    if (keel::abs(a0) < 1e-15) return SignalStatus::kZeroLeadingCoeff;
    if (na < 2) return SignalStatus::kInvalidArgument;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<f64> ah(na, scratch.allocator());
    for (usize i = 0; i < na; ++i) ah[i] = a[i] / a0;

    const usize nb = b.size();
    Array<f64> bh(na, 0.0, scratch.allocator());
    if (nb >= na) {
        for (usize i = 0; i < na; ++i) bh[i] = b[nb - na + i] / a0;
    } else {
        for (usize i = 0; i < nb; ++i) bh[na - nb + i] = b[i] / a0;
    }

    const usize order = na - 1;
    const f64 d_val = bh[0];

    out.order = order;
    out.d = d_val;
    out.a_col_major = Array<f64>(order * order, 0.0, alloc);
    for (usize j = 0; j < order; ++j) {
        out.a_col_major[0 + j * order] = -ah[j + 1];
    }
    for (usize i = 0; i + 1 < order; ++i) {
        out.a_col_major[(i + 1) + i * order] = 1.0;
    }

    out.b = Array<f64>(order, 0.0, alloc);
    out.b[0] = 1.0;

    out.c = Array<f64>(order, 0.0, alloc);
    for (usize j = 0; j < order; ++j) {
        out.c[j] = bh[j + 1] - ah[j + 1] * d_val;
    }
    return SignalStatus::kOk;
}

SignalStatus ss2tf(ArrayRef<f64> a_col_major, usize order, ArrayRef<f64> b, ArrayRef<f64> c, f64 d,
                   TfResult& out, Allocator* alloc) {
    if (a_col_major.size() < order * order || b.size() < order || c.size() < order) {
        return SignalStatus::kInvalidArgument;
    }
    if (order == 0) {
        out.b = Array<f64>(1, d, alloc);
        out.a = Array<f64>(1, 1.0, alloc);
        return SignalStatus::kOk;
    }

    ArenaTemp scratch = keel::get_scratch(alloc);
    auto mat_mul = [order](ArrayRef<f64> x, ArrayRef<f64> y, Array<f64>& res) noexcept {
        for (usize i = 0; i < order * order; ++i) res[i] = 0.0;
        for (usize i = 0; i < order; ++i) {
            for (usize j = 0; j < order; ++j) {
                f64 s = 0.0;
                for (usize k = 0; k < order; ++k) {
                    s += x[k * order + i] * y[j * order + k];
                }
                res[j * order + i] = s;
            }
        }
    };
    Array<f64> tmp_v(order, 0.0, scratch.allocator());
    auto dot_cmb = [order, &b, &c, &tmp_v](ArrayRef<f64> mat) noexcept {
        for (usize i = 0; i < order; ++i) {
            f64 s = 0.0;
            for (usize k = 0; k < order; ++k) {
                s += mat[k * order + i] * b[k];
            }
            tmp_v[i] = s;
        }
        f64 res = 0.0;
        for (usize i = 0; i < order; ++i) res += c[i] * tmp_v[i];
        return res;
    };

    Array<f64> m_mat(order * order, 0.0, scratch.allocator());
    Array<f64> am_mat(order * order, 0.0, scratch.allocator());
    for (usize i = 0; i < order; ++i) m_mat[i * order + i] = 1.0;

    Array<f64> a_coeffs(order + 1, 0.0, scratch.allocator());
    a_coeffs[order] = 1.0;
    Array<f64> cmb(order, 0.0, scratch.allocator());
    cmb[0] = dot_cmb(m_mat.as_ref());

    for (usize k = 1; k <= order; ++k) {
        mat_mul(a_col_major, m_mat.as_ref(), am_mat);
        f64 trace = 0.0;
        for (usize i = 0; i < order; ++i) trace += am_mat[i * order + i];
        const f64 ck = -trace / as<f64>(k);
        a_coeffs[order - k] = ck;
        for (usize i = 0; i < order * order; ++i) m_mat[i] = am_mat[i];
        for (usize i = 0; i < order; ++i) m_mat[i * order + i] += ck;
        if (k < order) cmb[k] = dot_cmb(m_mat.as_ref());
    }

    Array<f64> b_coeffs(order + 1, 0.0, scratch.allocator());
    b_coeffs[order] = d;
    for (usize k = 0; k < order; ++k) {
        b_coeffs[order - 1 - k] = d * a_coeffs[order - 1 - k] + cmb[k];
    }

    out.a = Array<f64>(order + 1, alloc);
    out.b = Array<f64>(order + 1, alloc);
    for (usize i = 0; i <= order; ++i) {
        out.a[i] = a_coeffs[order - i];
        out.b[i] = b_coeffs[order - i];
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
