#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

f64 peak_prominence(ArrayRef<f64> x, usize pi, f64 h, usize& lb, usize& rb) noexcept {
    const usize n = x.size();
    f64 left_min = Limits<f64>::infinity();
    lb = 0;
    for (usize j = pi; j-- > 0;) {
        if (x[j] > h) {
            lb = j;
            break;
        }
        left_min = keel::min(left_min, x[j]);
    }
    f64 right_min = Limits<f64>::infinity();
    rb = n - 1;
    for (usize j = pi + 1; j < n; ++j) {
        if (x[j] > h) {
            rb = j;
            break;
        }
        right_min = keel::min(right_min, x[j]);
    }
    const f64 base = keel::max(left_min, right_min);
    return h - base;
}

f64 peak_width(ArrayRef<f64> x, usize pi, f64 h, f64 prom, usize lb, usize rb,
               ArrayRef<f64> locs, PeakWidthRef width_ref) noexcept {
    const usize n = x.size();
    const bool has_locs = !locs.empty();
    auto pos = [&](usize i) noexcept -> f64 {
        return has_locs ? locs[i] : as<f64>(i + 1);
    };

    if (width_ref == PeakWidthRef::kHalfHeight) {
        const f64 ref = 0.5 * h;
        f64 lcross = pos(pi);
        {
            usize plateau = pi;
            bool crossed = false;
            for (usize k = pi; k-- > 0;) {
                if (x[k] > x[k + 1]) {
                    break;
                }
                if (x[k] <= ref) {
                    const f64 dx = x[k + 1] - x[k];
                    const f64 f = (dx != 0.0) ? ((ref - x[k]) / dx) : 0.0;
                    lcross = pos(k) + f * (pos(k + 1) - pos(k));
                    crossed = true;
                    break;
                }
                plateau = k;
            }
            if (!crossed) {
                lcross = pos(plateau);
            }
        }
        f64 rcross = pos(pi);
        {
            usize plateau = pi;
            bool crossed = false;
            for (usize k = pi + 1; k < n; ++k) {
                if (x[k] > x[k - 1]) {
                    break;
                }
                if (x[k] <= ref) {
                    const f64 dx = x[k - 1] - x[k];
                    const f64 f = (dx != 0.0) ? ((ref - x[k]) / dx) : 0.0;
                    rcross = pos(k) - f * (pos(k) - pos(k - 1));
                    crossed = true;
                    break;
                }
                if (x[k] < x[k - 1]) {
                    plateau = k;
                }
            }
            if (!crossed) {
                rcross = pos(plateau);
            }
        }
        return rcross - lcross;
    }

    const f64 ref = h - 0.5 * prom;
    f64 lcross = pos(lb);
    for (usize j = pi; j > lb; --j) {
        if (x[j - 1] < ref && x[j] >= ref) {
            const f64 dx = x[j] - x[j - 1];
            const f64 f = (dx != 0.0) ? ((ref - x[j - 1]) / dx) : 0.0;
            lcross = pos(j - 1) + f * (pos(j) - pos(j - 1));
            break;
        }
    }
    f64 rcross = pos(rb);
    for (usize j = pi; j < rb; ++j) {
        if (x[j] >= ref && x[j + 1] < ref) {
            const f64 dx = x[j] - x[j + 1];
            const f64 f = (dx != 0.0) ? ((x[j] - ref) / dx) : 0.0;
            rcross = pos(j) + f * (pos(j + 1) - pos(j));
            break;
        }
    }
    return rcross - lcross;
}

} // namespace

SignalStatus findpeaks(ArrayRef<f64> x, ArrayRef<f64> locations, const PeakOptions& opts,
                       PeakResult& out, Allocator* alloc) {
    const usize n = x.size();
    if (!locations.empty() && locations.size() < n) {
        return SignalStatus::kInvalidArgument;
    }
    if (keel::is_nan(opts.min_height) || keel::is_nan(opts.threshold) ||
        (opts.use_min_prom && keel::is_nan(opts.min_prom)) ||
        (opts.use_min_dist && keel::is_nan(opts.min_dist)) ||
        (opts.use_width_filter && (keel::is_nan(opts.min_width) || keel::is_nan(opts.max_width)))) {
        return SignalStatus::kInvalidArgument;
    }

    Array<f64> val(alloc);
    Array<f64> loc(alloc);
    Array<usize> sidx(alloc);

    if (n >= 3) {
        for (usize i = 1; i + 1 < n; ++i) {
            const f64 v = x[i];
            if (keel::is_nan(v) || keel::is_nan(x[i - 1]) || keel::is_nan(x[i + 1])) {
                continue;
            }
            if (!(v > x[i - 1] && v > x[i + 1])) {
                continue;
            }
            if (!(v > opts.min_height)) {
                continue;
            }
            const f64 drop = keel::min(v - x[i - 1], v - x[i + 1]);
            if (drop < opts.threshold) {
                continue;
            }
            val.push_back(v);
            loc.push_back(locations.empty() ? as<f64>(i + 1) : locations[i]);
            sidx.push_back(i);
        }
    }

    const usize m = val.size();
    Array<f64> prom(m, 0.0, alloc);
    Array<f64> wid(m, 0.0, alloc);
    for (usize k = 0; k < m; ++k) {
        usize lb = 0;
        usize rb = 0;
        prom[k] = peak_prominence(x, sidx[k], val[k], lb, rb);
        wid[k] = peak_width(x, sidx[k], val[k], prom[k], lb, rb, locations, opts.width_ref);
    }

    Array<u8> keep(m, u8{1}, alloc);
    if (opts.use_min_prom) {
        for (usize k = 0; k < m; ++k) {
            if (!(prom[k] >= opts.min_prom)) {
                keep[k] = 0;
            }
        }
    }
    if (opts.use_width_filter) {
        for (usize k = 0; k < m; ++k) {
            if (!(wid[k] >= opts.min_width && wid[k] <= opts.max_width)) {
                keep[k] = 0;
            }
        }
    }

    if (opts.use_min_dist && opts.min_dist > 0.0 && m > 1) {
        Array<usize> order(m, usize{0}, alloc);
        for (usize k = 0; k < m; ++k) {
            order[k] = k;
        }
        keel::sort(order, [&](usize a, usize b) noexcept {
            if (val[a] != val[b]) return val[a] > val[b];
            if (loc[a] != loc[b]) return loc[a] < loc[b];
            return a < b;
        });
        for (usize oi = 0; oi < m; ++oi) {
            const usize a = order[oi];
            if (keep[a] == 0) continue;
            for (usize oj = oi + 1; oj < m; ++oj) {
                const usize b = order[oj];
                if (keep[b] != 0 && keel::abs(loc[a] - loc[b]) < opts.min_dist) {
                    keep[b] = 0;
                }
            }
        }
    }

    Array<usize> surv(alloc);
    surv.reserve(m);
    for (usize k = 0; k < m; ++k) {
        if (keep[k] != 0) {
            surv.push_back(k);
        }
    }

    if (opts.sort_mode == PeakSort::kAscend) {
        keel::sort(surv, [&](usize a, usize b) noexcept {
            if (val[a] != val[b]) return val[a] < val[b];
            if (loc[a] != loc[b]) return loc[a] < loc[b];
            return a < b;
        });
    } else if (opts.sort_mode == PeakSort::kDescend) {
        keel::sort(surv, [&](usize a, usize b) noexcept {
            if (val[a] != val[b]) return val[a] > val[b];
            if (loc[a] != loc[b]) return loc[a] < loc[b];
            return a < b;
        });
    }

    usize keep_n = surv.size();
    if (opts.max_peaks >= 0 && as<usize>(opts.max_peaks) < keep_n) {
        keep_n = as<usize>(opts.max_peaks);
    }

    out.values = Array<f64>(keep_n, 0.0, alloc);
    out.locations = Array<f64>(keep_n, 0.0, alloc);
    out.widths = Array<f64>(keep_n, 0.0, alloc);
    out.prominences = Array<f64>(keep_n, 0.0, alloc);
    for (usize k = 0; k < keep_n; ++k) {
        const usize s = surv[k];
        out.values[k] = val[s];
        out.locations[k] = loc[s];
        out.widths[k] = wid[s];
        out.prominences[k] = prom[s];
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
