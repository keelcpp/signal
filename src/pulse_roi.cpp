#include "keel/signal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/array.h"
#include "keel/math.h"

namespace keel::signal {

namespace {

constexpr f64 kLowPct = 0.10;
constexpr f64 kHighPct = 0.90;
constexpr f64 kMidPct = 0.50;
constexpr i32 kHistBins = 100;

inline f64 crossing_idx(ArrayRef<f64> x, usize i, f64 level) noexcept {
    const f64 a = x[i];
    const f64 b = x[i + 1];
    if (a == b) return as<f64>(i);
    return as<f64>(i) + (level - a) / (b - a);
}

inline f64 idx_to_time(f64 idx, f64 fs) noexcept {
    return (fs > 0.0) ? (idx / fs) : (idx + 1.0);
}

struct Transition {
    i32 sign = 0;
    f64 start_idx = 0.0;
    f64 end_idx = 0.0;
};

Array<Transition> find_transitions(ArrayRef<f64> x, f64 low, f64 high, Allocator* alloc) {
    Array<Transition> out(alloc);
    const usize n = x.size();
    if (n < 2 || !(high > low)) return out;

    const f64 range = high - low;
    const f64 trail = low + kLowPct * range;
    const f64 lead = low + kHighPct * range;

    enum class State : u8 { kBelow = 0, kMid, kAbove };
    auto classify = [trail, lead](f64 v) noexcept -> State {
        if (v <= trail) return State::kBelow;
        if (v >= lead) return State::kAbove;
        return State::kMid;
    };

    State cur = classify(x[0]);
    f64 pending_start = 0.0;
    bool in_rise = false;
    bool in_fall = false;

    for (usize i = 1; i < n; ++i) {
        const State s = classify(x[i]);
        if (s == State::kBelow) {
            in_rise = false;
            if (in_fall) {
                out.push_back(Transition{-1, pending_start, crossing_idx(x, i - 1, trail)});
                in_fall = false;
            }
        } else if (s == State::kAbove) {
            in_fall = false;
            if (in_rise) {
                out.push_back(Transition{+1, pending_start, crossing_idx(x, i - 1, lead)});
                in_rise = false;
            }
        }

        if (cur == State::kBelow && s != State::kBelow && !in_rise) {
            in_rise = true;
            pending_start = crossing_idx(x, i - 1, trail);
            if (s == State::kAbove) {
                out.push_back(Transition{+1, pending_start, crossing_idx(x, i - 1, lead)});
                in_rise = false;
            }
        }
        if (cur == State::kAbove && s != State::kAbove && !in_fall) {
            in_fall = true;
            pending_start = crossing_idx(x, i - 1, lead);
            if (s == State::kBelow) {
                out.push_back(Transition{-1, pending_start, crossing_idx(x, i - 1, trail)});
                in_fall = false;
            }
        }
        cur = s;
    }
    return out;
}

struct DirectedCross {
    f64 idx = 0.0;
    i32 dir = 0;
};

Array<DirectedCross> mid_crosses_directed(ArrayRef<f64> x, f64 mid, Allocator* alloc) {
    Array<DirectedCross> out(alloc);
    for (usize i = 0; i + 1 < x.size(); ++i) {
        const f64 a = x[i];
        const f64 b = x[i + 1];
        if (a < mid && b >= mid) {
            out.push_back(DirectedCross{crossing_idx(x, i, mid), +1});
        } else if (a > mid && b <= mid) {
            out.push_back(DirectedCross{crossing_idx(x, i, mid), -1});
        }
    }
    return out;
}

} // namespace

SignalStatus statelevels(ArrayRef<f64> x, StateLevels& out) noexcept {
    const usize n = x.size();
    if (n == 0) {
        out = {0.0, 0.0};
        return SignalStatus::kOk;
    }
    bool have_finite = false;
    f64 xmin = 0.0;
    f64 xmax = 0.0;
    for (usize i = 0; i < n; ++i) {
        if (!keel::is_finite(x[i])) continue;
        if (!have_finite) {
            xmin = x[i];
            xmax = x[i];
            have_finite = true;
        } else {
            if (x[i] < xmin) xmin = x[i];
            if (x[i] > xmax) xmax = x[i];
        }
    }
    if (!have_finite || !(xmax > xmin)) {
        out = {xmin, xmax};
        return SignalStatus::kOk;
    }

    const f64 width = (xmax - xmin) / as<f64>(kHistBins);
    i32 hist[kHistBins]{};
    for (usize i = 0; i < n; ++i) {
        if (!keel::is_finite(x[i])) continue;
        i32 b = as<i32>((x[i] - xmin) / width);
        if (b >= kHistBins) b = kHistBins - 1;
        if (b < 0) b = 0;
        hist[b] += 1;
    }

    const f64 mid = (xmin + xmax) * 0.5;
    const i32 split_bin = as<i32>((mid - xmin) / width);
    i32 low_bin = 0;
    i32 high_bin = kHistBins - 1;
    i32 low_max = -1;
    i32 high_max = -1;
    for (i32 b = 0; b < split_bin; ++b) {
        if (hist[b] > low_max) {
            low_max = hist[b];
            low_bin = b;
        }
    }
    for (i32 b = split_bin; b < kHistBins; ++b) {
        if (hist[b] > high_max) {
            high_max = hist[b];
            high_bin = b;
        }
    }
    out.low = xmin + (as<f64>(low_bin) + 0.5) * width;
    out.high = xmin + (as<f64>(high_bin) + 0.5) * width;
    return SignalStatus::kOk;
}

SignalStatus midcross(ArrayRef<f64> x, f64 fs, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    out = Array<f64>(alloc);
    for (usize i = 0; i + 1 < x.size(); ++i) {
        const f64 a = x[i];
        const f64 b = x[i + 1];
        if ((a < mid && b >= mid) || (a > mid && b <= mid) || (a == mid && b != mid && i == 0)) {
            out.push_back(idx_to_time(crossing_idx(x, i, mid), fs));
        }
    }
    return SignalStatus::kOk;
}

SignalStatus risetime_full(ArrayRef<f64> x, f64 fs, bool rising, RiseFallResult& out,
                           Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<Transition> trs = find_transitions(x, lh.low, lh.high, scratch.allocator());

    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    const f64 range = lh.high - lh.low;
    out.ll = lh.low + kLowPct * range;
    out.ul = lh.low + kHighPct * range;
    out.dur = Array<f64>(alloc);
    out.lt = Array<f64>(alloc);
    out.ut = Array<f64>(alloc);

    for (usize i = 0; i < trs.size(); ++i) {
        const Transition& t = trs[i];
        if (rising && t.sign <= 0) continue;
        if (!rising && t.sign >= 0) continue;
        const f64 lower_cross = rising ? t.start_idx : t.end_idx;
        const f64 upper_cross = rising ? t.end_idx : t.start_idx;
        out.dur.push_back((t.end_idx - t.start_idx) / div_f);
        out.lt.push_back(idx_to_time(lower_cross, fs));
        out.ut.push_back(idx_to_time(upper_cross, fs));
    }
    return SignalStatus::kOk;
}

SignalStatus slewrate(ArrayRef<f64> x, f64 fs, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<Transition> trs = find_transitions(x, lh.low, lh.high, scratch.allocator());
    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    const f64 range = (lh.high - lh.low) * (kHighPct - kLowPct);
    out = Array<f64>(alloc);
    for (usize i = 0; i < trs.size(); ++i) {
        const f64 dur = (trs[i].end_idx - trs[i].start_idx) / div_f;
        out.push_back((dur > 0.0) ? (as<f64>(trs[i].sign) * range / dur) : 0.0);
    }
    return SignalStatus::kOk;
}

SignalStatus overshoot(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc) {
    out = Array<f64>(alloc);
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 range = lh.high - lh.low;
    if (range <= 0.0) return SignalStatus::kOk;
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<Transition> trs = find_transitions(x, lh.low, lh.high, scratch.allocator());
    for (usize k = 0; k < trs.size(); ++k) {
        if (trs[k].sign <= 0) continue;
        const usize wbeg = as<usize>(keel::ceil(trs[k].end_idx));
        const usize wend =
            (k + 1 < trs.size()) ? as<usize>(keel::floor(trs[k + 1].start_idx)) : x.size();
        if (wbeg >= wend || wbeg >= x.size()) continue;
        f64 peak = x[wbeg];
        for (usize i = wbeg + 1; i < wend && i < x.size(); ++i) {
            if (x[i] > peak) peak = x[i];
        }
        out.push_back(100.0 * (peak - lh.high) / range);
    }
    return SignalStatus::kOk;
}

SignalStatus undershoot(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc) {
    out = Array<f64>(alloc);
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 range = lh.high - lh.low;
    if (range <= 0.0) return SignalStatus::kOk;
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<Transition> trs = find_transitions(x, lh.low, lh.high, scratch.allocator());
    for (usize k = 0; k < trs.size(); ++k) {
        if (trs[k].sign >= 0) continue;
        const usize wbeg = as<usize>(keel::ceil(trs[k].end_idx));
        const usize wend =
            (k + 1 < trs.size()) ? as<usize>(keel::floor(trs[k + 1].start_idx)) : x.size();
        if (wbeg >= wend || wbeg >= x.size()) continue;
        f64 trough = x[wbeg];
        for (usize i = wbeg + 1; i < wend && i < x.size(); ++i) {
            if (x[i] < trough) trough = x[i];
        }
        out.push_back(100.0 * (lh.low - trough) / range);
    }
    return SignalStatus::kOk;
}

SignalStatus settlingtime(ArrayRef<f64> x, f64 fs, f64 tol, Array<f64>& out, Allocator* alloc) {
    out = Array<f64>(alloc);
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 range = lh.high - lh.low;
    if (range <= 0.0) return SignalStatus::kOk;
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<Transition> trs = find_transitions(x, lh.low, lh.high, scratch.allocator());
    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    for (usize k = 0; k < trs.size(); ++k) {
        const Transition& t = trs[k];
        const f64 dest = (t.sign > 0) ? lh.high : lh.low;
        const f64 band = tol * range;
        const usize wend =
            (k + 1 < trs.size()) ? as<usize>(keel::floor(trs[k + 1].start_idx)) : x.size();
        usize last_out = as<usize>(keel::floor(t.start_idx));
        for (usize i = last_out; i + 1 < wend && i < x.size(); ++i) {
            if (keel::abs(x[i] - dest) > band) last_out = i;
        }
        const f64 settled = as<f64>(last_out) - t.start_idx;
        out.push_back(keel::max(0.0, settled) / div_f);
    }
    return SignalStatus::kOk;
}

SignalStatus pulsewidth(ArrayRef<f64> x, f64 fs, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<DirectedCross> crs = mid_crosses_directed(x, mid, scratch.allocator());
    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    out = Array<f64>(alloc);
    for (usize i = 0; i + 1 < crs.size(); ++i) {
        if (crs[i].dir == +1 && crs[i + 1].dir == -1) {
            out.push_back((crs[i + 1].idx - crs[i].idx) / div_f);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus pulseperiod(ArrayRef<f64> x, f64 fs, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<DirectedCross> crs = mid_crosses_directed(x, mid, scratch.allocator());
    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    Array<f64> rising(scratch.allocator());
    for (usize i = 0; i < crs.size(); ++i) {
        if (crs[i].dir == +1) rising.push_back(crs[i].idx);
    }
    out = Array<f64>(alloc);
    for (usize i = 0; i + 1 < rising.size(); ++i) {
        out.push_back((rising[i + 1] - rising[i]) / div_f);
    }
    return SignalStatus::kOk;
}

SignalStatus pulsesep(ArrayRef<f64> x, f64 fs, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<DirectedCross> crs = mid_crosses_directed(x, mid, scratch.allocator());
    const f64 div_f = (fs > 0.0) ? fs : 1.0;
    out = Array<f64>(alloc);
    for (usize i = 0; i + 1 < crs.size(); ++i) {
        if (crs[i].dir == -1 && crs[i + 1].dir == +1) {
            out.push_back((crs[i + 1].idx - crs[i].idx) / div_f);
        }
    }
    return SignalStatus::kOk;
}

SignalStatus dutycycle(ArrayRef<f64> x, Array<f64>& out, Allocator* alloc) {
    StateLevels lh;
    (void)statelevels(x, lh);
    const f64 mid = lh.low + kMidPct * (lh.high - lh.low);
    ArenaTemp scratch = keel::get_scratch(alloc);
    const Array<DirectedCross> crs = mid_crosses_directed(x, mid, scratch.allocator());
    out = Array<f64>(alloc);
    for (usize i = 0; i + 2 < crs.size(); ++i) {
        if (crs[i].dir == +1 && crs[i + 1].dir == -1 && crs[i + 2].dir == +1) {
            const f64 width = crs[i + 1].idx - crs[i].idx;
            const f64 period = crs[i + 2].idx - crs[i].idx;
            if (period > 0.0) out.push_back(width / period);
        }
    }
    return SignalStatus::kOk;
}

// ============================================================================
// Signal Regions of Interest (ROI)
// ============================================================================

SignalStatus binmask2sigroi(ArrayRef<u8> mask, Array<SigRoi>& out, Allocator* alloc) {
    out = Array<SigRoi>(alloc);
    bool in_run = false;
    i64 cur_start = 0;
    for (usize i = 0; i < mask.size(); ++i) {
        const bool v = (mask[i] != 0);
        if (v && !in_run) {
            in_run = true;
            cur_start = as<i64>(i + 1);
        } else if (!v && in_run) {
            in_run = false;
            out.push_back(SigRoi{cur_start, as<i64>(i)});
        }
    }
    if (in_run) {
        out.push_back(SigRoi{cur_start, as<i64>(mask.size())});
    }
    return SignalStatus::kOk;
}

SignalStatus sigroi2binmask(ArrayRef<SigRoi> rois, i64 len_user, Array<u8>& out_mask,
                            Allocator* alloc) {
    if (len_user == 0) return SignalStatus::kInvalidArgument;
    i64 len = len_user;
    if (len < 0) {
        len = 0;
        for (usize i = 0; i < rois.size(); ++i) {
            if (rois[i].end > len) len = rois[i].end;
        }
    }
    out_mask = Array<u8>(as<usize>(len), u8{0}, alloc);
    if (len == 0) return SignalStatus::kOk;
    for (usize i = 0; i < rois.size(); ++i) {
        const i64 s = keel::max<i64>(1, rois[i].start);
        const i64 e = keel::min<i64>(len, rois[i].end);
        for (i64 k = s; k <= e; ++k) out_mask[as<usize>(k - 1)] = 1;
    }
    return SignalStatus::kOk;
}

SignalStatus extendsigroi(ArrayRef<SigRoi> rois, i64 l_pre, i64 l_post, Array<SigRoi>& out,
                          Allocator* alloc) {
    if (l_pre < 0 || l_post < 0) return SignalStatus::kInvalidArgument;
    out = Array<SigRoi>(rois.size(), SigRoi{}, alloc);
    for (usize i = 0; i < rois.size(); ++i) {
        i64 s = rois[i].start - l_pre;
        const i64 e = rois[i].end + l_post;
        if (s < 1) s = 1;
        out[i] = SigRoi{s, e};
    }
    return SignalStatus::kOk;
}

SignalStatus shortensigroi(ArrayRef<SigRoi> rois, i64 l_pre, i64 l_post, Array<SigRoi>& out,
                           Allocator* alloc) {
    if (l_pre < 0 || l_post < 0) return SignalStatus::kInvalidArgument;
    out = Array<SigRoi>(alloc);
    for (usize i = 0; i < rois.size(); ++i) {
        const i64 s = rois[i].start + l_pre;
        const i64 e = rois[i].end - l_post;
        if (s <= e) out.push_back(SigRoi{s, e});
    }
    return SignalStatus::kOk;
}

SignalStatus mergesigroi(ArrayRef<SigRoi> rois, i64 sep, Array<SigRoi>& out, Allocator* alloc) {
    if (sep < 0) return SignalStatus::kInvalidArgument;
    out = Array<SigRoi>(alloc);
    if (rois.empty()) return SignalStatus::kOk;

    ArenaTemp scratch = keel::get_scratch(alloc);
    Array<SigRoi> sorted(rois, scratch.allocator());
    keel::sort(sorted, [](const SigRoi& a, const SigRoi& b) noexcept {
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });

    for (usize k = 0; k < sorted.size(); ++k) {
        const i64 s = sorted[k].start;
        const i64 e = sorted[k].end;
        if (!out.empty()) {
            const i64 prev_end = out[out.size() - 1].end;
            if (s <= prev_end || (s - prev_end - 1) <= sep) {
                if (e > prev_end) out[out.size() - 1].end = e;
                continue;
            }
        }
        out.push_back(SigRoi{s, e});
    }
    return SignalStatus::kOk;
}

SignalStatus removesigroi(ArrayRef<SigRoi> rois, i64 max_len, Array<SigRoi>& out,
                          Allocator* alloc) {
    out = Array<SigRoi>(alloc);
    for (usize i = 0; i < rois.size(); ++i) {
        const i64 len = rois[i].end - rois[i].start + 1;
        if (len > max_len) out.push_back(rois[i]);
    }
    return SignalStatus::kOk;
}

} // namespace keel::signal
