//! Generic adaptive timing utility.

use std::fmt;
use std::time::{Duration, Instant};

/// Statistics returned by [`time_adaptive`] and [`time_adaptive_with`].
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub n_iters: usize,
    pub mean_s: f64,
    pub stddev_s: f64,
    /// Coefficient of variation: `stddev / mean` (scale-free noise metric).
    pub cv: f64,
    pub min_s: f64,
    pub max_s: f64,
}

/// Formats a duration in seconds to 3 significant figures with an appropriate
/// unit (s, ms, µs, ns).
pub fn fmt_duration(secs: f64) -> String {
    let (val, unit) = if secs >= 1.0 {
        (secs, "s")
    } else if secs >= 1e-3 {
        (secs * 1e3, "ms")
    } else if secs >= 1e-6 {
        (secs * 1e6, "µs")
    } else {
        (secs * 1e9, "ns")
    };

    // 3 significant figures: compute decimal places from magnitude.
    let digits = if val >= 100.0 {
        0
    } else if val >= 10.0 {
        1
    } else {
        2
    };
    format!("{val:.digits$} {unit}")
}

impl fmt::Display for TimingStats {
    /// Displays as `213 ms ± 71.0 ms (33 iters)` for multi-iter results,
    /// or just `6.18 s (1 iter)` when there is only a single sample (no
    /// meaningful stddev).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n_iters <= 1 {
            write!(f, "{} (1 iter)", fmt_duration(self.mean_s))
        } else {
            write!(
                f,
                "{} ± {} ({} iters)",
                fmt_duration(self.mean_s),
                fmt_duration(self.stddev_s),
                self.n_iters,
            )
        }
    }
}

fn stats_from_samples(samples: &[f64]) -> TimingStats {
    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    TimingStats {
        n_iters: samples.len(),
        mean_s: mean,
        stddev_s: stddev,
        cv: stddev / mean,
        min_s: min,
        max_s: max,
    }
}

/// Adaptively times a fallible closure that reports its own duration.
///
/// Like [`time_adaptive`], but instead of wall-clock timing the closure, the
/// closure itself returns a [`Duration`] representing the cost of the
/// operation it wants measured.  The adaptive budget logic still uses
/// wall-clock time so that the total profiling session respects the budget,
/// but the *samples* come from the closure's reported durations.
///
/// This is useful when the interesting cost is a sub-operation (e.g. GPU
/// kernel time measured via hardware events) rather than the full closure
/// wall-clock time which may include launch overhead.
pub fn time_adaptive_with<E>(
    mut f: impl FnMut() -> Result<Duration, E>,
    budget_s: f64,
) -> Result<TimingStats, E> {
    const WARMUP_FRACTION: f64 = 0.25;
    const OVER_BUDGET_FACTOR: f64 = 1.5;

    // Phase 1: single probe — warmup and cost estimate.
    let wall_start = Instant::now();
    let wall_before = Instant::now();
    let probe_duration = f()?;
    let probe_wall = wall_before.elapsed().as_secs_f64();
    let est_per_iter = probe_wall.max(1e-9);

    if probe_wall > budget_s * OVER_BUDGET_FACTOR {
        let s = probe_duration.as_secs_f64();
        return Ok(TimingStats {
            n_iters: 1,
            mean_s: s,
            stddev_s: 0.0,
            cv: 0.0,
            min_s: s,
            max_s: s,
        });
    }

    // Phase 2: fill remaining warmup budget with un-timed iterations.
    let warmup_budget = budget_s * WARMUP_FRACTION;
    let remaining_warmup = (warmup_budget - probe_wall).max(0.0);
    let n_extra_warmup = (remaining_warmup / est_per_iter) as u32;
    for _ in 0..n_extra_warmup {
        f()?;
    }

    // Phase 3: timed measurements over the remaining ~3/4 of budget.
    let elapsed = wall_start.elapsed().as_secs_f64();
    let remaining = (budget_s - elapsed).max(0.0);
    let n_timed = ((remaining / est_per_iter) as u32).max(1);
    let mut samples: Vec<f64> = Vec::with_capacity(n_timed as usize);
    for _ in 0..n_timed {
        samples.push(f()?.as_secs_f64());
    }

    Ok(stats_from_samples(&samples))
}

/// Adaptively times a fallible closure within a wall-time `budget_s`.
///
/// A single probe run starts the clock and estimates per-iteration cost.
/// If that one run already exceeds `1.5 × budget_s`, it is returned as the
/// sole sample — budget is better respected than forcing more iterations.
/// Otherwise the budget is split: ~1/4 for warmup (probe + extra un-timed
/// passes), ~3/4 for timed measurements. Fast operations get many warmup
/// passes; slow ones keep only the single probe as warmup.
pub fn time_adaptive<E>(
    mut f: impl FnMut() -> Result<(), E>,
    budget_s: f64,
) -> Result<TimingStats, E> {
    time_adaptive_with(
        || {
            let t = Instant::now();
            f()?;
            Ok(t.elapsed())
        },
        budget_s,
    )
}
