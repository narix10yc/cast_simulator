//! Generic adaptive timing utility.

use std::fmt;
use std::time::Instant;

/// Statistics returned by [`time_adaptive`].
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
fn fmt_duration(secs: f64) -> String {
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
    /// Displays as `213 ms ± 71.0 ms (33 iters)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ± {} ({} iter{})",
            fmt_duration(self.mean_s),
            fmt_duration(self.stddev_s),
            self.n_iters,
            if self.n_iters == 1 { "" } else { "s" },
        )
    }
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
    const WARMUP_FRACTION: f64 = 0.25;
    const OVER_BUDGET_FACTOR: f64 = 1.5;

    // Phase 1: single probe — warmup and cost estimate.
    let t_start = Instant::now();
    let t = Instant::now();
    f()?;
    let probe_time = t.elapsed().as_secs_f64();
    let est_per_iter = probe_time.max(1e-9);

    if probe_time > budget_s * OVER_BUDGET_FACTOR {
        return Ok(TimingStats {
            n_iters: 1,
            mean_s: probe_time,
            stddev_s: 0.0,
            cv: 0.0,
            min_s: probe_time,
            max_s: probe_time,
        });
    }

    // Phase 2: fill remaining warmup budget with un-timed iterations.
    let warmup_budget = budget_s * WARMUP_FRACTION;
    let remaining_warmup = (warmup_budget - probe_time).max(0.0);
    let n_extra_warmup = (remaining_warmup / est_per_iter) as u32;
    for _ in 0..n_extra_warmup {
        f()?;
    }

    // Phase 3: timed measurements over the remaining ~3/4 of budget.
    let elapsed = t_start.elapsed().as_secs_f64();
    let remaining = (budget_s - elapsed).max(0.0);
    let n_timed = ((remaining / est_per_iter) as u32).max(1);
    let mut samples: Vec<f64> = Vec::with_capacity(n_timed as usize);
    for _ in 0..n_timed {
        let t = Instant::now();
        f()?;
        samples.push(t.elapsed().as_secs_f64());
    }

    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(TimingStats {
        n_iters: samples.len(),
        mean_s: mean,
        stddev_s: stddev,
        cv: stddev / mean,
        min_s: min,
        max_s: max,
    })
}
