use std::collections::HashSet;
use std::sync::Arc;

use crate::{
    cost_model::{FusionConfig, FusionDecision, FusionLog},
    types::QuantumGate,
    CircuitGraph, GateId,
};

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: size-2 canonicalization (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

pub fn apply_size_two_fusion(cg: &mut CircuitGraph) {
    let n_absorbed = absorb_single_qubit_gates(cg);
    if n_absorbed > 0 {
        cg.squeeze();
    }

    let n_two_qubit_fused = fuse_adjacent_two_qubit_gates(cg);
    if n_two_qubit_fused > 0 {
        cg.squeeze();
    }
}

fn absorb_single_qubit_gates(cg: &mut CircuitGraph) -> usize {
    let mut n_fused = 0;
    let n_rows = cg.n_rows();
    let n_qubits = cg.n_qubits();

    for row in 0..n_rows {
        for qubit in 0..n_qubits {
            if absorb_single_qubit_gate(cg, row, qubit) {
                n_fused += 1;
            }
        }
    }

    n_fused
}

fn absorb_single_qubit_gate(cg: &mut CircuitGraph, row: usize, qubit: usize) -> bool {
    let gate_id = match cg.gate_id_at(row, qubit) {
        Some(gate_id) => gate_id,
        None => return false,
    };

    let gate = match cg.gate_arc(gate_id) {
        Some(gate) if gate.n_qubits() == 1 && gate.is_unitary() => gate.clone(),
        _ => return false,
    };

    for next_row in row + 1..cg.n_rows() {
        let Some(next_gate_id) = cg.gate_id_at(next_row, qubit) else {
            continue;
        };
        let Some(next_gate) = cg.gate(next_gate_id) else {
            continue;
        };
        // A channel gate on this qubit is a causal barrier; cannot fuse past it.
        if !next_gate.is_unitary() {
            break;
        }
        let fused = next_gate.matmul(&gate);
        cg.remove_gate_at(row, qubit);
        cg.remove_gate_at(next_row, qubit);
        cg.insert_gate_at_row(next_row, fused);
        return true;
    }

    for prev_row in (0..row).rev() {
        let Some(prev_gate_id) = cg.gate_id_at(prev_row, qubit) else {
            continue;
        };
        let Some(prev_gate) = cg.gate(prev_gate_id) else {
            continue;
        };
        if !prev_gate.is_unitary() {
            break;
        }
        let fused = gate.matmul(prev_gate);
        cg.remove_gate_at(row, qubit);
        cg.remove_gate_at(prev_row, qubit);
        cg.insert_gate_at_row(prev_row, fused);
        return true;
    }

    false
}

fn fuse_adjacent_two_qubit_gates(cg: &mut CircuitGraph) -> usize {
    let mut n_fused = 0;
    if cg.n_rows() < 2 {
        return n_fused;
    }

    let mut row = 0;
    while row + 1 < cg.n_rows() {
        for qubit in 0..cg.n_qubits() {
            if fuse_adjacent_two_qubit_gate(cg, row, qubit) {
                n_fused += 1;
            }
        }
        row += 1;
    }

    n_fused
}

fn fuse_adjacent_two_qubit_gate(cg: &mut CircuitGraph, row: usize, qubit: usize) -> bool {
    let next_row = row + 1;
    let Some(left_gate_id) = cg.gate_id_at(row, qubit) else {
        return false;
    };
    let Some(left_gate) = cg.gate(left_gate_id) else {
        return false;
    };
    if left_gate.n_qubits() != 2
        || !left_gate.is_unitary()
        || left_gate.qubits()[0] as usize != qubit
    {
        return false;
    }

    let Some(right_gate_id) = cg.gate_id_at(next_row, qubit) else {
        return false;
    };
    let Some(right_gate) = cg.gate(right_gate_id) else {
        return false;
    };
    if right_gate.n_qubits() != 2
        || !right_gate.is_unitary()
        || right_gate.qubits() != left_gate.qubits()
    {
        return false;
    }

    let fused = right_gate.matmul(left_gate);
    cg.remove_gate_at(row, qubit);
    cg.remove_gate_at(next_row, qubit);
    cg.insert_gate_at_row(next_row, fused);
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: agglomerative fusion with cost-model decisions
// ─────────────────────────────────────────────────────────────────────────────

/// Number of distinct qubits in the union of two sorted qubit slices.
/// Used as a cheap pre-filter before any matrix multiply.
fn union_qubit_count(a: &[u32], b: &[u32]) -> usize {
    let mut union = Vec::with_capacity(a.len() + b.len());
    for &qubit in a.iter().chain(b.iter()) {
        if !union.contains(&qubit) {
            union.push(qubit);
        }
    }
    union.len()
}

/// Attempts to fuse a cluster of gates starting at `(start_row, start_qubit)`.
///
/// Returns the number of *additional* gates consumed (0 = nothing changed).
/// Mirrors C++ `impl::startFusion` from `ImplOptimize.cpp`.
///
/// Algorithm:
/// 1. Seed: the gate at `(start_row, start_qubit)`. Skip if it is not the gate's
///    lowest qubit (prevents re-processing the same multi-qubit seed gate).
/// 2. Same-row sweep: absorb gates to the right in the same row while the
///    union qubit count stays ≤ `cdd_size`.
/// 3. Cross-row loop: advance one row at a time, absorbing one gate per pass
///    from the next row (on any qubit the product gate already covers).
///    Restart the loop on each absorption; exit when no progress is made.
/// 4. Cost check: compute `benefit = Σ old_costs / (new_cost + 1e-10) - 1.0`.
///    Accept if `benefit >= config.benefit_margin`; otherwise rollback by
///    restoring all original gates at their original rows.
fn start_fusion(
    cg: &mut CircuitGraph,
    config: &FusionConfig,
    cdd_size: usize,
    start_row: usize,
    start_qubit: usize,
    log: Option<&mut FusionLog>,
) -> usize {
    let Some(seed_id) = cg.gate_id_at(start_row, start_qubit) else {
        return 0;
    };
    let seed_gate = cg.gate_arc(seed_id).unwrap().clone();

    // Only process a gate from its lowest qubit to avoid duplicate seeds.
    // Channel gates cannot participate in matmul-based fusion.
    if seed_gate.qubits()[0] as usize != start_qubit || !seed_gate.is_unitary() {
        return 0;
    }

    // `tentative` tracks every original gate (and its row) that is part of the
    // candidate cluster. Index 0 is always the seed.
    let mut tentative: Vec<(Arc<QuantumGate>, usize)> = vec![(seed_gate.clone(), start_row)];
    let mut absorbed_ids: HashSet<GateId> = HashSet::from([seed_id]);

    let mut prod_id = seed_id;
    let mut prod_row = start_row;
    let mut prod_gate = seed_gate;

    // ── Step B: same-row sweep ────────────────────────────────────────────────
    for qubit in (start_qubit + 1)..cg.n_qubits() {
        let Some(cand_id) = cg.gate_id_at(prod_row, qubit) else {
            continue;
        };
        if absorbed_ids.contains(&cand_id) {
            continue;
        }
        let cand_gate = cg.gate_arc(cand_id).unwrap().clone();
        if !cand_gate.is_unitary() {
            continue;
        }
        if union_qubit_count(prod_gate.qubits(), cand_gate.qubits()) > cdd_size {
            continue;
        }

        tentative.push((cand_gate, prod_row));
        absorbed_ids.insert(cand_id);

        let anchor = prod_gate.qubits()[0] as usize;
        prod_id = cg.fuse_gates_in_same_row(prod_row, anchor, qubit).unwrap();
        // The newly created product gate must be absorbed too, otherwise the
        // next Phase-B iteration will find it in the same row and try to fuse
        // it with itself (same gate_id on both sides → fuse_gates_in_same_row
        // returns None and the unwrap panics).
        absorbed_ids.insert(prod_id);
        prod_gate = cg.gate_arc(prod_id).unwrap().clone();
    }

    // ── Step C: cross-row loop ────────────────────────────────────────────────
    'outer: loop {
        let next_row = prod_row + 1;
        if next_row >= cg.n_rows() {
            break;
        }
        // Snapshot qubit list before the borrow-checker complains about
        // simultaneous immutable + mutable borrows of `cg`.
        let prod_qubits: Vec<u32> = prod_gate.qubits().to_vec();
        for &q in &prod_qubits {
            let q = q as usize;
            let Some(cand_id) = cg.gate_id_at(next_row, q) else {
                continue;
            };
            if absorbed_ids.contains(&cand_id) {
                continue;
            }
            let cand_gate = cg.gate_arc(cand_id).unwrap().clone();
            if !cand_gate.is_unitary() {
                continue;
            }
            if union_qubit_count(prod_gate.qubits(), cand_gate.qubits()) > cdd_size {
                continue;
            }

            tentative.push((cand_gate, next_row));
            absorbed_ids.insert(cand_id);

            let anchor = prod_gate.qubits()[0] as usize;
            let (new_id, new_row) = cg
                .fuse_gates_across_rows(prod_row, anchor, next_row, q)
                .unwrap();
            prod_id = new_id;
            prod_row = new_row;
            absorbed_ids.insert(prod_id);
            prod_gate = cg.gate_arc(prod_id).unwrap().clone();
            continue 'outer;
        }
        break;
    }

    // Nothing was fused beyond the seed itself.
    if tentative.len() == 1 {
        return 0;
    }

    // ── Step D: cost-model decision ───────────────────────────────────────────
    let ztol = 1e-12;
    let old_cost: f64 = tentative
        .iter()
        .map(|(g, _)| config.cost_model.cost_of(g))
        .sum();
    let prod_gate_ref = cg.gate(prod_id).unwrap();
    let new_cost = config.cost_model.cost_of(prod_gate_ref);
    let benefit = old_cost / (new_cost + 1e-10) - 1.0;
    let accepted = benefit >= config.benefit_margin;

    if let Some(log) = log {
        log.decisions.push(FusionDecision {
            cdd_size,
            n_candidates: tentative.len(),
            candidates: tentative
                .iter()
                .map(|(g, _)| (g.effective_n_qubits(), g.arithmetic_intensity(ztol)))
                .collect(),
            product_n_qubits: prod_gate_ref.effective_n_qubits(),
            product_ai: prod_gate_ref.arithmetic_intensity(ztol),
            old_cost,
            new_cost,
            benefit,
            accepted,
        });
    }

    if !accepted {
        // Rollback: remove the product gate and restore all originals.
        let anchor = prod_gate.qubits()[0] as usize;
        cg.remove_gate_at(prod_row, anchor);
        for (orig_gate, orig_row) in tentative {
            cg.insert_gate_at_row(orig_row, orig_gate);
        }
        return 0;
    }

    tentative.len() - 1 // gates consumed (excludes the seed)
}

/// One agglomerative pass at a fixed candidate size `cdd_size`.
///
/// Iterates every `(row, qubit)` cell and calls [`start_fusion`]. Row count is
/// snapshotted before the loop; `start_fusion` may tombstone gate slots but
/// never removes rows, so indices remain stable throughout. After the pass,
/// calls [`CircuitGraph::squeeze`] to compact the graph.
///
/// Returns the total number of gates consumed (0 = graph unchanged).
pub fn apply_gate_fusion(
    cg: &mut CircuitGraph,
    config: &FusionConfig,
    cdd_size: usize,
    mut log: Option<&mut FusionLog>,
) -> usize {
    let mut n_fused = 0;
    // Snapshot n_rows / n_qubits: start_fusion mutates the graph but never
    // adds or removes rows, so the dimensions are stable for this pass.
    let n_rows = cg.n_rows();
    let n_qubits = cg.n_qubits();
    for row in 0..n_rows {
        for qubit in 0..n_qubits {
            n_fused += start_fusion(cg, config, cdd_size, row, qubit, log.as_deref_mut());
        }
    }
    if n_fused > 0 {
        cg.squeeze();
    }
    n_fused
}

/// Full optimizer: Phase 1 (size-2 canonicalization) followed by Phase 2
/// (agglomerative fusion from `cdd_size = 3` to `config.size_max`).
///
/// Each size level is repeated (multi-traverse) until no further fusions are
/// found, allowing earlier fusions to expose new opportunities after squeeze.
pub fn optimize(cg: &mut CircuitGraph, config: &FusionConfig) {
    apply_size_two_fusion(cg);
    for cdd_size in 3..=config.size_max {
        while apply_gate_fusion(cg, config, cdd_size, None) > 0 {}
    }
}

/// Like [`optimize`], but records every Phase-2 accept/reject decision in
/// the returned [`FusionLog`].
pub fn optimize_with_log(cg: &mut CircuitGraph, config: &FusionConfig) -> FusionLog {
    apply_size_two_fusion(cg);
    let mut log = FusionLog::new();
    for cdd_size in 3..=config.size_max {
        while apply_gate_fusion(cg, config, cdd_size, Some(&mut log)) > 0 {}
    }
    log
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{apply_gate_fusion, apply_size_two_fusion, optimize, union_qubit_count};
    use crate::{
        cost_model::{FusionConfig, SizeOnlyCostModel},
        types::{ComplexSquareMatrix, QuantumGate},
        CircuitGraph,
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashSet;

    // ═════════════════════════════════════════════════════════════════════════
    // Test helpers
    // ═════════════════════════════════════════════════════════════════════════

    /// Computes the full circuit unitary as a single [`QuantumGate`] by
    /// reducing all gates in row order.  Gates in the same row act on
    /// non-overlapping qubits and therefore commute, so intra-row ordering
    /// does not affect the result.
    fn circuit_unitary(cg: &CircuitGraph) -> QuantumGate {
        let mut gates: Vec<QuantumGate> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();
        for row in 0..cg.n_rows() {
            for qubit in 0..cg.n_qubits() {
                if let Some(gid) = cg.gate_id_at(row, qubit) {
                    if seen.insert(gid) {
                        gates.push(cg.gate(gid).unwrap().clone());
                    }
                }
            }
        }
        // Reduce: later gate on the left (g applied after acc).
        gates
            .into_iter()
            .reduce(|acc, g| g.matmul(&acc))
            .expect("circuit must be non-empty")
    }

    /// Asserts that `original` and `fused` implement the same unitary to
    /// within `tol` (element-wise maximum norm difference).
    fn assert_unitary_preserved(original: &CircuitGraph, fused: &CircuitGraph, tol: f64) {
        let u_orig = circuit_unitary(original);
        let u_fused = circuit_unitary(fused);
        assert_eq!(
            u_orig.qubits(),
            u_fused.qubits(),
            "qubit sets differ after fusion"
        );
        let diff = u_orig.matrix().maximum_norm_diff(u_fused.matrix());
        assert!(
            diff < tol,
            "fusion changed the circuit unitary (max diff = {diff:.2e}, tol = {tol:.2e})"
        );
    }

    /// Clones `cg`, applies `fusion_fn` to the clone, asserts the unitary is
    /// preserved, then returns the fused graph for further structural checks.
    fn check_fusion<F: FnOnce(&mut CircuitGraph)>(cg: &CircuitGraph, fusion_fn: F) -> CircuitGraph {
        let mut fused = cg.clone();
        fusion_fn(&mut fused);
        assert_unitary_preserved(cg, &fused, 1e-10);
        assert!(fused.check_consistency().is_ok());
        fused
    }

    /// Generates a reproducible random circuit of `n_qubits` qubits with
    /// `n_gates` random 1- or 2-qubit unitary gates, using `seed` for the RNG.
    fn random_circuit(n_qubits: u32, n_gates: usize, seed: u64) -> CircuitGraph {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cg = CircuitGraph::new();
        for _ in 0..n_gates {
            let k: u32 = if rng.gen_bool(0.5) { 1 } else { 2 };
            let k = k.min(n_qubits);
            let mut available: Vec<u32> = (0..n_qubits).collect();
            let mut targets = Vec::new();
            for _ in 0..k {
                let idx = rng.gen_range(0..available.len());
                targets.push(available.remove(idx));
            }
            targets.sort();
            let matrix = ComplexSquareMatrix::random_unitary_with_rng(1 << k, &mut rng);
            cg.insert_gate(QuantumGate::new(matrix, targets));
        }
        cg
    }

    // ═════════════════════════════════════════════════════════════════════════
    // union_qubit_count
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn union_qubit_count_disjoint() {
        assert_eq!(union_qubit_count(&[0, 1], &[2, 3]), 4);
    }

    #[test]
    fn union_qubit_count_overlapping() {
        assert_eq!(union_qubit_count(&[0, 1], &[1, 2]), 3);
    }

    #[test]
    fn union_qubit_count_identical() {
        assert_eq!(union_qubit_count(&[0, 1, 2], &[0, 1, 2]), 3);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Phase 1 structural checks (apply_size_two_fusion)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn absorbs_single_qubit_gate_into_next_row() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::x(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        apply_size_two_fusion(&mut cg);
        assert_eq!(cg.n_rows(), 1);
        assert_eq!(cg.gate_id_at(0, 0), cg.gate_id_at(0, 1));
        assert_eq!(
            cg.gate(cg.gate_id_at(0, 0).unwrap()),
            Some(&QuantumGate::cx(0, 1).matmul(&QuantumGate::x(0)))
        );
    }

    #[test]
    fn absorbs_single_qubit_gate_into_previous_row() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::x(0));
        apply_size_two_fusion(&mut cg);
        assert_eq!(cg.n_rows(), 1);
        assert_eq!(
            cg.gate(cg.gate_id_at(0, 0).unwrap()),
            Some(&QuantumGate::x(0).matmul(&QuantumGate::cx(0, 1)))
        );
    }

    #[test]
    fn leaves_isolated_single_qubit_gate_alone() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::x(0));
        apply_size_two_fusion(&mut cg);
        assert_eq!(cg.n_rows(), 1);
        assert_eq!(cg.gate(0), Some(&QuantumGate::x(0)));
    }

    #[test]
    fn fuses_adjacent_two_qubit_gates_on_same_targets() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cz(0, 1));
        apply_size_two_fusion(&mut cg);
        assert_eq!(cg.n_rows(), 1);
        assert_eq!(cg.gate_id_at(0, 0), cg.gate_id_at(0, 1));
        assert_eq!(
            cg.gate(cg.gate_id_at(0, 0).unwrap()),
            Some(&QuantumGate::cz(0, 1).matmul(&QuantumGate::cx(0, 1)))
        );
    }

    #[test]
    fn does_not_fuse_adjacent_two_qubit_gates_on_different_targets() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        apply_size_two_fusion(&mut cg);
        assert_eq!(cg.n_rows(), 2);
        assert_eq!(cg.gate(0), Some(&QuantumGate::cx(0, 1)));
        assert_eq!(cg.gate(1), Some(&QuantumGate::cx(1, 2)));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Phase 2 structural checks (apply_gate_fusion)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn fuses_chain_cx_cx_cx_to_single_four_qubit_gate() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::cx(2, 3));
        let config = FusionConfig::size_only(4);
        apply_gate_fusion(&mut cg, &config, 4, None);
        assert_eq!(cg.n_rows(), 1);
        assert_eq!(cg.gate(cg.gate_id_at(0, 0).unwrap()).unwrap().n_qubits(), 4);
        assert!(cg.check_consistency().is_ok());
    }

    #[test]
    fn respects_size_max_cap() {
        // CX(0,1) and CX(2,3) pack into the same row; CX(1,2) occupies row 1.
        // cdd_size=2: same-row fusion needs 4 qubits (rejected) and cross-row
        // fusion needs 3 qubits (rejected). Graph is unchanged.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::cx(2, 3));
        let n_before = cg.n_rows();
        let config = FusionConfig::size_only(2);
        apply_gate_fusion(&mut cg, &config, 2, None);
        assert_eq!(cg.n_rows(), n_before);
        assert!(cg.check_consistency().is_ok());
    }

    #[test]
    fn optimize_reduces_cx_chain() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::cx(2, 3));
        cg.insert_gate(QuantumGate::cx(3, 4));
        optimize(&mut cg, &FusionConfig::default());
        assert!(cg.n_rows() < 4);
        assert!(cg.check_consistency().is_ok());
    }

    #[test]
    fn rollback_when_benefit_margin_is_max() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        let config = FusionConfig {
            size_max: 6,
            benefit_margin: f64::MAX,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size: 6,
                max_ai: usize::MAX,
                zero_tol: 0.0,
            }),
        };
        let n_before = cg.n_rows();
        apply_gate_fusion(&mut cg, &config, 3, None);
        assert_eq!(cg.n_rows(), n_before);
        assert!(cg.check_consistency().is_ok());
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Phase 1 CORRECTNESS: apply_size_two_fusion preserves the circuit unitary
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn phase1_correct_single_qubit_chain() {
        // H → X → H on the same wire: should fuse to a single 1-qubit gate.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::x(0));
        cg.insert_gate(QuantumGate::h(0));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_single_qubit_before_two_qubit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::x(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_single_qubit_after_two_qubit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::h(1));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_sandwich_single_around_two_qubit() {
        // H(0) · CX(0,1) · H(0) — implements a CZ up to local phases.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::h(0));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_adjacent_two_qubit_same_targets() {
        // CX(0,1) · CZ(0,1) — same targets, fused by Phase 1.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cz(0, 1));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_cx_cx_is_identity() {
        // CX · CX = I: after fusion the matrix should be (close to) identity.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        let fused = check_fusion(&cg, apply_size_two_fusion);
        // Structural: collapsed to one gate.
        assert_eq!(fused.n_rows(), 1);
        // Mathematical: that gate is the 2×2 identity on {0,1}.
        let u = circuit_unitary(&fused);
        let eye = ComplexSquareMatrix::eye(4);
        assert!(
            u.matrix().maximum_norm_diff(&eye) < 1e-10,
            "CX·CX should be identity"
        );
    }

    #[test]
    fn phase1_correct_multi_qubit_mixed() {
        // Multi-qubit circuit with multiple 1-qubit absorptions.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::h(1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::x(2));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_rz_chain_on_single_qubit() {
        // Three RZ rotations collapse to one.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::rz(0.3, 0));
        cg.insert_gate(QuantumGate::rz(0.7, 0));
        cg.insert_gate(QuantumGate::rz(1.1, 0));
        check_fusion(&cg, apply_size_two_fusion);
    }

    #[test]
    fn phase1_correct_swap_adjacent() {
        // SWAP(0,1) · SWAP(0,1) = I.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::swap(0, 1));
        cg.insert_gate(QuantumGate::swap(0, 1));
        let fused = check_fusion(&cg, apply_size_two_fusion);
        assert_eq!(fused.n_rows(), 1);
        let u = circuit_unitary(&fused);
        assert!(u.matrix().maximum_norm_diff(&ComplexSquareMatrix::eye(4)) < 1e-10);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Phase 2 CORRECTNESS: apply_gate_fusion preserves the circuit unitary
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn phase2_correct_cx_chain_3qubit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(3), 3, None);
        });
    }

    #[test]
    fn phase2_correct_cx_chain_4qubit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::cx(2, 3));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(4), 4, None);
        });
    }

    #[test]
    fn phase2_correct_parallel_then_bridge() {
        // Row 0: CX(0,1), CX(2,3).  Row 1: CX(1,2) bridges them.
        // With cdd_size=4, all three can fuse.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(2, 3));
        cg.insert_gate(QuantumGate::cx(1, 2));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(4), 4, None);
        });
    }

    #[test]
    fn phase2_correct_h_cx_bell_state_circuit() {
        // H(0) then CX(0,1): the canonical Bell-state preparation.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(2), 3, None);
        });
    }

    #[test]
    fn phase2_correct_toffoli_chain() {
        // Two Toffoli gates: CCX(0,1,2) then CCX(1,2,3).
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::ccx(0, 1, 2));
        cg.insert_gate(QuantumGate::ccx(1, 2, 3));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(4), 4, None);
        });
    }

    #[test]
    fn phase2_correct_mixed_rz_cx_rz() {
        // RZ–CX–RZ pattern common in ZZ-rotation circuits.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::rz(0.4, 0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::rz(-0.4, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &FusionConfig::size_only(3), 3, None);
        });
    }

    // ═════════════════════════════════════════════════════════════════════════
    // optimize() CORRECTNESS: full pass (Phase 1 + Phase 2)
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn optimize_correct_bell_circuit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_ghz_3qubit() {
        // H(0) → CX(0,1) → CX(0,2): standard 3-qubit GHZ preparation.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(0, 2));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_ghz_4qubit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(0, 2));
        cg.insert_gate(QuantumGate::cx(0, 3));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_brick_wall_4qubit() {
        // Two layers of brick-wall two-qubit gates on 4 qubits.
        // Layer 1: CX(0,1), CX(2,3).  Layer 2: CX(1,2).  Layer 3: CX(0,1), CX(2,3).
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(2, 3));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(2, 3));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_brick_wall_5qubit_two_layers() {
        // Full two-layer brick wall on 5 qubits.
        let mut cg = CircuitGraph::new();
        // Layer 1
        for &(a, b) in &[(0u32, 1u32), (2, 3)] {
            cg.insert_gate(QuantumGate::cx(a, b));
        }
        // Layer 2
        for &(a, b) in &[(1u32, 2u32), (3, 4)] {
            cg.insert_gate(QuantumGate::cx(a, b));
        }
        // Layer 3 (same as layer 1)
        for &(a, b) in &[(0u32, 1u32), (2, 3)] {
            cg.insert_gate(QuantumGate::cx(a, b));
        }
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_parametric_zz_rotation() {
        // ZZ-rotation decomposition: RZ, CX, RZ, CX.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::rz(0.6, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_rx_ry_rz_cx_circuit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::rx(0.3, 0));
        cg.insert_gate(QuantumGate::ry(0.5, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::rz(0.7, 0));
        cg.insert_gate(QuantumGate::rx(1.1, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::ry(0.2, 0));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_u3_and_cx() {
        use std::f64::consts::PI;
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::u3(PI / 3.0, PI / 5.0, PI / 7.0, 0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::u3(PI / 4.0, PI / 6.0, PI / 8.0, 1));
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_toffoli_involved() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(2));
        cg.insert_gate(QuantumGate::ccx(0, 1, 2));
        cg.insert_gate(QuantumGate::h(2));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_swap_circuit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::swap(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        cg.insert_gate(QuantumGate::swap(1, 2));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_single_gate_noop() {
        // A single gate: fusion should be a no-op but unitary still correct.
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn optimize_correct_all_single_qubit_circuit() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::x(1));
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::z(1));
        cg.insert_gate(QuantumGate::rz(0.4, 0));
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Rollback CORRECTNESS: benefit_margin = f64::MAX must not change the unitary
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn rollback_correct_no_change_to_unitary() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::h(0));
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        let config = FusionConfig {
            size_max: 6,
            benefit_margin: f64::MAX,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size: 6,
                max_ai: usize::MAX,
                zero_tol: 0.0,
            }),
        };
        // Phase 2 rollback: unitary must be unchanged.
        check_fusion(&cg, |g| {
            apply_gate_fusion(g, &config, 6, None);
        });
    }

    #[test]
    fn rollback_correct_circuit_preserved_structurally() {
        let mut cg = CircuitGraph::new();
        cg.insert_gate(QuantumGate::cx(0, 1));
        cg.insert_gate(QuantumGate::cx(1, 2));
        let config = FusionConfig {
            size_max: 6,
            benefit_margin: f64::MAX,
            cost_model: Box::new(SizeOnlyCostModel {
                max_size: 6,
                max_ai: usize::MAX,
                zero_tol: 0.0,
            }),
        };
        let n_before = cg.n_rows();
        let fused = check_fusion(&cg, |g| {
            apply_gate_fusion(g, &config, 3, None);
        });
        // Row count unchanged (rollback restored all gates).
        assert_eq!(fused.n_rows(), n_before);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Config comparison: different configs must yield the same unitary
    // ═════════════════════════════════════════════════════════════════════════

    #[test]
    fn all_configs_yield_same_unitary() {
        // Run the same circuit through every built-in config and assert the
        // unitary is identical across all of them.
        let mut cg_base = CircuitGraph::new();
        cg_base.insert_gate(QuantumGate::h(0));
        cg_base.insert_gate(QuantumGate::cx(0, 1));
        cg_base.insert_gate(QuantumGate::cx(1, 2));
        cg_base.insert_gate(QuantumGate::rz(0.5, 2));
        cg_base.insert_gate(QuantumGate::cx(1, 2));
        cg_base.insert_gate(QuantumGate::cx(0, 1));

        let configs: &[(&str, FusionConfig)] = &[
            ("size_only(3)", FusionConfig::size_only(3)),
            ("size_only(4)", FusionConfig::size_only(4)),
            ("default", FusionConfig::default()),
            ("aggressive", FusionConfig::aggressive()),
        ];

        let u_ref = circuit_unitary(&cg_base);

        for (name, config) in configs {
            let mut cg = cg_base.clone();
            optimize(&mut cg, config);
            let u = circuit_unitary(&cg);
            assert_eq!(
                u.qubits(),
                u_ref.qubits(),
                "config {name}: qubit sets differ"
            );
            let diff = u_ref.matrix().maximum_norm_diff(u.matrix());
            assert!(
                diff < 1e-10,
                "config {name}: unitary differs (max diff = {diff:.2e})"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Random circuit CORRECTNESS (the most robust correctness signal)
    // ═════════════════════════════════════════════════════════════════════════

    // Each test: generate a reproducible random circuit, run optimize() with
    // balanced config, assert the unitary is preserved.

    #[test]
    fn random_3qubit_10gates_seed0() {
        let cg = random_circuit(3, 10, 0);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_3qubit_10gates_seed1() {
        let cg = random_circuit(3, 10, 1);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_4qubit_15gates_seed0() {
        let cg = random_circuit(4, 15, 0);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_4qubit_15gates_seed1() {
        let cg = random_circuit(4, 15, 1);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_4qubit_15gates_seed2() {
        let cg = random_circuit(4, 15, 2);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_4qubit_20gates_seed42() {
        let cg = random_circuit(4, 20, 42);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_5qubit_20gates_seed0() {
        let cg = random_circuit(5, 20, 0);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_5qubit_20gates_seed7() {
        let cg = random_circuit(5, 20, 7);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_5qubit_30gates_seed99() {
        let cg = random_circuit(5, 30, 99);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    // Same circuits through mild and aggressive to cover different size limits.

    #[test]
    fn random_4qubit_20gates_mild() {
        let cg = random_circuit(4, 20, 17);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::default()));
    }

    #[test]
    fn random_4qubit_20gates_aggressive() {
        let cg = random_circuit(4, 20, 17);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::aggressive()));
    }

    #[test]
    fn random_5qubit_20gates_size_only_3() {
        let cg = random_circuit(5, 20, 123);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::size_only(3)));
    }

    #[test]
    fn random_5qubit_20gates_size_only_5() {
        let cg = random_circuit(5, 20, 123);
        check_fusion(&cg, |g| optimize(g, &FusionConfig::size_only(5)));
    }
}
