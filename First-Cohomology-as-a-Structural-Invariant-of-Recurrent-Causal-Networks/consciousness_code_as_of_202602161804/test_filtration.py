
"""
test_filtration.py - Verification of Theorem A (DAG ⇔ Strict Filtration)
========================================================================

Theorem A states that a strict filtration presheaf (with strictly increasing
dimensions along causal paths) can only be defined on a Directed Acyclic Graph.
Any cycle creates a contradiction in the dimension function d(x) < ... < d(x).

This test verifies:
1. DAG Construction: Successfully creates a FiltrationPresheaf on a feedforward network.
2. Cycle Obstruction: Fails to create a valid FiltrationPresheaf on a cycle.
"""

import sys
import os
# Ensure we import from local source, not installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from anomalon_kernel.domain.invariants.catkit.causal_set import CausalSet, CausalEvent
from anomalon_kernel.domain.invariants.catkit.strict_filtration import StrictFiltrationPresheaf as FiltrationPresheaf
from anomalon_kernel.domain.invariants.catkit.consciousness import build_feedforward, build_recurrent

def test_dag_filtration_success():
    """Theorem A (Forward): A DAG admits a strict filtration."""
    print("\n[Test] DAG Filtration Construction")
    # Build a small feedforward network (DAG)
    # 3 layers, width 2
    dag = build_feedforward(n_layers=3, width=2)
    
    # Attempt to create FiltrationPresheaf
    fp = FiltrationPresheaf(dag)
    
    # Verify no cycles detected
    assert not fp.has_cycles(), "DAG should not have cycles"
    
    # Verify dimensions are assigned and increasing
    depths = fp._compute_depths()
    print(f"  Depths: {depths}")
    
    # Check strict growth along edges
    for a, b in fp.covering_edges():
        da = fp.dimension(a)
        db = fp.dimension(b)
        print(f"  Edge {a.label} -> {b.label}: dim {da} -> {db}")
        assert da < db, f"Strict growth violated: {da} >= {db} at {a.label}->{b.label}"
        
    print("  => Success: Filtration defined on DAG.")

def test_cycle_obstruction():
    """Theorem A (Reverse): A cycle obstructs strict filtration."""
    print("\n[Test] Cycle Obstruction")
    
    # Manually build a simple 3-cycle: 0->1->2->0
    # build_recurrent is probabilistic, so let's be explicit
    events = [CausalEvent(i, f"node_{i}") for i in range(3)]
    # Relations must be transitive for CausalSet
    # But a cycle implies everything precedes everything in the cycle
    # CausalSet usually expects a poset (antisymmetric). 
    # If we force a cycle in 'relations', CausalSet might behave strictly or leniently.
    # Let's try to construct it.
    
    relations = set()
    # Cycle 0->1, 1->2, 2->0
    relations.add((events[0].label, events[1].label))
    relations.add((events[1].label, events[2].label))
    relations.add((events[2].label, events[0].label))
    
    # Transitive closure for the cycle: everything connects to everything
    relations.add((events[0].label, events[2].label)) # 0->2 via 1
    relations.add((events[1].label, events[0].label)) # 1->0 via 2
    relations.add((events[2].label, events[1].label)) # 2->1 via 0
    # Reflexive
    for i in range(3):
        relations.add((events[i].label, events[i].label))
        
    cycle_cs = CausalSet(events=tuple(events), relations=tuple(relations))
    
    # Attempt to create FiltrationPresheaf
    fp = FiltrationPresheaf(cycle_cs)
    
    # It should detect cycles
    has_cycles = fp.has_cycles()
    print(f"  Cycle detected: {has_cycles}")
    assert has_cycles, "Should detect cycle in 0->1->2->0"
    
    # Depth calculation should return -1 for cyclic nodes
    depths = fp._compute_depths()
    print(f"  Depths: {depths}")
    assert depths[0] == -1, "Depth should be undefined (-1) for cyclic node"
    
    # Dimensions should fallback to base_dim (no growth)
    d0 = fp.dimension(events[0])
    d1 = fp.dimension(events[1])
    print(f"  Dimensions: {d0}, {d1}")
    assert d0 == fp.d_base
    assert d1 == fp.d_base
    
    # This implies H^1 is degenerate (identity maps), NOT a strict filtration
    # We conceptually fail to define the 'Strict' filtration.
    print("  => Success: Cycle identified as obstruction.")

if __name__ == "__main__":
    test_dag_filtration_success()
    test_cycle_obstruction()
