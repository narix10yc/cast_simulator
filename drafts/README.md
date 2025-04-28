## CAST Quantum Circuit Description Language
### Grammer
```

```

### Keywords
All keywords start in captial letter. Keywords include `Circuit`, `Channel`, `All`, `If`, `Repeat`, and standard gates.

### Standard Gates
- All captical letters.
- Builtin support includes `X`, `Y`, `Z`, `H`, `CX`, `CY`, `CZ`, `RX`, `RY`, `RZ`.
- Optionally be followed by square brakets `[]` for attributes. For example, `X[phase=pi]` means the $-X$ gate.

### Top-level Statements
```
top_level_stmt ::= circuit_stmt | channel_stmt | gate_stmt ;
```

### Circuit
Grammer:
```
`Circuit`
```

Example:
```
Circuit my_circuit {
  X 0;
  H 1;
  RZ(Pi) 2;
}
```
defines a circuit with 3 gates.

Gate fusion is denoted by `@`. For example,
```
Circuit my_circuit2 {
  X 0
@ Y 0;
  RZ(pi) 2;
}
```
Because $YX=-iZ$, this circuit is equivalent to
```
Circuit<phase=-Pi/4> {
  Z 0;
  RZ(pi) 2;
}
```

### Parameter
We support parameter by `#n`
```
Circuit my_circuit3 {
  RX(#0) 0;
}
```
This circuit is likely compiled to
```
Circuit[nqubits=1, nparams=1] {
  RX(#0) 0;
}
```

## Gate

## Channel
We define quantum channels by the `Channel` keyword. For example, to define a symmetric depolarizing channel with strength `p`, we write
```
Channel symmetric_depolarizing(p) {
  X p/3;
  Y p/3;
  Z p/3;
}
```
Similarly, we can define a general single-qubit channel
```
Channel single_qubit_noise(px, py, pz) {
  X px;
  Y py;
  Z pz;
}
```

The channel body must be a list of `pauli_component_stmt`.
```
pauli_component_stmt ::= pauli_string expr ";" ;
```

## Measurements and conditionals
```
Circuit[noise=symmetric_depolarizing(0.1)]
my_noisy_circuit {
  H 0;
  // Reset qubit 0 to 0
  If (Measure 0)
    X 0;
  H 1;
  Measure 1;
}
```

```
Gate reset(q) {
  If (Measure q)
    X q;
}

Circuit qc {
  Call reset(0);
}
```

### TODO: Convenient Grammer
```
Gate[nqubits=2, nparams=4] block_gate(a, b) {
  RX(#.) a;
  RX(#.) b;
  RZ(#.) a;
  RZ(#.) b;
  CX a b;
}

Circuit[nqubits=4, nparams=20] block_circuit {
  H All;
  Repeat (5) {
    block_gate(0, 1);
    block_gate(2, 3);
    block_gate(1, 2);
    block_gate(3, 0);
  }
  H All;
}
```

- Range operator `...` and `..<`
- List operator `[]`
```
Circuit qc {
  H 0...9;
  CX 0 1..<10;
  Measure [1, 2, 3];
}
```

### TODO: Error Correction Code
```
// Auto-deduce this is a [[5,1,3]] code.
ECC five_qubit_code {
  XZZXI;
  IXZZX;
  XIXZZ;
  ZXIXZ;
}

// Auto-deduce this circuit has 5 physical qubits
Circuit[ecc=five_qubit_code] ecc_circuit {
  H 0;
  Measure 0;
}
```