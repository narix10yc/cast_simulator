## CAST Quantum Circuit Description Language
### Grammer
```

```

### Keywords
All keywords start in captial letter. Keywords include `Circuit`, `Channel`, `All`, `If`, `Repeat`, and standard gates.

### Standard Gates
- All captical letters.
- Builtin support includes `X`, `Y`, `Z`, `H`, `CX`, `CY`, `CZ`, `RX`, `RY`, `RZ`.
- Parameters should follow round brackets. For example, `RX(-Pi) 0` means applying a RX gate with angle $-\pi$ to qubit 0.

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

### Attributes
Attributes are key-value pairs surrounded by angle brackets `<>`. Circuits and gates can have attributes. For example, `Circuit<phase=Pi>` defines a circuit with global phase $e^{i\pi}$.

Here is a list of supported attributes:
- Circuit supports `nqubits`, `nparams`, `phase`, `noise` attributes.
- Gate supports `noise` attribute.

## Channel
Grammer
```
channel_stmt ::= "Channel" identifier 
[ parameter_decl_expr ] 
"{" 
{ pauli_component_stmt }
"}"
```

We define quantum channels by the `Channel` keyword. For example, to define a symmetric depolarizing channel with strength `p`, we write
```
Channel sdc(p) {
  X p/3;
  Y p/3;
  Z p/3;
}
```
Similarly, we can define a two-qubit channel
```
Channel two_qubit_noise(pxx, pyy, pzz) {
  XX pxx;
  YY pyy;
  ZZ pzz;
}
```

The channel body must be a list of `pauli_component_stmt`.
```
pauli_component_stmt ::= pauli_string expr ";" ;
```

`pauli_string` is a string of `X` `Y` `Z` optionally followed by numbers that specify which qubit to act on. For example, `XIIYI` is equivalent to `X4Y1`. To adjust the size of the Pauli string, we could add a dummy `I<n>` term. For example, the 4-qubit `IIXI` could be written as `I3X1`.

To use channels in circuit definition, use `<noise=...>` as an attribute to the gates. For example,
```
Circuit qc {
  H<noise=sdc(0.01)> 0;
}
```

We can also impose a global error model by using the `noise=...` attribute after the `Circuit` keyword.
```
Circuit<noise=sdc(0.01)>
my_noisy_circuit {
  H 0;
  CX 0 1;
}
```
Notice that for multi-qubit gates, `Circuit<noise=...>` will add indepedent noise to every target qubit. For example, the above circuit is equivalent to
```
Circuit my_noisy_circuit {
  H 0;
  I<noise=sdc(0.01)> 0;
  CX 0 1;
  I<noise=sdc(0.01)> 0;
  I<noise=sdc(0.01)> 1;
}
```

### TODO: Measurements and Conditionals

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

## CAST IR

```
cast.circuit @my_qc(%q : cast.qubits<6>) -> i1
{
  cast.circuit_graph;
  cast.if (cast.measure(0))
  {
    cast.circuit_graph;
  } // else
  {}
  cast.circuit_graph;
  cast.out(0);
}
```

```
cast.circuit @my_qc(%q : cast.qubits<6>) -> cast.dm<6>
{
  cast.circuit_graph;
  cast.if (cast.measure(0))
  {
    cast.circuit_graph;
  } // else
  {}
  cast.circuit_graph;
  cast.out_dm();
}
```

```
Circuit my_circuit {
  H 0;
  CX 0 1;
  If (Measure 0) {
    X 0;
  }
  RZ(Pi/4) 0;
}
```

```
cast.circuit @my_circuit(%q : cast.qubits<2>) -> i1
{
  cast.circuit_graph;
  cast.if_measure(0) {
    cast.circuit_graph
  }{}
  cast.circuit_graph;
  return cast.out_measure(0);
}
```