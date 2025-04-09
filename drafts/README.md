## CAST Quantum Circuit Description Language

### Basic Gates
- All captical letters.
- Builtin support includes `X`, `Y`, `Z`, `H`, `CX`, `CY`, `CZ`, `RX`, `RY`, `RZ`.
- Optionally be followed by square brakets `[]` for attributes. For example, `X[phase=pi]` means the $-X$ gate.

### Circuit
Example:
```
circuit my_circuit {
  X[phase=pi] 0;
  H 1;
  RZ(pi) 2;
}
```
This circuit will be compiled to
```
circuit[nqubits=3, nparams=0, phase=pi] my_circuit {
  X 0;
  H 1;
  RZ(pi) 2;
}
```

Gate fusion is denoted by `@`. For example,
```
circuit my_circuit2 {
  X 0
@ Y 0;
  RZ(pi) 2;
}
```
This circuit is likely compiled to
```
circuit[nqubits=3, nparams=0, phase=pi/4] {
  Z 0;
  RZ(pi) 2;
}
```

We support parameter by `#n`
```
circuit my_circuit3 {
  RX(#0) 0;
}
```
This circuit is likely compiled to
```
circuit[nqubits=1, nparams=1] {
  RX(#0) 0;
}
```
