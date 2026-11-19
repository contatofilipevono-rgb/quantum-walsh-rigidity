OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
bit[6] c;

// Prepare |+>^{\otimes n}
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];

// Diagonal hypergraph phase oracle U_f
cz q[0], q[3];
cz q[1], q[4];
cz q[2], q[5];
ctrl(2) @ z q[3], q[4], q[5];

// Final Hadamard layer H^{\otimes n} and X-basis measurement
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
