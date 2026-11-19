OPENQASM 3.0;
include "stdgates.inc";
qubit[8] q;
bit[8] c;

// Prepare |+>^{\otimes n}
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];

// Diagonal hypergraph phase oracle U_f
cz q[0], q[1];
cz q[2], q[3];
ctrl(3) @ z q[4], q[5], q[6], q[7];

// Final Hadamard layer H^{\otimes n} and X-basis measurement
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
c[4] = measure q[4];
c[5] = measure q[5];
c[6] = measure q[6];
c[7] = measure q[7];
