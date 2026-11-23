OPENQASM 3.0;
include "stdgates.inc";
// IQM Garnet (Crystal-20) Physically Routed v4 | iqm20_flat_cccz | X-BASIS FLATNESS (H U_f H |0^20>)
// Physical Qubit Map: q[0]=QB1, q[1]=QB2, q[2]=QB3, q[3]=QB4, q[4]=QB5, q[5]=QB6, q[6]=QB7, q[7]=QB8, q[8]=QB9, q[9]=QB10, q[10]=QB11, q[11]=QB12, q[12]=QB13, q[13]=QB14, q[14]=QB15, q[15]=QB16, q[16]=QB17, q[17]=QB18, q[18]=QB19, q[19]=QB20
// Steiner-Star/Path Synthesis: 0 SWAPs, 26 native CZs (100% adjacent)
qubit[20] q;
bit[20] c;

// Prepare |+>^{otimes 20}
ry(1.570796326795) q[0]; // QB1
ry(1.570796326795) q[1]; // QB2
ry(1.570796326795) q[2]; // QB3
ry(1.570796326795) q[3]; // QB4
ry(1.570796326795) q[4]; // QB5
ry(1.570796326795) q[5]; // QB6
ry(1.570796326795) q[6]; // QB7
ry(1.570796326795) q[7]; // QB8
ry(1.570796326795) q[8]; // QB9
ry(1.570796326795) q[9]; // QB10
ry(1.570796326795) q[10]; // QB11
ry(1.570796326795) q[11]; // QB12
ry(1.570796326795) q[12]; // QB13
ry(1.570796326795) q[13]; // QB14
ry(1.570796326795) q[14]; // QB15
ry(1.570796326795) q[15]; // QB16
ry(1.570796326795) q[16]; // QB17
ry(1.570796326795) q[17]; // QB18
ry(1.570796326795) q[18]; // QB19
ry(1.570796326795) q[19]; // QB20

// Physically routed U_f on IQM Garnet couplers
cz q[0], q[1]; // QB1-QB2
cz q[2], q[6]; // QB3-QB7
cz q[3], q[4]; // QB4-QB5
cz q[5], q[10]; // QB6-QB11
cz q[7], q[12]; // QB8-QB13
cz q[13], q[8]; // QB14-QB9
cz q[9], q[14]; // QB10-QB15
cz q[11], q[16]; // QB12-QB17
cz q[18], q[19]; // QB19-QB20
rz(0.392699081699) q[10];
rz(0.392699081699) q[14];
rz(0.392699081699) q[16];
rz(0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[14], q[15]; // QB15-QB16
ry(-1.570796326795) q[15];
rz(0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[16], q[15]; // QB17-QB16
ry(-1.570796326795) q[15];
rz(0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[14], q[15]; // QB15-QB16
ry(-1.570796326795) q[15];
rz(0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[16];
cz q[15], q[16]; // QB16-QB17
ry(-1.570796326795) q[16];
ry(1.570796326795) q[15];
cz q[16], q[15]; // QB17-QB16
ry(-1.570796326795) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[14], q[15]; // QB15-QB16
ry(-1.570796326795) q[15];
rz(0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[14];
cz q[15], q[14]; // QB16-QB15
ry(-1.570796326795) q[14];
ry(1.570796326795) q[15];
cz q[14], q[15]; // QB15-QB16
ry(-1.570796326795) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
rz(-0.392699081699) q[15];
ry(1.570796326795) q[15];
cz q[10], q[15]; // QB11-QB16
ry(-1.570796326795) q[15];
cz q[17], q[16]; // QB18-QB17

// Final Hadamard layer and logical-mapped measurement
ry(-1.570796326795) q[0];
ry(-1.570796326795) q[1];
ry(-1.570796326795) q[2];
ry(-1.570796326795) q[3];
ry(-1.570796326795) q[4];
ry(-1.570796326795) q[5];
ry(-1.570796326795) q[6];
ry(-1.570796326795) q[7];
ry(-1.570796326795) q[8];
ry(-1.570796326795) q[9];
ry(-1.570796326795) q[10];
ry(-1.570796326795) q[11];
ry(-1.570796326795) q[12];
ry(-1.570796326795) q[13];
ry(-1.570796326795) q[14];
ry(-1.570796326795) q[15];
ry(-1.570796326795) q[16];
ry(-1.570796326795) q[17];
ry(-1.570796326795) q[18];
ry(-1.570796326795) q[19];
c[0] = measure q[0]; // logical 0 -> physical QB1
c[1] = measure q[1]; // logical 1 -> physical QB2
c[2] = measure q[2]; // logical 2 -> physical QB3
c[3] = measure q[3]; // logical 3 -> physical QB4
c[4] = measure q[4]; // logical 4 -> physical QB5
c[5] = measure q[5]; // logical 5 -> physical QB6
c[6] = measure q[6]; // logical 6 -> physical QB7
c[7] = measure q[7]; // logical 7 -> physical QB8
c[8] = measure q[8]; // logical 8 -> physical QB9
c[9] = measure q[9]; // logical 9 -> physical QB10
c[10] = measure q[10]; // logical 10 -> physical QB11
c[11] = measure q[11]; // logical 11 -> physical QB12
c[12] = measure q[12]; // logical 12 -> physical QB13
c[13] = measure q[13]; // logical 13 -> physical QB14
c[14] = measure q[15]; // logical 14 -> physical QB16
c[15] = measure q[16]; // logical 15 -> physical QB17
c[16] = measure q[14]; // logical 16 -> physical QB15
c[17] = measure q[17]; // logical 17 -> physical QB18
c[18] = measure q[18]; // logical 18 -> physical QB19
c[19] = measure q[19]; // logical 19 -> physical QB20
