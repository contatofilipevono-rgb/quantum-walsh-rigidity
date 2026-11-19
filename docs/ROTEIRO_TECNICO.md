# Roteiro Técnico: Rigidez Modular de Walsh e Necessidade Antipodal em Computação Quântica

## 1. Dicionário Matemática Booleana $\leftrightarrow$ Computação Quântica

| Objeto Booleano | Objeto em Computação Quântica |
|---|---|
| Função booleana $f : \mathbb{F}_2^n \to \mathbb{F}_2$ de grau $d$ | Oráculo diagonal de fase $U_f = \text{diag}((-1)^{f(x)})$ no $d$-ésimo nível da hierarquia de Clifford ($\mathcal{C}_d$) |
| Monômio de grau $k$ na ANF de $f$ | Porta multi-controlada $C^{k-1}Z$ ($k=1$: $Z$, $k=2$: $CZ$, $k=3$: $CCZ$, $k=4$: $CCCZ$) |
| Estado $\|\psi_f\rangle = U_f H^{\otimes n}\|0^n\rangle$ | **Estado hipergrafo** (*hypergraph state*) de grau $\le 4$ |
| Transformada de Walsh $W_f(b)$ | Amplitude na base $X$: $\langle b \| H^{\otimes n} \| \psi_f \rangle = W_f(b) / 2^n$ |
| $f$ é **bent** ($\|W_f(b)\| = 2^{n/2}\;\forall b$) | **Estado hipergrafo plano** (*flat hypergraph state*): distribuição de medição na base $X$ exatamente uniforme ($2^{-n}$) |
| Simetria rotacional ($f(\sigma(x)) = f(x)$) | Circuito / *ansatz* invariante por translação em hardware com topologia em **anel** ($C_n$) |

---

## 2. Teoremas Aplicados (Corolários Diretos dos Preprints)

### Teorema A — Obstrução Antipodal em Arquiteturas em Anel
*Origem:* `universal_antipodal_walsh_absolute` (DOI `10.6084/m9.figshare.34165260.v2`, certificado em Lean 4).

Seja $|\psi_f\rangle$ um estado hipergrafo em $n = 2H$ qubits ($H \ge 1$) preparado por um circuito diagonal invariante por rotação com portas em $\{Z, CZ, CCZ\}$ (grau $\le 3$). Se $|\psi_f\rangle$ tem espectro plano na base $X$ (isto é, $f$ é *bent*), então o circuito **contém obrigatoriamente toda a órbita antipodal de portas** $\{CZ_{i,\,i+H} : 0 \le i < H\}$.

**Consequências práticas para hardware e compilação:**
1. **Impossibilidade local em anel:** nenhum *ansatz* rotacionalmente simétrico de grau $\le 3$ com alcance de interação $r < H$ (ex.: vizinhos próximos ou alcance limitado em um anel de qubits) pode preparar um estado hipergrafo plano.
2. **Impossibilidade puramente cúbica:** circuitos rotacionalmente simétricos compostos apenas por portas $CCZ$ (sem camada $CZ$) nunca geram espectro plano, mesmo com conectividade total.
3. **Custo mínimo de roteamento (SWAPs) em anel:** em um anel de $n = 2H$ qubits com conectividade apenas entre vizinhos próximos, realizar um acoplamento antipodal $CZ_{i,\,i+H}$ exige distância topológica $H$, impondo uma sobrecarga mínima de $\Omega(H)$ camadas de SWAPs para qualquer compilação fiel.

---

### Teorema B — Certificação Modular de *Flatness* para Circuitos $\{Z, CZ, CCZ, CCCZ\}$
*Origem:* Teorema uniforme $8/15$ ($m \ge 4$) e teorema assintótico $1/2$ ($m \ge m_0$) (DOI `10.6084/m9.figshare.34963642.v4`).

Seja $U_f$ um circuito diagonal com portas em $\{Z, CZ, CCZ, CCCZ\}$ em $n = 4m$ qubits ($m \ge 4$, $M = 2^m$). Seja $p$ um inteiro ímpar satisfazendo:
- $p \ge \frac{8}{15} M$ (válido para **todo** $m \ge 4$), ou
- $p \ge \frac{1}{2} M$ (válido para $m \ge m_0$).

Então $|\psi_f\rangle$ é um estado hipergrafo plano **se, e somente se**, para toda frequência $b \in \mathbb{F}_2^{4m}$:
$$W_f(b) \equiv \pm M^2 \pmod p.$$

**Consequências práticas para verificação de circuitos:**
1. **Redução de precisão pela metade (em bits):** a verificação exata exige inteiros até $\pm 2^n$ (ou $\pm 2^{n/2}$ após normalização, isto é, $\approx n/2$ bits). A verificação modular precisa apenas de aritmética módulo $p \approx 2^{n/4}$ ($\approx n/4$ bits — metade da largura de palavra).
2. **Módulos concretos imediatos:**
   - $n = 16$ qubits ($m = 4, M = 16$): **$p = 9$** (4 bits; igual para $8/15$ e $1/2$).
   - $n = 20$ qubits ($m = 5, M = 32$): **$p = 19$** (5 bits; $8/15$).
   - $n = 24$ qubits ($m = 6, M = 64$): **$p = 35$** (6 bits; $8/15$).
   - $n = 28$ qubits ($m = 7, M = 128$): **$p = 69$** (7 bits; $8/15$).
3. **Detecção garantida de falhas de compilação (0 falsos positivos):** qualquer erro de compilação/otimização (omissão, inserção ou troca de portas $\{Z, CZ, CCZ, CCCZ\}$) que destrua a planura do espectro é **matematicamente garantido** de violar $W_f(b) \equiv \pm M^2 \pmod p$ em pelo menos uma frequência $b$.

---

## 3. Plano de Avanço Rápido (2 Semanas)

- **Semana 1 (Validação numérica — já iniciada em `experiments/run_experiments.py`):**
  1. Benchmark de certificação modular $p = 9$ ($n = 16$) e $p = 19$ ($n = 20$) sob injeção de falhas em circuitos $\{CZ, CCZ, CCCZ\}$ (comparando taxa de detecção de $p \ge 8M/15$ vs. módulos menores como $p = 3, 5, 7$).
  2. Verificação da obstrução antipodal em anel ($n = 6, 8, 10, 12$) e tabela de sobrecarga de SWAPs em compilação para anel.
  3. Caracterização do *magic* (Stabilizer Rényi Entropy $M_2$) dos estados hipergrafo planos de grau 3 e 4 vs. estados grafo (Clifford, $M_2 = 0$).
- **Semana 2 (Redação e submissão):**
  1. Redigir manuscrito conciso em inglês (8–12 páginas) com código aberto reproduzível.
  2. Publicar preprint no arXiv (`quant-ph`) / Figshare e submeter para *Quantum Information Processing*, *Physical Review A* ou *IEEE Transactions on Quantum Engineering*.
