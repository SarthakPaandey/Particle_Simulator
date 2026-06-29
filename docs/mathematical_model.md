# Mathematical Model

## Beam State

The beam is represented by a 2D state vector in one transverse plane:

$$X = \begin{pmatrix} x \\ x' \end{pmatrix}$$

where:
- $x$ = horizontal beam position [m]
- $x'$ = horizontal beam divergence [rad]

## Transfer Matrices

Each lattice element transforms the state:

$$X_{k+1} = M_k \, X_k$$

### Drift (length $L$)

$$M_{\text{drift}}(L) = \begin{pmatrix} 1 & L \\ 0 & 1 \end{pmatrix}$$

A particle travelling a distance $L$ with angle $x'$ gains position $\Delta x = L \, x'$; angle is unchanged.

### Thin-Lens Quadrupole

**Focusing** (focal length $f$):

$$M_{\text{QF}}(f) = \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$$

**Defocusing**:

$$M_{\text{QD}}(f) = \begin{pmatrix} 1 & 0 \\ 1/f & 1 \end{pmatrix}$$

Position is unchanged (thin lens); angle receives a kick proportional to position: $\Delta x' = \mp x/f$.

### Corrector Magnet

$$X_{\text{after}} = X_{\text{before}} + \begin{pmatrix} 0 \\ \theta \end{pmatrix}$$

A corrector adds a fixed angle $\theta$ (the "kick") independent of position.

## Orbit Errors

Errors are modelled as random angular kicks at lattice elements:

$$\varepsilon \sim \mathcal{N}(0,\, \sigma_{\text{error}}^2)$$

$$X_{\text{after}} = X_{\text{before}} + \begin{pmatrix} 0 \\ \varepsilon \end{pmatrix}$$

## BPM Measurement

$$x_{\text{BPM}} = x_{\text{true}} + \eta, \qquad \eta \sim \mathcal{N}(0,\, \sigma_{\text{BPM}}^2)$$

## Response Matrix

For $N$ BPMs and $M$ correctors, the response matrix $R \in \mathbb{R}^{N \times M}$ is:

$$R_{ij} = \frac{\partial x_i}{\partial \theta_j} \approx \frac{x_i(\theta_j + \Delta\theta) - x_i(\theta_j)}{\Delta\theta}$$

## Correction Problem

We seek $c$ such that $R\,c \approx -b$, i.e.:

$$\min_c \|R\,c + b\|_2^2$$

### Least-Squares Solution

$$c^* = (R^\top R)^{-1} R^\top (-b) = -R^+ b$$

Implemented via `np.linalg.lstsq`.

### SVD Solution

Decompose $R = U \Sigma V^\top$, then the pseudo-inverse is:

$$R^+ = V \Sigma^+ U^\top$$

Small singular values can be truncated:

$$\frac{1}{\sigma_i} \to 0 \quad \text{if } \sigma_i < \text{cutoff}$$

### Iterative Feedback

$$c_{k+1} = c_k + \alpha \, \Delta c_k$$

$$\text{stop when } \text{RMS} < \text{tolerance or } k = k_{\max}$$

## Performance Metrics

$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$$

$$\text{Improvement} = 100 \times \frac{\text{RMS}_{\text{before}} - \text{RMS}_{\text{after}}}{\text{RMS}_{\text{before}}}$$
