# EIP-1559 simulations
Simulations of Ethereum's EIP-1559 Transaction Fee Market

This repository contains the code that was used to generate the bifurcation diagrams (Figure 3) in the paper 

- Dynamical Analysis of the EIP-1559 Ethereum Fee Market: https://dl.acm.org/doi/10.1145/3479722.3480993 and https://arxiv.org/abs/2102.10567.

There are two notebooks

- **main_EIP1559_quotient.ipynb**: this notebook has the learning rate, d, (currently set by default at d = 0.125), also called step-size or adjustment quotient, as its main bifurcation parameter. One may vary the distribution of valuations between normal, gamma and uniform and the update rule between the standard EIP-1559 (called linear in the notebook), the exponential version and the AMM mechanism. The plot in the paper (left panel of Figure 3) was generated with uniform demand distribution with mean μ = 210 and w = 20. Chaotic updates are observed for larger step-sizes relative to the observed demand.
- **main_EIP1559_valuations.ipynb**: this notebook has the range of valuations, w, as its bifurcation parameter. The plot in the paper (right panel of Figure 3) uses d = 0.125 and a uniform distribution with w as shown in the horizontal axis.

The notebooks have been updated to generate average base fee trajectories and block sizes reflecting ongoing work.
