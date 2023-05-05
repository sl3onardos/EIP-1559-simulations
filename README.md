# EIP-1559 simulations
Simulations of Ethereum's EIP-1559 Transaction Fee Market

This repository contains the code that was used to generate the bifurcation diagrams in the papers  

- Dynamical Analysis of the EIP-1559 Ethereum Fee Market: https://dl.acm.org/doi/10.1145/3479722.3480993 and https://arxiv.org/abs/2102.10567 and
- Optimality Despite Chaos in Fee Markets: https://fc23.ifca.ai/preproceedings/127.pdf and https://arxiv.org/abs/2212.07175.

There are two notebooks

- **main_EIP1559_quotient.ipynb**: this notebook has the learning rate, d, (currently set by default at d = 0.125), also called step-size or adjustment quotient, as its main bifurcation parameter. One may vary the distribution of valuations between normal, gamma and uniform and the update rule between the standard EIP-1559 (called linear in the notebook), the exponential version and the AMM mechanism.
- **main_EIP1559_valuations.ipynb**: this notebook has the range of valuations, w, as its bifurcation parameter. 

The notebooks show both individual trajectories (upper panels) and averages (bottom panels).
