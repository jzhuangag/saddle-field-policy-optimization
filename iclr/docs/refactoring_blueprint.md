# ICLR paper architecture

## Central claim

The paper studies joint policy optimization in two-player zero-sum Markov games.
Its central contribution is a geometry-adaptive optimizer that separates the first-order saddle-field direction from the Jacobian-induced curvature direction and selects their two step sizes through a local Lyapunov-drift model.

The Lyapunov model is the online design criterion for the policy update.
It does not replace the Markov-game objective or directly solve the dynamic program.

## Nine-page main-text budget

1. Introduction: 1.4 pages.
2. Problem and optimization challenge: 1.0 page.
3. Geometry-adaptive policy update: 2.2 pages.
4. Main theoretical guarantees: 1.5 pages.
5. Experiments: 2.6 pages.
6. Conclusion and required statements: 0.3 pages.

References and appendices follow the main text and are outside the nine-page initial-submission limit.

## Main-text narrative

1. Define the state, the two policies, the zero-sum return, and the learning objective.
2. Explain why simultaneous policy gradients rotate and why this differs from single-agent policy improvement.
3. Organize existing methods into value-based methods, direct policy optimization, proximal/extragradient corrections, and Jacobian-based game optimization.
4. Derive the shared local displacement of PPM and EGM and identify the field and curvature coordinates.
5. State the geometric criterion for useful curvature.
6. Present the stochastic two-direction policy update before the full technical assumptions.
7. Introduce the composite Lyapunov function and explain why negative conditional drift represents one-step progress in stationarity and policy-space optimality.
8. Present the two-variable QP and a compact algorithm box.
9. State one main finite-time theorem and one policy-performance corollary in the main text.
10. Connect every experiment to one part of this chain: geometry, activation, learning stability, or policy performance.

## Material moved to the appendix

- Full PPM and EGM expansion proof.
- Detailed smoothness and oracle assumptions.
- All Taylor coefficients, perturbation budgets, and auxiliary lemmas not needed to understand the algorithm.
- Closed-form boundary-case enumeration for the box QP.
- Complete finite-time proof and performance-bridge proof.
- Environment definitions, network details, hyperparameter grids, and additional plots.

## Revision rule

The ICLR manuscript is not a format conversion of the IEEE manuscript.
The main text will be rewritten around the learning question, the algorithm will appear before the long analysis, and the experiments will use a dedicated ICLR protocol.
