---
title: Algorithm
keywords: [LEoPart PDE-constrained projection particle level set static condensation]
tags: [Algorithm.md]
sidebar: fluid_dynamics_application
summary: Formulation of the particle transport, the PDE-constrained projection, the distance reconstruction and the volume correction.
---

## Dual representation

Each Lagrangian particle $$p$$ carries two values:

* a binary **phase indicator** $$c_p \in \{0, 1\}$$, which is 1 in the fluid with negative `DISTANCE` (the water in the hydraulic cases);
* a **signed distance** $$\phi_p$$.

On the mesh, two element-wise linear (discontinuous P1) fields are rebuilt from them every step:

* $$c_h$$, the projected indicator, obtained with the **PDE-constrained projection**. It is conservative, and its integral defines the volume of the fluid.
* $$\phi_h$$, the projected distance, obtained with an element-wise least-squares ($$\ell^2$$) projection. It gives the interface position and the nodal `DISTANCE` used by the two-fluid elements.

The two are linked by a volume correction: `DISTANCE` is shifted so that the volume of its negative side equals $$\int_\Omega c_h$$.

The indicator is never used to rebuild the interface geometry. A projected discontinuous indicator has over/undershoots and is not a good source of normals or curvature, while a signed distance would not be meaningful to conserve. Splitting the roles keeps each field doing what it does well.

## One time step

`ParticleMeshPdeProjectionProcess::Execute` replaces the Eulerian level set convection in `InitializeSolutionStep` of the solvers. One call performs:

1. **Face classification.** Each boundary face is classified as wall, outflow or inflow (see [boundaries](#boundary-faces)).
2. **Particle advection** with an explicit Runge-Kutta scheme (orders 1 to 3), through the nodal `VELOCITY` interpolated linearly in time between the previous (buffer 1) and the current (buffer 0) step.
3. **Population control.** The number of particles of every element is kept between the minimum and the maximum.
4. **PDE-constrained projection** of $$c_p$$ onto $$c_h$$.
5. **$$\ell^2$$ projection** of $$\phi_p$$ onto $$\phi_h$$, followed by a volume-weighted average to the nodal `DISTANCE`.
6. **Volume correction.** Find the shift $$s$$ such that the negative volume of `DISTANCE` + $$s$$ equals $$\int c_h$$, and apply it to the nodes, the elemental field and the particles.
7. **Output.** Write $$c_h$$ to the elemental (mean value) and nodal (volume-weighted average) `PHASE_FRACTION`.

After the Navier-Stokes solution the solvers redistance `DISTANCE` as usual. `ExecuteFinalizeSolutionStep` then adds the change of `DISTANCE` made since step 6 to the particle distances, a FLIP-like update: $$\phi_p \leftarrow \phi_p + \sum_a N_a(x_p)\,\Delta d_a$$. The particle indicators are never modified, so pure transport keeps them exact.

The order inside the two-fluid solvers is:

| Solver stage | Particle transport |
|---|---|
| `InitializeSolutionStep` | `Execute` (steps 1-7), then distance modification and nodal properties as before |
| `SolveSolutionStep` | unchanged |
| `FinalizeSolutionStep` | redistancing, distance modification finalize, then `ExecuteFinalizeSolutionStep` |

On the first call the process initializes itself from the current `DISTANCE` (see [initialization](#initialization)). DISTANCE may be set by processes or by `ModifyInitialGeometry`, which run after the solver `Initialize`, so the initialization is delayed until then.

## PDE-constrained projection

### Unknowns

On a simplex mesh, with $$d$$ the dimension and face $$i$$ of an element opposite to its node $$i$$:

* $$c_h|_K \in P_1(K)$$, the projected indicator, with $$d+1$$ nodal values per element;
* $$\lambda_h|_K \in P_0(K)$$, a Lagrange multiplier, one value per element. LEoPart recommends $$l = 0$$: the scheme is then independent of the time integration parameter θ.
* $$\bar c_h|_F \in P_1(F)$$, a single-valued facet field on every face shared by two elements, with $$d$$ values per facet (one per facet vertex). Boundary faces have no facet unknown.

### Optimization problem

The projection minimizes

$$J = \frac12 \sum_p \big(c_h(x_p) - c_p\big)^2 + \frac12 \sum_K \sum_{F \subset \partial K,\ F\ \text{shared}} \beta_K \int_F (\bar c_h - c_h)^2 + \frac12 \sum_K \zeta_K \int_K |\nabla c_h|^2$$

subject to one conservation statement per element (the test function of the constraint is the P0 multiplier):

$$\int_K (c_h - c^*) \,\mathrm{d}\Omega + \Delta t \sum_{F\ \text{shared}} \int_F (u\cdot n)\, \bar c_h \,\mathrm{d}\Gamma + \Delta t \sum_{F\ \text{open boundary}} \int_F (u\cdot n)\, \tilde c \,\mathrm{d}\Gamma = 0.$$

The terms are:

* $$c^*$$, the projected indicator of the previous step (pure transport, no constitutive update);
* $$u = \tfrac12 (u^n + u^{n+1})$$, the nodal velocity used in the fluxes;
* $$\tilde c$$, the boundary value. By default it is the level set indicator $$g = H(-d)$$ evaluated at each quadrature point of every open boundary face. With `outflow_flux_from_level_set: false` the outflowing part $$(u\cdot n)^+$$ uses the interior trace $$c_h$$ instead, as in LEoPart. Wall faces carry no flux.
* $$\beta_K$$ and $$\zeta_K$$, the facet and gradient penalties, scaled with the element size $$h_K$$ (the minimum height) so that they do not depend on the mesh units:
$$\beta_K = \beta\, h_K^{1-d}, \qquad \zeta_K = \zeta\, h_K^{2-d}.$$

### Why it is conservative

The constraint states that the content of every element changes only through its face fluxes. The flux through a shared facet uses the single-valued $$\bar c_h$$ and a single-valued normal velocity. The nodal velocity is continuous, and both elements use the same area normal with opposite signs. The contributions of the two elements therefore cancel exactly, and the global integral changes only by the open boundary fluxes.

No H(div) velocity reconstruction is needed. The velocity is only weakly divergence-free, and the small $$c\,\nabla\cdot u$$ mismatch between the particle (material) and the constraint (conservative) form appears as a slight over/undershoot, never as a loss of conservation.

### Local system and static condensation

The optimality conditions of each element form a symmetric saddle-point system:

$$
\begin{bmatrix} A & g & L \\ g^T & 0 & h^T \\ L^T & h & B \end{bmatrix}
\begin{bmatrix} c \\ \lambda \\ \bar c \end{bmatrix}
=
\begin{bmatrix} \sum_p N(x_p)\, c_p \\ \int_K c^* - \Delta t\, q_g \\ 0 \end{bmatrix}
$$

with:

* $$A = \sum_p N(x_p) N(x_p)^T + \beta_K \sum_F M_F + \zeta_K \int_K \nabla N \cdot \nabla N^T$$, where $$M_F$$ is the face mass matrix $$\vert F\vert(1+\delta_{ab})/(d(d+1))$$;
* $$g = \int_K N$$, plus the outflow trace coefficients $$\Delta t \int_{\partial K} (u\cdot n)^+ N$$ when the trace option is used;
* $$L = -\beta_K M_F$$ and $$B = \beta_K M_F$$ for each shared face;
* $$h = \Delta t \int_F (u\cdot n)\, \bar N$$ for each shared face, computed exactly from the nodal normal velocities and $$M_F$$;
* $$q_g$$, the prescribed boundary flux $$\int (u\cdot n)\, g$$. It is integrated with a 3-point (2D) or 6-point (3D) Gauss rule, because the integrand is not polynomial.

The element unknowns $$(c, \lambda)$$ are eliminated locally. The $$(d+2)\times(d+2)$$ block $$K = \begin{bmatrix} A & g \\ g^T & 0\end{bmatrix}$$ is solved with an LU factorization with partial pivoting, which gives the Schur complement and its right-hand side:

$$S_K = B - C^T K^{-1} C, \qquad r_K = -C^T K^{-1} R, \qquad C = \begin{bmatrix} L \\ h \end{bmatrix}.$$

The $$S_K$$ are assembled into a global system on the shared facet unknowns only. That system is symmetric positive definite: it is the Hessian of the strictly convex reduced least-squares problem. After it is solved, every element recovers $$[c; \lambda] = K^{-1}(R - C\bar c)$$.

The recovery solves the element constraint exactly for **any** $$\bar c$$. The local and global conservation therefore does not depend on the accuracy of the global solve, which only affects the optimality of the fit. The process reports the cell balance residual $$\vert\int_K (c_h - c^*) + \text{fluxes}\vert / \vert K\vert$$ and the relative residual of the facet solve.

## Particle advection

* **Seeding.** Particles are placed at uniformly distributed random barycentric coordinates (normalized exponential samples) inside each element. The random stream depends on the element Id, the random seed and a counter. Initial values: $$\phi_p$$ is `DISTANCE` interpolated at the particle, and $$c_p = 1$$ if $$\phi_p < 0$$.
* **Integration.** Each particle uses $$n = \lceil \vert u\vert\,\Delta t / (\text{CFL}_{max}\, h_K) \rceil$$ substeps (at most 1000) of an explicit RK1, RK2 (Heun) or RK3 scheme. The velocity at a stage point is the nodal velocity interpolated in space and linearly in time between the two steps.
* **Location.** The host element is found by walking through the face neighbours towards the most negative barycentric coordinate. When the walk hits the boundary, a bins search (`BinBasedFastPointLocator`) confirms whether the point really left the domain, which matters for non-convex domains.
* **Boundaries.** A particle leaving through an outflow face is deleted. A particle leaving through any other face is put back slightly inside along the face normal, or kept where it was if that fails.

### Boundary faces

A boundary face is a **wall** if all its nodes are impermeable: nodes flagged `SLIP`, or nodes with all the velocity components fixed to zero. Walls carry no indicator flux and push particles back.

The other boundary faces are **outflow** or **inflow** according to the sign of the face-averaged normal velocity. Walls have to be detected explicitly: the nodal velocity of slip walls has a small normal component with respect to the face normals, especially at edges and corners, and using it would create or destroy volume (see [results](Design_Decisions_and_Results.html#boundary-fluxes)).

### Population control

After the advection the number of particles of every element is brought into [`minimum`, `maximum`]:

* **Adding particles.** Each new particle is placed at a random position. Its values depend on the element:
  * if the element has an inflow face, both values come from the nodal `DISTANCE` (imposed by the inlet);
  * otherwise, if the element has particles, the indicator is copied from the closest one. The distance is extrapolated from it with the gradient of the elemental $$\phi_h$$.
  * otherwise (empty element), the values come from the elemental $$c_h$$ (snapped to 0 or 1) and $$\phi_h$$.
* **Removing particles.** The particles with the smallest nearest-neighbour distance are removed until the maximum is reached.

## Distance reconstruction

The element-wise $$\ell^2$$ projection minimizes $$\sum_p (\phi_h(x_p) - \phi_p)^2$$ in every element. A small Tikhonov term towards the particle mean ($$\varepsilon = 10^{-8}\,\mathrm{tr}(M_p)/(d+1)$$) keeps badly distributed particles well posed. Elements with fewer particles than nodes take the particle mean, and empty elements keep their previous values.

The nodal `DISTANCE` is the volume-weighted average of the element values at each node.

## Volume correction

The shift $$s$$ is the root of

$$f(s) = V^-\big(d + s\big) - \int_\Omega c_h, $$

where $$V^-$$ is the exact volume of the negative side of the linear distance field. It is computed with the standard modified shape functions, using the same split criterion as the geometry division utilities. $$f$$ does not increase with $$s$$.

The root is bracketed starting from a step of $$0.1\,h_{min}$$ that doubles at each try, and then refined with the Illinois variant of regula falsi until $$\vert f\vert$$ is below the relative tolerance times the domain volume. The correction is skipped when the target volume is empty or fills the domain.

This correction is a single global shift. The cell-wise conservation lives in $$c_h$$; `DISTANCE` only matches its total volume.

## Initialization

On the first step:

* the particles are seeded from `DISTANCE`;
* $$\phi_h$$ is the element-wise interpolation of `DISTANCE`;
* $$c_h$$ is the element-wise $$L^2$$ projection of $$H(-d)$$. Cut elements are integrated exactly on their negative side with the modified shape functions, so $$\int c_h^0$$ equals the initial negative volume of `DISTANCE`. The P1 mass matrix is inverted in closed form, $$M^{-1} = \tfrac{(d+1)(d+2)}{\vert K\vert}\big(I - \tfrac{1}{d+2}\mathbf{1}\mathbf{1}^T\big)$$.
