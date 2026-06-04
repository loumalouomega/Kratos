# CLAUDE.md — ConstitutiveLawsApplication

> Application-specific guidance. Inherits everything in the root `/CLAUDE.md`; only the
> constitutive-law specifics are documented here. Read the root file first for global conventions.
> The user (Vicente Mataix Ferrándiz) is one of the authors of this application.

## Purpose

The **ConstitutiveLawsApplication** is the central library of **material models** for Kratos
solid mechanics. It hosts hyperelastic laws, small- and finite-strain plasticity and damage,
thermal and composite laws, and the generic templated machinery (yield surfaces × plastic
potentials, integrators) that combines them. It is the natural companion of the
**StructuralMechanicsApplication**, which provides the elements that call these laws.

The laws are heavily **template-based**: e.g. small-strain isotropic plasticity is
`SmallStrainIsotropicPlasticity3D<YieldSurface, PlasticPotential>`, instantiated for all
combinations of VonMises / ModifiedMohrCoulomb / Tresca / DruckerPrager (and more). The
`constitutive_laws_application.cpp` registers ~hundreds of concrete instantiations
(≈480 `KRATOS_REGISTER_*` calls).

## Dependencies

- **KratosCore** (always).
- **StructuralMechanicsApplication** — provides the constitutive-law base infrastructure
  (`ConstitutiveLaw`, strain/stress utilities) and the elements that drive these laws; it is
  effectively a co-dependency for real simulations.

## Directory layout (application-specific)

```
custom_constitutive/
  small_strains/      # small-strain elastic, plasticity, damage, viscous laws
  finite_strains/     # hyperelastic (Neo-Hookean, Kirchhoff) + finite-strain plasticity
  composites/         # rule-of-mixtures, serial-parallel, orthotropic composite laws
  thermal/            # thermo-mechanical laws
  structural_elements_constitutive_laws/  # truss/beam/cable/membrane 1D & plane laws
  auxiliary_files/    # yield surfaces, plastic potentials, integrators, table_keys, helpers
custom_processes/     # material-state / damage / fatigue support processes
custom_utilities/     # tangent-operator, advanced constitutive-law utilities
custom_python/        # bindings: add_custom_constitutive_laws/processes/utilities_to_python
python_scripts/       # high_cycle_fatigue_analysis.py, pre-stress/damage helper processes,
  symbolic_generation/   #   sympy generators for symbolic laws
tests/
  cpp_tests/          # GTest + constitutive_laws_fast_suite.{h,cpp}
  test_ConstitutiveLawsApplication.py
```

## Build

- CMake libs: `KratosConstitutiveLawsCore` (SHARED) and `KratosConstitutiveLawsApplication`
  (pybind11). Compile definition: `CONSTITUTIVE_LAWS_APPLICATION=EXPORT,API`.
- Sources auto-globbed from `custom_*` — new `.cpp` files are picked up automatically.
- This is a **heavy compilation unit** (many template instantiations); unity builds help.

## Authoring a constitutive law

A `ConstitutiveLaw` subclass typically overrides:

```cpp
void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters& rValues) override;
void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;
void InitializeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;
void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters& rValues) override;
SizeType GetStrainSize() const override;
int Check(const Properties&, const GeometryType&, const ProcessInfo&) const override;
```

- Stress/strain measures (PK2, Cauchy, Kirchhoff) and the `Flags` on
  `ConstitutiveLaw::Parameters` (`COMPUTE_STRESS`, `COMPUTE_CONSTITUTIVE_TENSOR`) decide what
  must be filled. Respect the requested flags.
- Plasticity/damage laws compose a **yield surface** + **plastic potential** + **integrator**
  from `auxiliary_files/` via templates — prefer adding a new yield surface/potential and
  instantiating, over writing a monolithic law.
- Material parameters come from `Properties` (the `StructuralMaterials.json`), often through
  `Table`s for nonlinear/temperature-dependent data.

## Registration (critical)

- Every concrete law must be registered with `KRATOS_REGISTER_CONSTITUTIVE_LAW("Name", instance)`
  in `constitutive_laws_application.cpp`, and the **template instantiation** must exist.
  Forgetting either makes the law invisible to `StructuralMaterials.json`.
- New variables → `constitutive_laws_application_variables.{h,cpp}`.

## Testing

- **C++ fixture:** `KratosConstitutiveLawsFastSuite` (`constitutive_laws_fast_suite.h`);
  generic tests use `KratosCoreFastSuite`.
- **Python:** `test_ConstitutiveLawsApplication.py` is the suite entry point — register new
  `test_*.py` there. Many laws are validated by single-point integration drivers.

## Conventions & gotchas

- Some laws are **symbolically generated** (`python_scripts/symbolic_generation/`) — edit the
  generator and regenerate; don't hand-patch generated tangent/stress code.
- Keep `GetStrainSize()` consistent with the law's dimension (3 → 2D plane, 6 → 3D, etc.).
- The base `ConstitutiveLaw` interface lives in **KratosCore**; the structural-specific helpers
  live in **StructuralMechanicsApplication** — build both when testing real elements.
