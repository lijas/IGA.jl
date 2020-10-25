```@meta
EditURL = "<unknown>/docs/src/literate/example1.jl"
```

### Material parameters and state variables

Start by loading some necessary packages

```@example example1
using JuAFEM, SparseArrays, LinearAlgebra, Printf
```

We define a J₂-plasticity-material, containing material parameters and the elastic
stiffness Dᵉ (since it is constant)

```@example example1
struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}}
    G::T  # Shear modulus
    K::T  # Bulk modulus
    σ₀::T # Initial yield limit
    H::T  # Hardening modulus
    Dᵉ::S # Elastic stiffness tensor
end;
nothing #hide
```

Test

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

