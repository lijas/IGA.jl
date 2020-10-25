using JuAFEM, SparseArrays, LinearAlgebra, Printf

struct J2Plasticity{T, S <: SymmetricTensor{4, 3, T}}
    G::T  # Shear modulus
    K::T  # Bulk modulus
    σ₀::T # Initial yield limit
    H::T  # Hardening modulus
    Dᵉ::S # Elastic stiffness tensor
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

