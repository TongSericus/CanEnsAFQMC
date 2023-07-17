"""
    Low-rank Operations with StableLinearAlgebra Package
"""

import StableLinearAlgebra as Slinalg
import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!

function lowrank_truncation(d::AbstractVector{T}, N::Int, ϵ::Float64) where T
    Ns = length(d)
    # truncation from below
    Nϵ = N
    while Nϵ < Ns && d[Nϵ+1] / d[Nϵ] > ϵ
        Nϵ += 1
    end
    t = 1 : Nϵ
end

struct LDRLowRank{T<:Number, E<:AbstractFloat} <: Factorization{T}
    F::LDR{T,E}
    N::Int
    ϵ::Float64
    t::Base.RefValue{UnitRange{Int}}
end

function LDRLowRank(F::LDR{T,E}, N::Int, ϵ::Float64) where {T,E}
    t = lowrank_truncation(F.d, N, ϵ)
    LDRLowRank(F, N, ϵ, Ref(t))
end

function LDRLowRank(
    F::Vector{LDR{T,E}}, N::Int, ϵ::Float64; 
    F_lowrank::Cluster{LDRLowRank{T, E}} = Cluster(B = LDRLowRank{T, E}[])
) where {T,E}
    for f in F
        push!(F_lowrank.B, LDRLowRank(f, N, ϵ))
    end
    
    return F_lowrank
end

Base.copyto!(U::LDRLowRank, V::LDRLowRank, ignore...) = let 
    copyto!(U.F, V.F)
    U.t[] = V.t[]

    return nothing
end

@inline reset!(S::LDRLowRank) = let
    ldr!(S.F, I)
    S.t[] = 1:size(S.F.L)[1]
end

@inline LinearAlgebra.det(S::LDRLowRank) = let
    F = S.F
    @views prod(F.d[S.t[]]) * det(F.R[S.t[], :] * F.L[:, S.t[]])
end
@inline LinearAlgebra.eigvals(S::LDRLowRank) = let
    F = S.F
    @views eigvals(Diagonal(F.d[S.t[]]) * F.R[S.t[], :] * F.L[:, S.t[]], sortby = abs) 
end
LinearAlgebra.eigen(S::LDRLowRank) = let
    F = S.F
    λ, P = eigen(Diagonal(F.d[S.t[]]) * F.R[S.t[], :] * F.L[:, S.t[]], sortby = abs)
    Pₒ = F.L[:, S.t[]] * P
    X = (F.R * F.L)[S.t[], S.t[]] * P
    Pₒ⁻¹ = inv(X) * F.L[S.t[], :]
    λ, Pₒ, Pₒ⁻¹
end

function lmul!(U::AbstractMatrix{T}, V::LDRLowRank{T,E}, ws::LDRWorkspace{T,E}) where {T, E}
    F = V.F
    lmul!(U, F, ws)
    V.t[] = lowrank_truncation(F.d, V.N, V.ϵ)

    return nothing
end

function rmul!(U::LDRLowRank{T,E}, V::AbstractMatrix{T}, ws::LDRWorkspace{T,E}) where {T, E}
    F = U.F
    rmul!(F, V, ws)
    U.t[] = lowrank_truncation(F.d, U.N, U.ϵ)

    return nothing
end

function mul!(
    C::LDRLowRank{T,E}, 
    A::LDRLowRank{T,E}, B::LDRLowRank{T,E}, 
    ws::LDRWorkspace{T,E}
) where {T, E}
    M = C.F
    L = A.F
    R = B.F
    mul!(M, L, R, ws)
    C.t[] = lowrank_truncation(M.d, C.N, C.ϵ)

    return nothing
end

function LinearAlgebra.rmul!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    mul!(C, A, B)
    copyto!(A, C)

    return A
end

"""
    rmul_inv!(A::AbstractMatrix, B::AbstractMatrix, ws::LDRWorkspace)

    Compute A = A * B⁻¹ while overwriting A
"""
function rmul_inv!(A::AbstractMatrix, B::AbstractMatrix, ws::LDRWorkspace)
    B⁻¹ = ws.M′
    copyto!(B⁻¹, B)
    Slinalg.inv_lu!(B⁻¹, ws.lu_ws)

    AB⁻¹ = ws.M
    mul!(AB⁻¹, A, B⁻¹)
    copyto!(A, AB⁻¹)

    return A
end

function LinearAlgebra.lmul!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    mul!(C, A, B)
    copyto!(B, C)

    return B
end

"""
    lmul_inv!(A::AbstractMatrix, B::AbstractMatrix, ws::LDRWorkspace)

    Compute B = A⁻¹ * B while overwriting B
"""
function lmul_inv!(A::AbstractMatrix, B::AbstractMatrix, ws::LDRWorkspace)
    A⁻¹ = ws.M′
    copyto!(A⁻¹, A)
    Slinalg.inv_lu!(A⁻¹, ws.lu_ws)

    A⁻¹B = ws.M
    mul!(A⁻¹B, A⁻¹, B)
    copyto!(B, A⁻¹B)

    return B
end
