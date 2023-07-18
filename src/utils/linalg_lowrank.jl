"""
    Low-rank Operations with StableLinearAlgebra Package
"""

import StableLinearAlgebra as Slinalg
import StableLinearAlgebra: mul!, lmul!, rmul!, rdiv!

struct LDRLowRank{T<:Number, E<:AbstractFloat} <: Factorization{T}
    F::LDR{T,E}
    N::Int
    ϵ::Float64
    t::Base.RefValue{UnitRange{Int}}
end

function lowrank_truncation!(
    M::LDRLowRank{T,E}; 
    ws::LDRWorkspace{T,E} = ldr_workspace(M.F)
) where {T,E}
    F = M.F
    N = M.N
    ϵ = M.ϵ

    d = F.d
    R = F.R

    dk = ws.v
    Rk = ws.M
    transpose!(Rk, R)

    Ns = length(dk)
    @inbounds for k in N : Ns
        @views dk[k] = d[k] * sum(abs.(Rk[:, k]) .^ 2) / d[N]
    end

    # truncation from below
    Nϵ = Ns
    s = dk[Nϵ]
    while Nϵ > N+1 && sqrt(s) < ϵ
        Nϵ -= 1
        s += dk[Nϵ]
    end

    M.t[] = 1:Nϵ
    return nothing
end

function LDRLowRank(F::LDR{T,E}, N::Int, ϵ::Float64; doTruncation::Bool = false) where {T,E}
    M = LDRLowRank(F, N, ϵ, Ref(1:length(F.d)))
    doTruncation && lowrank_truncation!(M)
    return M
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
LinearAlgebra.eigvals(S::LDRLowRank, ws::LDRWorkspace) = let
    F = S.F
    Mat = @views ws.M[S.t[], S.t[]]
    @views mul!(Mat, F.R[S.t[], :], F.L[:, S.t[]])
    @views lmul!(Diagonal(F.d[S.t[]]), Mat)
    # diagonalize
    eigvals!(Mat, sortby = abs) 
end

LinearAlgebra.eigen(S::LDRLowRank) = let
    F = S.F
    λ, P = eigen(Diagonal(F.d[S.t[]]) * F.R[S.t[], :] * F.L[:, S.t[]], sortby = abs)
    Pₒ = F.L[:, S.t[]] * P
    X = (F.R * F.L)[S.t[], S.t[]] * P
    Pₒ⁻¹ = inv(X) * F.R[S.t[], :]
    λ, Pₒ, Pₒ⁻¹
end

function LinearAlgebra.eigen(S::LDRLowRank{T,E}, ws::LDRWorkspace{T,E}) where {T,E}
    F = S.F
    lowrank_truncation!(S, ws=ws)

    Mat = @views ws.M[S.t[], S.t[]]
    @views mul!(Mat, F.R[S.t[], :], F.L[:, S.t[]])
    @views lmul!(Diagonal(F.d[S.t[]]), Mat)

    λₒ, P = eigen!(Mat, sortby = abs)

    @views Pₒ = F.L[:, S.t[]] * P

    RL = ws.M
    mul!(RL, F.R, F.L)
    X = @view ws.M″[S.t[], S.t[]]
    @views mul!(X, RL[S.t[], S.t[]], P)
    X⁻¹ = X
    inv_lu!(X⁻¹, ws.lu_ws)
    @views Pₒ⁻¹ = X⁻¹ * F.R[S.t[], :]

    return λₒ, Pₒ, Pₒ⁻¹
end

function lmul!(
    U::AbstractMatrix{T}, V::LDRLowRank{T,E}, ws::LDRWorkspace{T,E}; 
    doTruncation::Bool = false
) where {T, E}
    F = V.F
    lmul!(U, F, ws)
    doTruncation && (V.t[] = lowrank_truncation!(V))

    return nothing
end

function rmul!(
    U::LDRLowRank{T,E}, V::AbstractMatrix{T}, ws::LDRWorkspace{T,E}; 
    doTruncation::Bool = false
) where {T, E}
    F = U.F
    rmul!(F, V, ws)
    doTruncation && (U.t[] = lowrank_truncation!(U))

    return nothing
end

function mul!(
    C::LDRLowRank{T,E}, 
    A::LDRLowRank{T,E}, B::LDRLowRank{T,E}, 
    ws::LDRWorkspace{T,E}; 
    doTruncation::Bool = false
) where {T, E}
    M = C.F
    L = A.F
    R = B.F
    mul!(M, L, R, ws)
    doTruncation && (C.t[] = lowrank_truncation!(C))

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
