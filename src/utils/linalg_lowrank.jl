"""
    Matrix Operations with truncations
"""
struct UDTlr{T<:Number} <: Factorization{T}
    U::AbstractMatrix{T}
    D::AbstractVector{T}
    T::AbstractMatrix{T}
    t::Base.RefValue{UnitRange{Int}}
end
Base.Matrix(F::UDTlr) = @views (F.U[:, F.t[]] * Diagonal(F.D[F.t[]])) * F.T[F.t[], :]

# iteration for destructuring into components
Base.iterate(S::UDTlr) = @views (S.U[:, S.t[]], Val(:D))
Base.iterate(S::UDTlr, ::Val{:D}) = @views (S.D[S.t[]], Val(:T))
Base.iterate(S::UDTlr, ::Val{:T}) = @views (S.T[S.t[], :], Val(:done))
Base.iterate(S::UDTlr, ::Val{:done}) = nothing

Base.similar(S::UDTlr) = let
    F = UDTlr(similar(S.U), similar(S.D), similar(S.T), Ref(S.t[]))
    copyto!(F.U, I)
    fill!(F.D, 1.0)
    copyto!(F.T, I)
    F
end

collectU(S::Vector{UDTlr{Ts}}) where Ts = [S[i].U for i in 1 : length(S)]
collectD(S::Vector{UDTlr{Ts}}) where Ts = [S[i].D for i in 1 : length(S)]
collectT(S::Vector{UDTlr{Ts}}) where Ts = [S[i].T for i in 1 : length(S)]

Base.copyto!(F::UDTlr{Tf}, S::UDTlr{Tf}, ignore...) where Tf = let
    copyto!(F.U, S.U)
    copyto!(F.D, S.D)
    copyto!(F.T, S.T)
    F.t[] = S.t[]

    nothing
end

# stable linear algebra operations
LinearAlgebra.det(S::UDTlr) = @views prod(S.D[S.t[]]) * det(S.T[S.t[], :] * S.U[:, S.t[]])
LinearAlgebra.eigvals(S::UDTlr) = @views eigvals(Diagonal(S.D[S.t[]]) * S.T[S.t[], :] * S.U[:, S.t[]], sortby = abs)
LinearAlgebra.eigen(S::UDTlr) = let 
    λ, P = eigen(Diagonal(S.D[S.t[]]) * S.T[S.t[], :] * S.U[:, S.t[]], sortby = abs)
    Pocc = S.U[:, S.t[]] * P
    X = (S.T * S.U)[S.t[], S.t[]] * P
    invPocc = inv(X) * S.T[S.t[], :]
    λ, Pocc, invPocc
end

UDTlr(n::Int) = UDTlr(Matrix(1.0I, n, n), ones(Float64, n), Matrix(1.0I, n, n), Ref(1:n))
reset!(S::UDTlr{Ts}) where Ts = copyto!(S, UDTlr(size(S.U)[1]))

function lowrank_truncation(D::Vector{T}, N::Int, ϵ::Float64) where {T<:Number}
    Ns = length(D)
    # truncation from below
    Dϵ = D[N] * ϵ
    Nϵ = N
    while Nϵ < Ns + 1 && D[Nϵ] > Dϵ
        Nϵ += 1
    end
    t = 1 : Nϵ - 1
end

function UDTlr(
    A::AbstractMatrix{Tf}, N::Int, ϵ::Float64
) where {Tf<:Number}
    F = qr!(A, Val(true))
    n = size(F.R)
    D = Vector{Tf}(undef, n[1])
    R = F.R
    @views F.p[F.p] = 1 : n[2]

    @inbounds for i in 1 : n[1]
        D[i] = abs(R[i, i])
    end
    lmul!(Diagonal(1 ./ D), R)
    t = lowrank_truncation(D, N, ϵ)

    UDTlr(Matrix(F.Q), D, R[:, F.p], Ref(t))
end

# convert regular decomposions to truncated ones
function UDTlr(F::UDT{Tf}, N::Int, ϵ::Float64) where Tf
    t = lowrank_truncation(F.D, N, ϵ)
    UDTlr(F.U, F.D, F.T, Ref(t))
end

function QR_lmul(B::AbstractMatrix{Tb}, A::UDTlr{Ta}, N::Int, ϵ::Float64) where {Ta, Tb}
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDTlr(mat, N, ϵ)

    UDTlr(F.U, F.D, F.T * A.T, F.t)
end

function QR_rmul(A::UDTlr{Ta}, B::AbstractMatrix{Tb}, N::Int, ϵ::Float64) where {Ta, Tb}
    mat = A.T * B
    lmul!(Diagonal(A.D), mat)
    F = UDTlr(mat, N, ϵ)

    UDTlr(A.U * F.U, F.D, F.T, F.t)
end

function QR_merge(A::UDTlr{Ta}, B::UDTlr{Tb}, N::Int, ϵ::Float64) where {Ta, Tb}
    """
    Compute the factorization of A * B
    """
    mat = @views A.T[A.t[], :] * B.U[:, B.t[]]
    lmul!(Diagonal(@view A.D[A.t[]]), mat)
    rmul!(mat, Diagonal(@view B.D[B.t[]]))
    F = UDTlr(mat, N, ϵ)
    U = (@view A.U[:, A.t[]]) * F.U
    T = F.T * @view B.T[B.t[], :]

    UDTlr(U[:, F.t[]], F.D[F.t[]], T[F.t[], :], F.t)
end

function QR_merge!(
    C::UDTlr{Tc}, A::UDTlr{Ta}, B::UDTlr{Tb}, N::Int, ϵ::Float64
) where {Ta, Tb, Tc}
    """
    Compute the factorization of A * B, writing the results into preallocated C
    """
    mat = @views A.T[A.t[], :] * B.U[:, B.t[]]
    lmul!(Diagonal(@view A.D[A.t[]]), mat)
    rmul!(mat, Diagonal(@view B.D[B.t[]]))
    l = minimum(size(mat))
    F = UDTlr(mat, N, ϵ)

    @views mul!(C.U[:, 1:l], A.U[:, A.t[]], F.U)
    @views mul!(C.T[1:l, :], F.T, B.T[B.t[], :])
    @inbounds for i in 1 : length(F.D)
        C.D[i] = F.D[i]
    end
    C.t[] = F.t[]

    return C
end

### Additional Functions Added for StableLinearAlgebra Package ###
import LinearAlgebra: lmul!, rmul!, mul!

function UDTlr(F::LDR{T,E}, N::Int, ϵ::Float64) where {T,E}
    t = lowrank_truncation(F.d, N, ϵ)
    UDTlr(F.L, F.d, F.R, Ref(t))
end

function lmul!(A::AbstractMatrix{T}, F::UDTlr{T}, N::Int, ϵ::Float64, ws::LDRWorkspace{T,E}) where {T, E}
    M = LDR(F.U, F.D, F.T)
    lmul!(A, M, ws)
    F.t[] = lowrank_truncation(F.D, N, ϵ)

    return nothing
end

function rmul!(F::UDTlr{T}, A::AbstractMatrix{T}, N::Int, ϵ::Float64, ws::LDRWorkspace{T,E}) where {T, E}
    M = LDR(F.U, F.D, F.T)
    rmul!(M, A, ws)
    F.t[] = lowrank_truncation(F.D, N, ϵ)

    return nothing
end

function mul!(C::UDTlr{T}, A::UDTlr{T}, B::UDTlr{T}, N::Int, ϵ::Float64, ws::LDRWorkspace{T,E}) where {T, E}
    """
    Compute the factorization of A * B, writing the results into preallocated C
    """
    M = LDR(C.U, C.D, C.T)
    L = LDR(A.U, A.D, A.T)
    R = LDR(B.U, B.D, B.T)
    mul!(M, L, R, ws)
    C.t[] = lowrank_truncation(C.D, N, ϵ)

    return nothing
end
