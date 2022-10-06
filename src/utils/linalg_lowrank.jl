"""
    Matrix Operations with truncations
"""
struct UDTlr{T} <: Factorization{T}
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

Base.similar(S::UDTlr) = UDTlr(similar(S.U), similar(S.D), similar(S.T), Ref(S.t[]))

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
    A::AbstractMatrix{T}, N::Int, ϵ::Float64
) where {T<:Number}
    F = qr!(A, Val(true))
    n = size(F.R)
    D = Vector{T}(undef, n[1])
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
function UDTlr(F::UDT, N::Int, ϵ::Float64)
    t = lowrank_truncation(F.D, N, ϵ)
    UDTlr(F.U, F.D, F.T, Ref(t))
end

function QR_lmul(B::AbstractMatrix, A::UDTlr, N::Int, ϵ::Float64)
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDTlr(mat, N, ϵ)

    UDTlr(F.U, F.D, F.T * A.T, F.t)
end

function QR_rmul(A::UDTlr, B::AbstractMatrix, N::Int, ϵ::Float64)
    mat = A.T * B
    lmul!(Diagonal(A.D), mat)
    F = UDTlr(mat, N, ϵ)

    UDTlr(A.U * F.U, F.D, F.T, F.t)
end

function QR_merge(A::UDTlr, B::UDTlr, N::Int, ϵ::Float64)
    mat = @views A.T[A.t[], :] * B.U[:, B.t[]]
    lmul!(Diagonal(@view A.D[A.t[]]), mat)
    rmul!(mat, Diagonal(@view B.D[B.t[]]))
    F = UDTlr(mat, N, ϵ)
    U = (@view A.U[:, A.t[]]) * F.U
    T = F.T * @view B.T[B.t[], :]

    UDTlr(U[:, F.t[]], F.D[F.t[]], T[F.t[], :], F.t)
end

function QR_merge!(
    C::UDTlr, A::UDTlr, B::UDTlr, N::Int, ϵ::Float64
)
    """
        write the results into preallocated C
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
