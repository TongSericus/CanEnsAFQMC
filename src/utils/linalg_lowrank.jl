### QRCP decomposion with Truncation ###
struct UDTlr{T<:FloatType} <: Factorization{T}
    U::Matrix{T}
    D::Vector{T}
    T::Matrix{T}
    t::UnitRange{Int64}
end
Base.Matrix(F::UDTlr) = @views (F.U[:, F.t] * Diagonal(F.D[F.t])) * F.T[F.t, :]

# iteration for destructuring into components
Base.iterate(S::UDTlr) = (S.U, Val(:D))
Base.iterate(S::UDTlr, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::UDTlr, ::Val{:T}) = (S.T, Val(:done))
Base.iterate(S::UDTlr, ::Val{:done}) = nothing

Base.similar(S::UDTlr) = UDTlr(similar(S.U), similar(S.D), similar(S.T), S.t)

# stable linear algebra operations
LinearAlgebra.det(S::UDTlr) = @views prod(S.D[S.t]) * det(S.T[S.t, :] * S.U[:, S.t])
LinearAlgebra.eigvals(S::UDTlr) = @views eigvals(S.T[S.t, :] * S.U[:, S.t] * Diagonal(S.D[S.t]), sortby = abs)

UDTlr(n::Int64) = UDTlr(Matrix(1.0I, n, n), ones(Float64, n), Matrix(1.0I, n, n), 1:n)

function lowrank_truncation(D::Vector{T}, N::Int64, ϵ::Float64) where {T<:FloatType}
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
    A::Matrix{T}, N::Int64, ϵ::Float64;
    isTruc::Bool = false
) where {T<:FloatType}
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

    isTruc ?
        UDTlr(Matrix(F.Q)[:, t], D[t], (R[:, F.p])[t, :], t) :
        UDTlr(Matrix(F.Q), D, R[:, F.p], t)
end

# convert regular decomposions to truncated ones
function UDTlr(F::UDT{T}, N::Int64, ϵ::Float64; isTruc::Bool = false) where {T<:FloatType}
    t = lowrank_truncation(F.D, N, ϵ)
    isTruc ?
        UDTlr(F.U[:, t], F.D[t], F.T[t, :], t) :
        UDTlr(F.U, F.D, F.T, t)
end

function QR_lmul(B::AbstractMatrix, A::UDTlr, N::Int64, ϵ::Float64)
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDTlr(mat, N, ϵ)
    T = F.T * A.T
    UDTlr(F.U, F.D, T, F.t)
end

function QR_rmul(A::UDTlr, B::AbstractMatrix, N::Int64, ϵ::Float64)
    mat = A.T * B
    lmul!(Diagonal(A.D), mat)
    F = UDTlr(mat, N, ϵ)
    U = A.U * F.U

    UDTlr(U, F.D, F.T, F.t)
end

function QR_merge(A::UDTlr, B::UDTlr, N::Int64, ϵ::Float64)
    mat = @views A.T[A.t, :] * B.U[:, B.t]
    lmul!(Diagonal(@view A.D[A.t]), mat)
    rmul!(mat, Diagonal(@view B.D[B.t]))
    F = UDTlr(mat, N, ϵ, isTruc=false)
    U = (@view A.U[:, A.t]) * F.U
    T = F.T * @view B.T[B.t, :]

    UDTlr(U[:, F.t], F.D[F.t], T[F.t, :], F.t)
end
