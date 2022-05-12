"""
    Matrix Operations
    Steal and modify some lines from the StableDQMC package
    for CE and more general calculation purposes
    See (https://github.com/carstenbauer/StableDQMC.jl)
"""
# QRCP decomposion
struct UDT{T<:FloatType} <: Factorization{T}
    Q::Matrix{T}
    D::Vector{T}
    T::Matrix{T}
end

# iteration for destructuring into components
Base.iterate(S::UDT) = (S.Q, Val(:D))
Base.iterate(S::UDT, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::UDT, ::Val{:T}) = (S.T, Val(:done))
Base.iterate(S::UDT, ::Val{:done}) = nothing

Base.similar(S::UDT) = UDT(similar(S.U), similar(S.D), similar(S.T))

LinearAlgebra.det(S::UDT) = prod(S.D) * det(S.Q) * det(S.T)
LinearAlgebra.eigvals(S::UDT) = eigvals(Diagonal(S.D) * S.T * S.Q)

function UDT(A::Matrix{T}) where {T<:FloatType}
    F = qr!(A, Val(true))
    n = size(A)
    D = Vector{T}(undef, n[1])
    R = F.R
    @views F.p[F.p] = 1 : n[2]

    @inbounds for i in 1 : n[1]
        D[i] = R[i, i]
    end
    lmul!(Diagonal(1 ./ D), R)
    UDT(Matrix(F.Q), D, R[:, F.p])
end

function UDT(
    D::Vector{T1}, P::Matrix{T2}, invP::Matrix{T2}
) where {T1<:FloatType, T2<:FloatType}
    """
    Transform P * D * P^-1 into QDT form
    """
    F = UDT(P * Diagonal(D))
    UDT(F.Q, F.D, F.T * invP)
end

function Base.inv(F::UDT)
    inv!(similar(F.Q), F)
end

function inv!(res::Matrix{T}, F::UDT{T}) where {T<:FloatType}
    tmp = similar(F.Q)
    ldiv!(tmp, lu(F.T), Diagonal(1 ./ F.D))
    mul!(res, tmp, F.Q')
    return res
end

function QRCP_update(
    Q::AbstractMatrix, D::AbstractMatrix, T::AbstractMatrix,
    B::AbstractMatrix, direction::Char
)
    """
    Calculate Q', D', T' with Q'D'T' = QDT * B or B * QDT

    # Arguments
    direction => which side of QDT would B be multiplied to
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr!(BQD, Val(true))
        D = Diagonal(QRCP_BQD.R)
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R
        # D^-1 * R * P^T
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T, smart permutation
        T = temp * T

        return QRCP_BQD.Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr!(DTB, Val(true))
        Q = Q * QRCP_DTB.Q
        D = Diagonal(QRCP_DTB.R)
        # D^-1 * R
        temp = inv(D) * QRCP_DTB.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_DTB.p] = temp[:, :]
        T = temp

        return Q, D, T

    else
        @error "direction can only be 'L' or 'R'"
    end
end

##### Low-rank Update #####
function QRCP_update_lowrank(
    Q::AbstractMatrix, D::AbstractMatrix, T::AbstractMatrix,
    B::AbstractMatrix, direction::Char,
    n::Int64, ξ::Float64
)
    """ 
    Calculate Q', D', T' with Q'D'T' = QDT * B or B * QDT

    In-place operations on Q, D and T

    # Arguments
    direction -> which side of QDT would B be multiplied to
    n -> filling (starting point for truncation from above)
    ξ -> truncation threshold
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr!(BQD, Val(true))
        d = diag(QRCP_BQD.R)
        nL = sum(abs.(d[n + 1 : end] / d[n]) .> ξ)
        Q = QRCP_BQD.Q[:, 1 : n + nL]
        D = Diagonal(d[1 : n + nL])
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R[1 : n + nL, :]
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T
        T = temp * T

        return Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr!(DTB, Val(true))
        d = diag(QRCP_DTB.R)
        nR = sum(abs.(d[n + 1 : end] / d[n]) .> ξ)
        Q = Q * QRCP_DTB.Q
        D .= Diagonal(QRCP_DTB.R)
        # D^-1 * R
        temp = inv(D) * QRCP_DTB.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_DTB.p] = temp[:, :]
        T .= temp

        return Q, D, T

    else
        @error "direction can only be 'L' or 'R'"
    end
end

function QRCP_merge(A::UDT, B::UDT)
    """
    Transform QaDaTa * QbDbTb into a single QDT form
    """
    mat = A.T * B.Q
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDT(mat)
    UDT(A.Q * F.Q, F.D, F.T * B.T)
end

function QRCP_merge!(A::UDT, B::UDT)
    """
    In-place transform of QaDaTa * QbDbTb, A would be rewritten
    """
    mat = A.T * B.Q
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDT(mat)
    mul!()
    UDT(A.Q * F.Q, F.D, F.T * B.T)
end

function QRCP_sum(A::UDT, B::UDT)
    """
    Transform QaDaTa + QbDbTb into a single QDT form
    """
    Qa, Da, Ta = A
    Qb, Db, Tb = B

    mat1 = Ta / Tb
    lmul!(Diagonal(Da), mat1)
    mat2 = Qa' * Qb
    rmul!(mat2, Diagonal(Db))
    Q, D, T = UDT(mat1 + mat2)

    mul!(mat1, Qa, Q)
    mul!(mat2, T, Tb)
    UDT(mat1, D, mat2)
end

function QRCP_inv_one_minus()
end