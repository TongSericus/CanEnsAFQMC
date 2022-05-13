"""
    Matrix Operations
    Steal and modify some lines from the StableDQMC package
    for CE and more general calculation purposes
    See (https://github.com/carstenbauer/StableDQMC.jl)
"""
# QRCP decomposion
struct UDT{T<:FloatType} <: Factorization{T}
    U::Matrix{T}
    D::Vector{T}
    T::Matrix{T}
end

# iteration for destructuring into components
Base.iterate(S::UDT) = (S.U, Val(:D))
Base.iterate(S::UDT, ::Val{:D}) = (S.D, Val(:T))
Base.iterate(S::UDT, ::Val{:T}) = (S.T, Val(:done))
Base.iterate(S::UDT, ::Val{:done}) = nothing

Base.similar(S::UDT) = UDT(similar(S.U), similar(S.D), similar(S.T))

# stable linear algebra operations
LinearAlgebra.det(S::UDT) = prod(S.D) * det(S.U) * det(S.T)
LinearAlgebra.eigvals(S::UDT) = eigvals(Diagonal(S.D) * S.T * S.U, sortby = abs)
LinearAlgebra.eigen(S::UDT) = let 
    eigenS = eigen(Diagonal(S.D) * S.T * S.U, sortby = abs)
    LinearAlgebra.Eigen(eigenS.values, S.U * eigenS.vectors)
end
Base.inv(F::UDT) = inv!(similar(F.U), F)
function inv!(res::Matrix{T}, F::UDT{T}) where {T<:FloatType}
    tmp = similar(F.U)
    ldiv!(tmp, lu(F.T), Diagonal(1 ./ F.D))
    mul!(res, tmp, F.U')
    return res
end


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
    Transform P * D * P^-1 into UDT form
    """
    F = UDT(P * Diagonal(D))
    UDT(F.U, F.D, F.T * invP)
end

function QRCP_lmul(B::Matrix{T}, A::UDT) where {T<:FloatType}
    """
    Compute B * UaDaTa
    """
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDT(mat)
    UDT(F.U, F.D, F.T * A.T)
end

function QRCP_rmul(A::UDT, B::Matrix{T}) where {T<:FloatType}
    """
    Compute UaDaTa * B
    """
    mat = A.T * B
    lmul!(Diagonal(A.D), mat)
    F = UDT(mat)
    UDT(A.U * F.U, F.D, F.T)
end

function QRCP_merge(A::UDT, B::UDT)
    """
    Compute UaDaTa * UbDbTb
    """
    mat = A.T * B.U
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDT(mat)
    UDT(A.U * F.U, F.D, F.T * B.T)
end

function QRCP_sum(A::UDT, B::UDT)
    """
    Compute UaDaTa + UbDbTb
    """
    Ua, Da, Ta = A
    Ub, Db, Tb = B

    mat1 = Ta / Tb
    lmul!(Diagonal(Da), mat1)
    mat2 = Ua' * Ub
    rmul!(mat2, Diagonal(Db))
    U, D, T = UDT(mat1 + mat2)

    mul!(mat1, Ua, U)
    mul!(mat2, T, Tb)
    UDT(mat1, D, mat2)
end
