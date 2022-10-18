"""
    Matrix Operations
    Steal and modify some lines from the StableDQMC package
    for CE and more general calculation purposes
    See (https://github.com/carstenbauer/StableDQMC.jl)
"""
### QRCP (column-pivoting) decomposion ###
struct UDT{T} <: Factorization{T}
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

collectU(S::Vector{UDT{Ts}}) where Ts = [S[i].U for i in 1 : length(S)]
collectD(S::Vector{UDT{Ts}}) where Ts = [S[i].D for i in 1 : length(S)]
collectT(S::Vector{UDT{Ts}}) where Ts = [S[i].T for i in 1 : length(S)]

Base.copyto!(F::UDT{T}, S::UDT{T}, ignore...) where {T} = let
    copyto!(F.U, S.U)
    copyto!(F.D, S.D)
    copyto!(F.T, S.T)

    nothing
end

Base.Matrix(S::UDT) = (S.U * Diagonal(S.D)) * S.T

# stable linear algebra operations
LinearAlgebra.det(S::UDT) = prod(S.D) * det(S.U) * det(S.T)
LinearAlgebra.eigvals(S::UDT) = eigvals(Diagonal(S.D) * S.T * S.U, sortby = abs)
LinearAlgebra.eigen(S::UDT) = let 
    eigenS = eigen(Diagonal(S.D) * S.T * S.U, sortby = abs)
    LinearAlgebra.Eigen(eigenS.values, S.U * eigenS.vectors)
end

UDT(n::Int64) = UDT(Matrix(1.0I, n, n), ones(Float64, n), Matrix(1.0I, n, n))

function UDT(A::AbstractMatrix{T}) where {T<:Number}
    F = qr!(A, Val(true))
    n = size(F.R)
    D = Vector{T}(undef, n[1])
    R = F.R
    @views F.p[F.p] = 1 : n[2]

    @inbounds for i in 1 : n[1]
        D[i] = abs(real(R[i, i]))
    end
    lmul!(Diagonal(1 ./ D), R)
    UDT(Matrix(F.Q), D, R[:, F.p])
end

function UDT(
    D::Vector{Td}, P::Matrix{Tp}, invP::Matrix{Tp}
) where {Td<:Number, Tp<:Number}
    """
    Transform P * D * P^-1 into UDT form
    """
    F = UDT(P * Diagonal(D))
    UDT(F.U, F.D, F.T * invP)
end

function QR_lmul(B::AbstractMatrix, A::UDT)
    """
    Compute B * UaDaTa
    """
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDT(mat)
    UDT(F.U, F.D, F.T * A.T)
end

function QR_rmul(A::UDT, B::AbstractMatrix)
    """
    Compute UaDaTa * B
    """
    mat = A.T * B
    lmul!(Diagonal(A.D), mat)
    F = UDT(mat)
    UDT(A.U * F.U, F.D, F.T)
end

function QR_merge(A::UDT, B::UDT)
    """
    Compute UaDaTa * UbDbTb
    """
    mat = A.T * B.U
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDT(mat)
    UDT(A.U * F.U, F.D, F.T * B.T)
end

function QR_merge!(C::UDT, A::UDT, B::UDT)
    """
    Compute UaDaTa * UbDbTb
    """
    mat = A.T * B.U
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDT(mat)

    mul!(C.U, A.U, F.U)
    mul!(C.T, F.T, B.T)
    @inbounds for i in 1 : length(F.D)
        C.D[i] = F.D[i]
    end

    return C
end

function QR_sum(A::UDT, B::UDT)
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

### Regular QR decomposion ###
struct UDR{T} <: Factorization{T}
    U::Matrix{T}
    D::Vector{T}
    R::Matrix{T}
end
Base.Matrix(F::UDR) = (F.U * Diagonal(F.D)) * F.R

# iteration for destructuring into components
Base.iterate(S::UDR) = (S.U, Val(:D))
Base.iterate(S::UDR, ::Val{:D}) = (S.D, Val(:R))
Base.iterate(S::UDR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::UDR, ::Val{:done}) = nothing

Base.similar(S::UDR) = UDR(similar(S.U), similar(S.D), similar(S.R))

# stable linear algebra operations
LinearAlgebra.det(S::UDR) = prod(S.D) * det(S.R) * det(S.U)
LinearAlgebra.eigvals(S::UDR) = eigvals((Diagonal(S.D) * S.R) * S.U, sortby = abs)
LinearAlgebra.eigen(S::UDR) = let 
    eigenS = eigen((Diagonal(S.D) * S.R) * S.U, sortby = abs)
    LinearAlgebra.Eigen(eigenS.values, S.U * eigenS.vectors)
end

UDR(n::Int64) = UDR{Float64}(Matrix(1.0I, n, n), ones(Float64, n), Matrix(1.0I, n, n))

function UDR(A::AbstractMatrix{T}) where {T<:Number}
    F = qr!(A)
    n = size(A, 1)
    D = Vector{T}(undef, n)
    R = F.R

    @inbounds for i in 1 : n
        D[i] = abs(R[i, i])
    end
    lmul!(Diagonal(1 ./ D), R)
    UDR(Matrix(F.Q), D, R)
end

function QR_lmul(B::AbstractMatrix, A::UDR)
    """
    Compute B * UaDaRa
    """
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDR(mat)
    UDR(F.U, F.D, F.R * A.R)
end

function QR_lmul!(B::AbstractMatrix, A::UDR)
    """
    Compute B * UaDaRa in-place
    """
    mat = B * A.U
    rmul!(mat, Diagonal(A.D))
    F = UDR(mat)
    A.U .= F.U
    A.D .= F.D
    mat = F.R * A.R
    A.R .= mat
    return A
end

function QR_rmul(A::UDR, B::AbstractMatrix)
    """
    Compute UaDaRa * B
    """
    mat = A.R * B
    lmul!(Diagonal(A.D), mat)
    F = UDR(mat)
    UDR(A.U * F.U, F.D, F.R)
end

function QR_rmul!(A::UDR, B::AbstractMatrix)
    """
    Compute UaDaRa * B in-place
    """
    mat = A.R * B
    lmul!(Diagonal(A.D), mat)
    F = UDR(mat)
    mat = A.U * F.U
    A.U .= mat
    A.D .= F.D
    A.R .= F.R
    return A
end

function QR_merge(A::UDR, B::UDR)
    """
    Compute UaDaRa * UbDbRb
    """
    mat = A.R * B.U
    lmul!(Diagonal(A.D), mat)
    rmul!(mat, Diagonal(B.D))
    F = UDR(mat)
    UDR(A.U * F.U, F.D, F.R * B.R)
end
