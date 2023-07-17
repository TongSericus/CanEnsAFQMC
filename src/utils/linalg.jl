### A Struct Defined to Store a String of Matrices or Factorizations ###
Base.@kwdef struct Cluster{T}
    B::Vector{T}
end

Base.prod(C::Cluster{T}, a::Vector{Int64}) where T = @views prod(C.B[a])

Cluster(Ns::Int, N::Int; T::DataType = Float64) = (T == Float64) ? 
                                                Cluster(B = [Matrix((1.0I)(Ns)) for _ in 1 : N]) : 
                                                Cluster(B = [Matrix(((1.0+0.0im)*I)(Ns)) for _ in 1 : N])
Cluster(A::Factorization{T}, N::Int64) where T = Cluster(B = [similar(A) for _ in 1 : N])

import StableLinearAlgebra as Slinalg

# Set LDR equal to the identity matrix
reset!(S::LDR) = ldr!(S, I)

# iteration for destructuring into components
Base.iterate(S::LDR) = (S.L, Val(:d))
Base.iterate(S::LDR, ::Val{:d}) = (S.d, Val(:R))
Base.iterate(S::LDR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::LDR, ::Val{:done}) = nothing

Base.similar(S::LDR{T, E}) where {T, E} = ldr(S)

# Diagonalization
LinearAlgebra.eigvals(F::LDR{T, E}) where {T, E} = eigvals(Diagonal(F.d) * F.R * F.L, sortby = abs)
