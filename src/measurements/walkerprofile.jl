struct WalkerProfile{T1,T2,T3 <: FloatType}
    """
        All the measurement-related information of a single walker
    """
    weight::T1
    expβϵ::Vector{T2}
    P::Matrix{T2}
    invP::Matrix{T2}
    G::Matrix{T3}
end

function WalkerProfile(system::System, walker::Walker, spin::Int64)
    expβϵ, P = eigen(walker.F[spin])
    invP = inv(P)
    ni = occ_projection(system.V, system.N[spin], expβϵ)
    t = sum(real(ni) .< 1e-6) + 1 : length(ni)
    G = @views P[:, t] * Diagonal(ni[t]) * invP[t, :]

    system.isReal && return WalkerProfile{Float64, eltype(expβϵ), Float64}(walker.weight[spin], expβϵ, P, invP, real(G))
    return WalkerProfile{ComplexF64, eltype(expβϵ), eltype(G)}(walker.weight[spin], expβϵ, P, invP, G)
end

function ovlp_coeff(
    P::Matrix{T}, invP::Matrix{T}, 
    i::Int64, j::Int64, k::Int64, l::Int64
    ) where {T<:FloatType}
    """
    Return a matrix with the (λ, ν)th element being
    P^{-1}[λ, i] * P[j, λ] * P^{-1}[ν, k] * P[l, ν]
    """
    return @views (invP[:, i] .* P[j, :]) * (invP[:, k] .* P[l, :])'
end

function compute_G2(
    system::System,
    ni::Vector{T}, nij::Matrix{T},
    P::Matrix{M}, invP::Matrix{M}
) where {T<:FloatType, M<:FloatType}
    """
    Calculate the elements of the one/two-body density matrices
    # Arguments
    ni -> ⟨n_{i}⟩, level occupancy in the eigen basis
    nij -> ⟨n_{i} n_{j}⟩, second-order correlation matrix in the eigen basis
    P -> transformation matrix

    # Returns
    G2 -> two-body density matrix
    """
    G2 = zeros(T, system.V, system.V, system.V, system.V)
    
    for l = 1 : system.V
        for k = 1 : system.V
            for j = 1 : system.V
                for i = 1 : system.V
                    G2[i, j, k, l] = sum(
                        ovlp_coeff(P, invP, i, j, k, l) .* nij -
                        ovlp_coeff(P, invP, i, l, k, j) .* (nij .- ni)
                    )
                end
            end
        end
    end

    return G2
end