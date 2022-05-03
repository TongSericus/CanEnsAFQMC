struct WalkerProfile{T1<:FloatType, T2<:FloatType}
    """
        All the measurement-related information of a single walker
    """
    weight::ComplexF64
    expβϵ::Vector{T1}
    P::Matrix{T1}
    invP::Matrix{T1}
    G::Matrix{T2}
end

function WalkerProfile(system::System, walker::Walker, spin::Int64)
    walker_eigen = eigen(walker.D[spin] * walker.T[spin] * walker.Q[spin], sortby = abs)
    P = walker.Q[spin] * walker_eigen.vectors
    ni = occ_projection(system.V, system.N[spin], walker_eigen.values, system.expiφ)

    return WalkerProfile(
        walker.weight[spin],
        walker_eigen.values,
        P, inv(P), 
        P * Diagonal(ni) * inv(P)
    )

end
