struct WalkerProfile{T<:FloatType}
    """
        All the measurement-related information of a single walker
    """
    weight::Float64
    expβϵ::Vector{T}
    P::Matrix{T}
    invP::Matrix{T}
    G::Matrix{T}
end

function WalkerProfile(system::System, walker::Walker, spin::Int64)
    walker_eigen = eigen(walker.D[spin] * walker.T[spin] * walker.Q[spin], sortby = abs)
    P = walker.Q[spin] * walker_eigen.vectors
    ni = occ_projection(system.Ns, system.N[spin], walker_eigen.values, system.expiφ)

    return WalkerProfile(
        walker.weight[spin],
        walker_eigen.values,
        P, inv(P), 
        P * Diagonal(ni) * inv(P)
    )

end