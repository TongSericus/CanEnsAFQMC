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
    expβϵ, P = eigen(walker.F[spin])
    invP = inv(P)
    ni = occ_projection(system.V, system.N[spin], expβϵ, system.expiφ)
    G = P * Diagonal(ni) * invP
    WalkerProfile(walker.weight[spin], expβϵ, P, invP, G)
end
