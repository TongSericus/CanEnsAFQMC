struct WalkerProfile{W<:FloatType, E<:FloatType, Ns}
    """
        All the measurement-related information of a single walker
    """
    weight::W
    expβϵ::Vector{E}
    P::Matrix{E}
    invP::Matrix{E}
    G::SizedMatrix{Ns, Ns, ComplexF64}
end

function WalkerProfile(system::System, walker::Walker, spin::Int64)
    λ, Pocc, invPocc = eigen(walker.F[spin])
    ni = occ_recursion(length(λ), system.N[spin], λ)
    G = Pocc * Diagonal(ni) * invPocc
    G = SizedMatrix{system.V, system.V, ComplexF64}(complex(G))

    return WalkerProfile{eltype(walker.weight), eltype(λ), system.V}(walker.weight[spin], λ, Pocc, invPocc, G)
end
