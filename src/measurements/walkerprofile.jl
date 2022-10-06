struct WalkerProfile{W<:FloatType, E<:FloatType, Tg<:FloatType}
    """
        All the measurement-related information of a single walker
    """
    weight::W
    expβϵ::Vector{E}
    P::Matrix{E}
    invP::Matrix{E}
    G::Matrix{Tg}
end

function computeG(F::UDTlr, N::Int64)
    λ, Pocc, invPocc = eigen(F)
    ni = occ_recursion(length(λ), N, λ)
    G = Pocc * Diagonal(ni) * invPocc
end

function WalkerProfile(system::System, walker::Walker, spin::Int64)
    λ, Pocc, invPocc = eigen(walker.F[spin])
    ni = occ_recursion(length(λ), system.N[spin], λ)
    G = Pocc * Diagonal(ni) * invPocc

    return WalkerProfile{eltype(walker.weight), eltype(λ), eltype{G}}(walker.weight[spin], λ, Pocc, invPocc, G)
end
