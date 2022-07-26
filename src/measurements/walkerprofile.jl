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

    system.isReal && return WalkerProfile{Float64, eltype(λ), system.V}(walker.weight[spin], λ, Pocc, invPocc, G)
    return WalkerProfile{ComplexF64, eltype(λ), system.V}(walker.weight[spin], λ, Pocc, invPocc, G)
end

function WalkerProfile(system::System, walker::GCEWalker, spin::Int64)
    expβϵ, Pocc, invPocc = eigen(walker.F[spin])
    λ = expβϵ * exp(system.β * system.μ)
    ni = λ ./ (1 .+ λ)
    G = Pocc * Diagonal(ni) * invPocc
    G = SizedMatrix{system.V, system.V, ComplexF64}(complex(G))

    system.isReal && return WalkerProfile{Float64, eltype(expβϵ), system.V}(walker.sgn[spin] * walker.logweight[spin], expβϵ, Pocc, invPocc, G)
    return WalkerProfile{ComplexF64, eltype(expβϵ), system.V}(walker.sgn[spin] * walker.logweight[spin], expβϵ, Pocc, invPocc, G)
end

function WalkerProfile(system::System, walker::GCEWalker, μ::Float64, spin::Int64)
    expβϵ, Pocc, invPocc = eigen(walker.F[spin])
    λ = expβϵ * exp(system.β * μ)
    ni = λ ./ (1 .+ λ)
    G = Pocc * Diagonal(ni) * invPocc
    G = SizedMatrix{system.V, system.V, ComplexF64}(complex(G))

    system.isReal && return WalkerProfile{Float64, eltype(λ), system.V}(walker.sgn[spin] * walker.logweight[spin], expβϵ, Pocc, invPocc, G)
    return WalkerProfile{ComplexF64, eltype(expβϵ), system.V}(walker.sgn[spin] * walker.logweight[spin], expβϵ, Pocc, invPocc, G)
end
