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
    G = P * Diagonal(ni) * invP
    system.isReal && return WalkerProfile{Float64, eltype(expβϵ), Float64}(walker.weight[spin], expβϵ, P, invP, real(G))
    return WalkerProfile{ComplexF64, eltype(expβϵ), eltype(G)}(walker.weight[spin], expβϵ, P, invP, G)
end
