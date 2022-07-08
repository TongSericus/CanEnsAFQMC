struct GCEWalker{W<:FloatType, T<:FloatType, F<:Factorization{T}, C}
    """
        GCE walker
    """
    sgn::Vector{W}
    logweight::Vector{W}
    auxfield::Matrix{Int64}
    F::Vector{F}
    cluster::Cluster{C}
end

function calc_pf(β::Float64, μ::Float64, expβϵ::Vector{T}) where {T<:FloatType}
    λ = expβϵ * exp(β * μ)
    λ .+= 1
    sgn = prod(sign.(λ))
    logZ = sum(log.(abs.(λ)))
    return sgn, logZ
end

function GCEWalker(system::System, qmc::QMC)
    """
        GCE walker
    """
    # initialize a random field configuration
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1
    F, cluster = initial_propagation(auxfield, system, qmc)
    # diagonalize the decomposition
    λ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight
    sgn1, logZ1 = calc_pf(system.β, system.μ, λ[1])
    sgn2, logZ2 = calc_pf(system.β, system.μ, λ[2])

    system.isReal && return GCEWalker{Float64, eltype(F[1].U), typeof(F[1]), eltype(cluster.B)}(real([sgn1, sgn2]), [logZ1, logZ2], auxfield, F, cluster)
    return GCEWalker{ComplexF64, eltype(F[1].U), typeof(F[1]), eltype(cluster.B)}([sgn1, sgn2], [logZ1, logZ2], Z, auxfield, F, cluster)
end

function GCEWalker(system::System, qmc::QMC, μ0::Float64)
    """
        GCE walker
    """
    # initialize a random field configuration
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1
    F, cluster = initial_propagation(auxfield, system, qmc)
    # diagonalize the decomposition
    λ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight
    sgn1, logZ1 = calc_pf(system.β, μ0, λ[1])
    sgn2, logZ2 = calc_pf(system.β, μ0, λ[2])

    system.isReal && return GCEWalker{Float64, eltype(F[1].U), typeof(F[1]), eltype(cluster.B)}(real([sgn1, sgn2]), [logZ1, logZ2], auxfield, F, cluster)
    return GCEWalker{ComplexF64, eltype(F[1].U), typeof(F[1]), eltype(cluster.B)}([sgn1, sgn2], [logZ1, logZ2], Z, auxfield, F, cluster)
end
