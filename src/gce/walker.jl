struct GCEWalker{T<:FloatType, C}
    """
        GCE walker
    """
    α::Matrix{Float64}
    expβμ::Float64
    auxfield::Matrix{Int64}
    G::Vector{Matrix{T}}
    cluster::Cluster{C}
end

function GCEWalker(system::System, qmc::QMC; auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1, μ = system.μ)
    """
        GCE walker
    """
    L = size(auxfield)[2]
    (L % qmc.stab_interval == 0) || @error "# of time slices should be divisible by the stablization interval"
    F, cluster = initial_propagation(auxfield, system, qmc, K = div(L, qmc.stab_interval))

    expβμ = exp(system.β * μ)
    # Factorizations are shifted by exp(-ΔτK) to make the following updates rank-1
    F = [shiftB(F[1], system.Bk), shiftB(F[2], system.Bk)]
    F = [computeG(F[1], expβμ), computeG(F[2], expβμ)]
    G = [Matrix(F[1]), Matrix(F[2])]

    α = system.auxfield[1, 1] / system.auxfield[2, 1]
    α = [α - 1 1/α - 1; 1/α - 1 α - 1]

    system.isReal && return GCEWalker{Float64, eltype(cluster.B)}(α, expβμ, auxfield, G, cluster)
    return GCEWalker{ComplexF64, eltype(cluster.B)}(α, expβμ, auxfield, G, cluster)
end

unshiftG(walker::GCEWalker, system::System) = [inv(system.Bk) * walker.G[1] * system.Bk, inv(system.Bk) * walker.G[2] * system.Bk]
