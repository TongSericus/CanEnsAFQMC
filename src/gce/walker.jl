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
    F, cluster = run_full_propagation(auxfield, system, qmc, K = div(L, qmc.stab_interval))

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

### A New Scheme using StableLinearAlgebra Package ###
struct TempDataGC{Tf<:Number, F<:Factorization{Tf}, C}
    """
        Preallocated data
        
        FC -> list of all partial factorizations
        Fτ = (Q, D, T) -> matrix decompositions of the walker are stored and updated as the propagation goes
        FM -> merge of Ft and FC[t]
        P -> Poisson-binomial matrix
    """
    FC::Cluster{F}
    Fτ::Vector{F}
    FM::Vector{F}
    cluster::Cluster{C}
end

struct GCWalker{Tw<:Number, Ts<:Number, T<:Number, F<:Factorization{T}, C}
    """
        GCE walker
    """
    weight::Vector{Tw}
    sign::Vector{Ts}
    # Use reference to make chemical potential tunable on the fly
    expβμ::Base.RefValue{Float64}
    auxfield::Matrix{Int64}
    F::Vector{F}
    ws::LDRWorkspace{T}
    G::Vector{Matrix{T}}
    tempdata::TempDataGC{T, F, C}
    cluster::Cluster{C}
end

function GCWalker(
    system::System, qmc::QMC; 
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1, μ = system.μ
)
    Ns = system.V
    k = qmc.stab_interval
    system.isReal ? T = Float64 : T = ComplexF64

    weight = zeros(Float64, 2)
    sign = zeros(T, 2)

    G = [Matrix{T}(undef, Ns, Ns), Matrix{T}(undef, Ns, Ns)]
    ws = ldr_workspace(G[1])
    F, cluster, Fcluster = run_full_propagation(auxfield, system, qmc, ws)

    tempdata = TempDataGC(
        Fcluster,
        ldrs(G[1], 2), ldrs(G[1], 2),
        Cluster(Ns, 2 * k)
    )

    expβμ = exp(system.β * μ)
    weight[1], sign[1] = inv_IpμA!(G[1], F[1], expβμ, ws)
    weight[2], sign[2] = inv_IpμA!(G[2], F[2], expβμ, ws)

    return GCWalker(-weight, sign, Ref(expβμ), auxfield, F, ws, G, tempdata, cluster)
end
