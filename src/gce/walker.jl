"""
    GC walker definitions
"""
abstract type GCWalker end

struct TempDataGC{Tf<:Number, F<:Factorization{Tf}, C}
    """
        Preallocated data
    """
    # All partial factorizations
    FC::Cluster{F}

    # Factorization of two unit matrices for spin-up and spin-down
    Fτ::Vector{F}

    # Preallocated multiplication of FC[i] and Fτ[i]
    FM::Vector{F}

    # Temporal array of matrices with the ith element B̃_i being
    #   B̃_i = B_{(cidx-1)k + i}, where cidx is the index of the current imaginary time slice to be updated
    # Note that the spin-up and spin-down matrices are strored as 
    #   the first and the second half of the array, respectively
    cluster::Cluster{C}
end

struct HubbardGCWalker{Tw<:Number, Ts<:Number, T<:Number, F<:Factorization{T}, E, C} <: GCWalker
    """
        GC walker for specifically the Hubbard model that allows for rank-1 update
    """
    α::Matrix{Float64}

    # Statistical weights of the walker, stored in the logarithmic form, while signs are the phases
    weight::Vector{Tw}
    sign::Vector{Ts}

    # Use reference to make chemical potential tunable on the fly
    expβμ::Base.RefValue{Float64}

    auxfield::Matrix{Int64}
    F::Vector{F}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    # Temporal data used in each cluter update to avoid memory allocations
    tempdata::TempDataGC{T, F, C}

    # Temporal array of matrices with the ith element B̃_i being
    #   B̃_i = B_{(i-1)k + 1} ⋯ B_{ik}, where k is the stablization interval defined as qmc.stab_interval
    # Note that the spin-up and spin-down matrices are strored as 
    #   the first and the second half of the array, respectively
    cluster::Cluster{C}
end

function HubbardGCWalker(system::System, qmc::QMC; auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1, μ = system.μ)
    """
        Initialize a Hubbard GCE walker
    """
    Ns = system.V
    k = qmc.stab_interval
    system.isReal ? T = Float64 : T = ComplexF64

    weight = zeros(Float64, 2)
    sign = zeros(T, 2)

    G = [Matrix{T}(undef, Ns, Ns), Matrix{T}(undef, Ns, Ns)]
    ws = ldr_workspace(G[1])
    F, cluster, Fcluster = run_full_propagation_reverse(auxfield, system, qmc, ws)

    tempdata = TempDataGC(
        Fcluster,
        ldrs(G[1], 2), ldrs(G[1], 2),
        Cluster(Ns, 2 * k)
    )

    expβμ = exp(system.β * μ)
    weight[1], sign[1] = inv_IpμA!(G[1], F[1], expβμ, ws)
    weight[2], sign[2] = inv_IpμA!(G[2], F[2], expβμ, ws)

    α = system.auxfield[1, 1] / system.auxfield[2, 1]
    α = [α - 1 1/α - 1; 1/α - 1 α - 1]

    return HubbardGCWalker(α, -weight, sign, Ref(expβμ), auxfield, F, ws, G, tempdata, cluster)
end

### GC Walker for General Model ###
struct GeneralGCWalker{Tw<:Number, Ts<:Number, T<:Number, F<:Factorization{T}, E, C} <: GCWalker
    # Statistical weights of the walker, stored in the logarithmic form, while signs are the phases
    weight::Vector{Tw}
    sign::Vector{Ts}

    # Use reference to make chemical potential tunable on the fly
    expβμ::Base.RefValue{Float64}

    auxfield::Matrix{Int64}
    F::Vector{F}
    ws::LDRWorkspace{T, E}
    G::Vector{Matrix{T}}

    # Temporal data used in each cluter update to avoid memory allocations
    tempdata::TempDataGC{T, F, C}

    # Temporal array of matrices with the ith element B̃_i being
    #   B̃_i = B_{(i-1)k + 1} ⋯ B_{ik}, where k is the stablization interval defined as qmc.stab_interval
    # Note that the spin-up and spin-down matrices are strored as 
    #   the first and the second half of the array, respectively
    cluster::Cluster{C}
end

function GeneralGCWalker(
    system::System, qmc::QMC; 
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1, μ = system.μ
)
    """
    Initialize a general GCE walker
    """
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

    return GeneralGCWalker(-weight, sign, Ref(expβμ), auxfield, F, ws, G, tempdata, cluster)
end
