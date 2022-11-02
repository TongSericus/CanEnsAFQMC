"""
    QMC Walker Definitions
"""
### A Struct Defined to Store a String of Matrices or Factorizations ###
Base.@kwdef struct Cluster{T}
    B::Vector{T}
end

Base.prod(C::Cluster{T}, a::Vector{Int64}) where T = @views prod(C.B[a])

Cluster(Ns::Int64, N::Int64) = Cluster(B = [Matrix(1.0I, Ns, Ns) for _ in 1 : N])
Cluster(A::Factorization{Tf}, N::Int64) where Tf = Cluster(B = [similar(A) for _ in 1 : N])

### Random Walker in the Canonical Ensemble ###
struct TempData{Tf<:Number, Tp<:Number, F<:Factorization{Tf}, C}
    """
        Preallocated data
    """
    # All partial factorizations
    FC::Cluster{F}

    # Factorization of two unit matrices for spin-up and spin-down
    Fτ::Vector{F}

    # Preallocated multiplication of FC[i] and Fτ[i]
    FM::Vector{F}

    # Preallocated matrix to store all intermidate values in Poisson binomial recursion
    P::Matrix{Tp}

    # Temporal array of matrices with the ith element B̃_i being
    #   B̃_i = B_{(cidx-1)k + i}, where cidx is the index of the current imaginary time slice to be updated
    # Note that the spin-up and spin-down matrices are strored as 
    #   the first and the second half of the array, respectively
    cluster::Cluster{C}
end

struct Walker{Tw<:Number, Ts<:Number, T<:Number, F<:Factorization{T}, E, C}
    """
        All the MC information is stored in a walker struct
    """
    # Statistical weights of the walker, stored in the logarithmic form, while signs are the phases
    weight::Vector{Tw}
    sign::Vector{Ts}

    auxfield::Matrix{Int64}
    F::Vector{F}
    ws::LDRWorkspace{T, E}

    # Temporal data used in each cluter update to avoid memory allocations
    tempdata::TempData{T, Ts, F, C}

    # Temporal array of matrices with the ith element B̃_i being
    #   B̃_i = B_{(i-1)k + 1} ⋯ B_{ik}, where k is the stablization interval defined as qmc.stab_interval
    # Note that the spin-up and spin-down matrices are strored as 
    #   the first and the second half of the array, respectively
    cluster::Cluster{C}
end

function Walker(
    system::Hubbard, qmc::QMC; 
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1,
    Tw::DataType = Float64,
    Ts::DataType = ComplexF64)
    """
        Initialize a walker with a random configuration
    """
    L = system.L
    Ns = system.V
    N = system.N

    k = qmc.stab_interval
    ϵ = qmc.lrThld

    ws = ldr_workspace(zeros(Tw, Ns, Ns))

    F, cluster, Fcluster = run_full_propagation(auxfield, system, qmc, ws)

    tempdata = TempData(
        Fcluster, 
        similar.(F), similar.(F), 
        zeros(Ts, system.V+1, system.V), 
        Cluster(Ns, 2 * k)
    )

    weight = zeros(Tw, 2)
    sgn = zeros(Ts, 2)

    weight[1], sgn[1] = compute_PF(F[1], system.N[1], PMat=tempdata.P)
    weight[2], sgn[2] = compute_PF(F[2], system.N[2], PMat=tempdata.P)

    return Walker(weight, sgn, auxfield, F, ws, tempdata, cluster)
end
