struct System
    """
        Constants in the simulation
        
        Ns -> number of sites in each dimension
        V -> volume of the lattce
        N[1] -> number of spin-ups
        N[2] -> number of spin-downs
        t -> hopping constant
        U -> repulsion constant
        kinetic_matrix -> one-body, kinetic matrix (used for one-body measurements)
        μ -> chemical potential used for the GCE calculations
        Δτ -> imaginary time interval
        L -> β / Δτ
        auxfield -> discrete HS variables sorted by field variables (±1) and spins (up/down),
                    for instance, auxfield[2][1] represents spin-up section with σ = -1
        Bk -> exponential of the kinetic matrix
        BT -> trial propagator matrix
        BT_inv -> inverse of trial propagator matrix
    """
    ### Model Constants ###
    Ns::Tuple{Int64, Int64}
    V::Int64
    N::Tuple{Int64, Int64}
    t::Float64
    U::Float64
    kinetic_matrix::Array{Float64,2}
    μ::Float64
    expβμ::Float64
    ### AFQMC Constants ###
    Δτ::Float64
    L::Int64
    ### Automatically-Generated Constants ###
    auxfield::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
    Bk::Array{Float64,2}
    BT::Array{Float64,2}
    BT_inv::Array{Float64,2}

    function System(
        Ns::Tuple{Int64, Int64}, N::Tuple{Int64, Int64},
        t::Float64, U::Float64,
        μ::Float64, Δτ::Float64, L::Int64
    )
        if Ns[2] == 1 
            kinetic_matrix = kinetic_matrix_hubbard1D(Ns[1], t)
        else
            kinetic_matrix = kinetic_matrix_hubbard2D(Ns[1], Ns[2], t)
        end
        γ = atanh(sqrt(tanh(Δτ * U / 4)))
        auxfield = ((exp(2 * γ - Δτ * U / 2), exp(-2 * γ - Δτ * U / 2)),
                (exp(-2 * γ - Δτ * U / 2), exp(2 * γ - Δτ * U / 2)))
        return new(
            Ns, prod(Ns), N, t, U, kinetic_matrix, μ, exp(Δτ * L * μ), Δτ, L, auxfield,
            exp(-kinetic_matrix * Δτ/2), exp(-kinetic_matrix * Δτ), inv(exp(-kinetic_matrix * Δτ))
        )
    end

end

struct QMC
    """
        Struct that contains all static parameters needed for MC.
        
        nprocs -> number of processors
        nsamples -> number of Metropolis samples per processor
        ntot_samples -> number of Metropolis samples
        nwarmups -> number of warm-up steps
        nblocks -> number of repeated random walks
        nwalkers -> number of walkers per processor
        ntot_walkers -> total number of walkers
        stab_interval -> number of matrix multiplications before the stablization
        update_interval -> number of steps after which the population control and calibration is required
        lowrank -> if enabling the low-rank truncation
        lowrank_threshold -> low-rank truncation threshold (from above)
    """
    nprocs::Int64
    ### MCMC (Metropolis) ###
    nwarmups::Int64
    nsamples::Int64
    ntot_samples::Int64
    ### Branching Ramdom Walk ###
    nblocks::Int64
    nwalkers::Int64
    ntot_walkers::Int64
    ### Numerical Stablization ###
    stab_interval::Int64
    update_interval::Int64
    ### Optimizations ###
    lowrank::Bool
    lowrank_threshold::Float64
end

mutable struct Walker{T1<:MatrixType, T2<:MatrixType}
    """
        All the MC information carried by a single walker

        weight -> weight of the walker (spin-up/down portions are stored separately)
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        Q, D, T -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Vector{Float64}
    auxfield::Array{Int64,2}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
    update_count::Int64
end

struct TrialWalker{T1<:MatrixType, T2<:MatrixType}
    """
        Walker information for each trial step
    """
    weight::Vector{Float64}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
end

struct WalkerProfile{T<:FloatType}
    """
        All the (spin-resolved) measurement-related information of a single walker
    """
    weight::Float64
    expβϵ::Vector{T}
    P::Matrix{T}
    invP::Matrix{T}
end

mutable struct ConstrainedWalker{T1<:MatrixType, T2<:MatrixType}
    """
        All the sampling information carried by a single walker

        weight -> weight of the walker
        Pl -> ratio between the weights before and after every update step
        auxfield -> configurations of the walker
        Q, D, T -> matrix decompositions of the walker are stored and updated as the propagation goes
    """
    weight::Float64
    Pl::Vector{Float64}
    auxfield::Array{Int64,2}
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
end

mutable struct Temp{T1<:MatrixType, T2<:MatrixType}
    """
        All the temporal information stored in the simulation
    """
    Q::Vector{T1}
    D::Vector{T2}
    T::Vector{T1}
end

struct GeneralMeasure
    ### Momentum distribution constants ###
    sympts::Vector{Vector{Float64}}
    npoints::Int64
    DFTmats::Vector{Matrix{ComplexF64}}

    function GeneralMeasure(
        system::System, sympts::Vector{Vector{Float64}}, npoints::Int64
    )
        rmat = generate_rmat(system)
        kpath = generate_kpath(sympts, npoints)
        DFTmats = generate_DFTmat(kpath, rmat)
        return new(sympts, npoints, DFTmats)
    end
end

struct EtgMeasure
    """
        Constants for measuring entanglement entropy

        Aidx -> List of site indices in subsystem A
        k -> List of particle number sectors to be resolved
        LA -> Size of the subsystem A
        expiφ -> Discrete Fourier frequencies
    """
    Aidx::Vector{Int64}
    k::Vector{Int64}
    ### Automatically-Generated Constants ###
    LA::Int64
    expiφ::Vector{ComplexF64}

    function EtgMeasure(Ns::Int64, Aidx::Vector{Int64}, k::Vector{Int64})
        return new(
            Aidx, k, length(Aidx),
            exp.(im * [2 * π * m / (Ns + 1) for m = 1 : Ns + 1])
        )
    end

end