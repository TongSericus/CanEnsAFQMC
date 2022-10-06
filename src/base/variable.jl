export QMC
export GeneralMeasure, EtgMeasure

struct QMC
    """
        All static parameters needed for QMC.
        
        nsamples -> number of Metropolis samples
        nwarmups -> number of warm-up steps
        nblocks -> number of repeated random walks
        nwalkers -> number of walkers
        isCP -> Use Column-Pivoting QR decomposion
        stab_interval -> number of matrix multiplications before the stablization
        update_interval -> number of steps after which the population control and calibration is required
        isLowrank -> if enabling the low-rank truncation
        lrThld -> low-rank truncation threshold (from above)
    """
    ### MCMC (Metropolis) ###
    nwarmups::Int64
    nsamples::Int64
    ### Numerical Stablization ###
    isCP::Bool
    stab_interval::Int64
    K::Int64
    update_interval::Int64
    ### Speed-ups ###
    isLowrank::Bool
    lrThld::Float64
    isRepart::Bool
    rpThld::Float64

    function QMC(system::System, nwarmups::Int64, nsamples::Int64,
        isCP::Bool, stab_interval::Int64, update_interval::Int64,
        isLowrank::Bool, lrThld::Float64,
        isRepart::Bool, rpThld::Float64
    )
        (system.L % stab_interval == 0) || @error "# of time slices should be divisible by the stablization interval"
        return new(
            nwarmups, nsamples,
            isCP, stab_interval, div(system.L, stab_interval), update_interval,
            isLowrank, lrThld, isRepart, rpThld
        )
    end
end

struct GeneralMeasure
    """
        Parameters for measuring regular observables
    """
    ### Momentum distribution constants ###
    kpath::Vector{Vector{Float64}}
    DFTmats::Vector{Matrix{ComplexF64}}

    function GeneralMeasure(
        system::System, kpath::Vector{Vector{Float64}}
    )
        rmat = generate_rmat(system)
        DFTmats = generate_DFTmat(kpath, rmat)
        return new(kpath, DFTmats)
    end
end

struct EtgMeasure
    """
        Constants for measuring entanglement entropy

        Aidx -> List of site indices in subsystem A
        LA -> Size of the subsystem A
    """
    Aidx::Vector{Vector{Int64}}
    LA::Vector{Int64}

    function EtgMeasure(Aidx::Vector{Vector{Int64}})
        return new(
            Aidx, length.(Aidx)
        )
    end
end