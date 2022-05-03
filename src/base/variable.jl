export QMC
export GeneralMeasure, EtgMeasure

struct QMC
    """
        All static parameters needed for QMC.
        
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
    ntotsamples::Int64
    ### Branching Ramdom Walk ###
    nblocks::Int64
    nwalkers::Int64
    ntotwalkers::Int64
    ### Numerical Stablization ###
    stab_interval::Int64
    update_interval::Int64
    ### Optimizations ###
    isLowrank::Bool
    lrThreshold::Float64
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