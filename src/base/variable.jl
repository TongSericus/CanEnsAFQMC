"""
    QMC simulations parameters
        
    nsamples -> number of Metropolis samples
    nwarmups -> number of warm-up steps
    measure_interval -> number of sweeps between two measurements

    stab_interval -> number of matrix multiplications before the stablization
    update_interval -> number of steps after which the population control and calibration is required

    isLowrank -> if enabling the low-rank truncation
    lrThld -> low-rank truncation threshold (from above)
"""
struct QMC
    ### Simulation-related ###
    nwarmups::Int64
    nsamples::Int64
    measure_interval::Int64
    
    ### Numerical Stablization ###
    stab_interval::Int64
    K::Int64
    K_interval::Vector{Int64}

    ### MCMC-related ###
    useClusterUpdate::Bool
    cluster_list::Vector{Vector{Int}}
    num_FourierPoints::Int
    forceSymmetry::Bool
    useHeatbath::Bool
    saveRatio::Bool

    ### Speed-ups / Approximations ###
    isLowrank::Bool
    lrThld::Float64

    function QMC(
        system::System; 
        nwarmups::Int64 = 512, 
        nsamples::Int64 = 1024, 
        measure_interval::Int64 = 1,
        stab_interval::Int64 = 10,
        useClusterUpdate::Bool = false,
        cluster_list::Vector{Vector{Int}} = [[i] for i in 1:system.V],
        num_FourierPoints::Int = system.V + 1,
        forceSymmetry::Bool = false,
        isLowrank::Bool = false,
        lrThld::Float64 = 1e-6,
        useHeatbath::Bool = false,
        saveRatio::Bool = false
    )
        # number of clusters
        system.L % stab_interval == 0 ? K = div(system.L, stab_interval) : K = div(system.L, stab_interval) + 1
        # group the rest of the matrices as the last cluster
        Le = mod(system.L, stab_interval)
        K_interval = [stab_interval for _ in 1 : K]
        Le == 0 || (K_interval[end] = Le)

        return new(
            nwarmups, nsamples, measure_interval,
            stab_interval, K, K_interval,
            useClusterUpdate, cluster_list, 
            num_FourierPoints,
            forceSymmetry,
            useHeatbath, saveRatio,
            isLowrank, lrThld
        )
    end
end
