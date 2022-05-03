using CanEnsAFQMC, JLD

system = Hubbard(
    ### Model Constants ###
    # number of sites in each dimension (NsX, NsY)
    (4, 2),
    # number of spin-ups/downs (nup, ndn)
    (4, 4),
    # hopping constant t
    1.0,
    # on-site repulsion constant U
    1.0,
    # chemical potential used for the GCE calculations
    0.5,
    ### AFQMC Constants ###
    # imaginary time interval (Δτ)
    0.02,
    # number of imaginary time slices L = β / Δτ
    250
)

qmc = QMC(
    # number of processors (not working for now)
    1,
    ### MCMC (Metropolis) ###
    # number of warm-up runs
    1,
    # number of Metropolis samples per processor
    10,
    # number of Metropolis samples (not working for now)
    1,
    ### Branching Ramdom Walk ###
    # number of repeated random walks
    1,
    # number of walkers per processor
    1,
    # total number of walkers
    20,
    ### Numerical Stablization ###
    # stablization interval
    10,
    # control/calibration interval
    10,
    ### Optimizations ###
    # low-rank truncation?
    false,
    # threshold for low-rank truncation
    1e-3
)

etg = EtgMeasure(
    # Site indices of the subsystem
    [[1, 2, 3, 4],
    [1, 5],
    [1, 2, 5, 6],
    [1, 2, 3, 5, 6, 7]
])

function replica_run(worker_id, system, qmc)
    # initialize two copies of walker
    walker1 = Walker(system, qmc)
    walker2 = Walker(system, qmc)
    temp1 = QDT(
        deepcopy(walker1.Q),
        deepcopy(walker1.D),
        deepcopy(walker1.T)
    )
    temp2 = QDT(
        deepcopy(walker2.Q),
        deepcopy(walker2.D),
        deepcopy(walker2.T)
    )
    sample_list = Vector{MCSample}()

    ### Monte Carlo Sampling ###
    ####### Warm-up Step #######
    for i = 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        sweep!_replica(system, qmc, walker1, walker2, temp1, temp2)
    end
    for i = 1 : qmc.nsamples
        sample = MCSample()
        sweep!_replica(system, qmc, walker1, walker2, temp1, temp2)
        walker1_profile = [WalkerProfile(system, walker1, 1), WalkerProfile(system, walker1, 2)]
        walker2_profile = [WalkerProfile(system, walker2, 1), WalkerProfile(system, walker2, 2)]

        # Sign
        push!(sample.sgn, sgn(prod(walker1.weight) * prod(walker2.weight)))

        # Entanglement measurements
        for k = 1 : length(etg.Aidx)
            # spin-up sector
            expS2_up, expS2n_up = measure_renyi2_entropy(system, etg.Aidx[k], 1, walker1_profile[1], walker2_profile[1])
            # spin-down sector
            expS2_dn, expS2n_dn = measure_renyi2_entropy(system, etg.Aidx[k], 2, walker1_profile[2], walker2_profile[2])
            # merge
            push!(sample.expS2, expS2_up * expS2_dn)
            push!(sample.expS2n, expS2n_up * expS2n_dn')
        end

        push!(sample_list, sample)

    end

    beta = system.Δτ * system.L
    filename = "Replica_Lx$(system.Ns[1])_Ly$(system.Ns[2])_U$(system.U)_beta$(beta)_$(worker_id)"
    jldopen("../data/$filename.jld", "w") do file
        addrequire(file, CanEnsAFQMC)
        write(file, "sample_list", sample_list)
    end
end
