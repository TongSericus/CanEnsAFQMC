using CanEnsAFQMC, JLD

const system = Hubbard(
    ### Model Constants ###
    # number of sites in each dimension (NsX, NsY)
    (6, 6),
    # number of spin-ups/downs (nup, ndn)
    (18, 18),
    # hopping constant t
    1.0,
    # on-site repulsion constant U
    2.0,
    # chemical potential used for the GCE calculations
    0.5,
    ### AFQMC Constants ###
    # inverse temperature (β)
    1.0,
    # number of imaginary time slices L = β / Δτ
    100
)

const qmc = QMC(
    ### MCMC (Metropolis) ###
    system,
    # number of warm-up runs
    50,
    # number of Metropolis samples per processor
    Int64(1e4),
    ### Branching Ramdom Walk ###
    # number of repeated random walks
    1,
    # number of walkers
    1,
    ### Numerical Stablization ###
    # using QRCP?
    false,
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

const etg = EtgMeasure(
    # Site indices of the subsystem
    [collect(1:6), collect(7:36),
    collect(1:12), collect(13:36),
    collect(1:18), collect(19:36)
])

function replica_run(worker_id, system, qmc)
    # initialize two copies of walker
    walker1 = Walker(system, qmc)
    walker2 = Walker(system, qmc)

    T = system.isReal ? Float64 : ComplexF64
    sample_list = Vector{EtgSample{T, system.N[1] + 1, system.N[2] + 1}}()

    ### Monte Carlo Sampling ###
    ####### Warm-up Step #######
    for i in 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        sweep!_replica(system, qmc, walker1, walker2)
    end
    for i in 1 : qmc.nsamples
        sample = EtgSample{T, system.N[1] + 1, system.N[2] + 1}()
        sweep!_replica(system, qmc, walker1, walker2)
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
            if sum(expS2n_up) / expS2_up - 1 > 1e-10
                jldopen("../data/unstable_config$(k)_up.jld", "w") do file
                    addrequire(file, CanEnsAFQMC)
                    write(file, "walker_list", [walker1, walker2])
                end
                return nothing
            end
            if sum(expS2n_dn) / expS2_dn - 1 > 1e-10
                jldopen("../data/unstable_config$(k)_dn.jld", "w") do file
                    addrequire(file, CanEnsAFQMC)
                    write(file, "walker_list", [walker1, walker2])
                end
                return nothing
            end
            # merge
            push!(sample.expS2, expS2_up * expS2_dn)
            push!(sample.expS2n_up, expS2n_up)
            push!(sample.expS2n_dn, expS2n_dn)
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
