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
    4.0,
    # chemical potential used for the GCE calculations
    0.5,
    ### AFQMC Constants ###
    # inverse temperature (β)
    10.0,
    # number of imaginary time slices L = β / Δτ
    200
)

const qmc = QMC(
    ### MCMC (Metropolis) ###
    system,
    # number of warm-up runs
    10,
    # number of Metropolis samples per processor
    Int64(10),
    ### Branching Ramdom Walk ###
    # number of repeated random walks
    1,
    # number of walkers
    1,
    ### Numerical Stablization ###
    # using QRCP?
    true,
    # stablization interval
    10,
    # control/calibration interval
    5,
    ### Optimizations ###
    # low-rank truncation?
    true,
    # threshold for low-rank truncation
    1e-6,
    # repartition scheme?
    false, 
    # threshold for repartition
    0.02
)

function mcrun(worker_id, system, qmc)
    # initialize two copies of walker
    walker = Walker(system, qmc)

    T = system.isReal ? Float64 : ComplexF64
    sample_list = Vector{RegSample{T}}()

    ### Monte Carlo Sampling ###
    ####### Warm-up Step #######
    for i in 1 : qmc.nwarmups
        # sweep the entire space-time lattice
        walker = sweep!(system, qmc, walker)
    end
    for i in 1 : qmc.nsamples
        walker = sweep!(system, qmc, walker)
        walker_profile = [WalkerProfile(system, walker, 1), WalkerProfile(system, walker, 2)]

        # energy measurements
        Ek, Ep, Etot = measure_energy(system, walker_profile)

        push!(sample_list, RegSample{T}(sgn(prod(walker.weight)), Ek, Ep, Etot))
    end

    filename = "Lx$(system.Ns[1])_Ly$(system.Ns[2])_U$(system.U)_beta$(system.β)_$(worker_id)"
    jldopen("../data/$filename.jld", "w") do file
        addrequire(file, CanEnsAFQMC)
        write(file, "sample_list", sample_list)
    end
end
