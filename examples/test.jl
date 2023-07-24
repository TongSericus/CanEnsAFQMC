using CanEnsAFQMC, GenericLinearAlgebra

Lx, Ly = 8, 8
T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (32, 32),
    # t, U
    T, -6.0,
    # μ
    0.0,
    # β, L
    5.0, 50,
    # subsystem indices (if one chooses to measure local distributions)
    Aidx=collect(1:8),
    # data type of the system
    sys_type=Float64,
    # if use charge decomposition
    useChargeHST=true,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    nwarmups=128, nsamples=1024, measure_interval=5,
    # stablization interval
    stab_interval=10,
    # use cluster update, i.e., flipping multiple spins instead of a single spin in every Metropolis test
    useClusterUpdate=true,
    # indices of lattice sites being simultaneously flipped
    cluster_size=3,
    # number of Fourier points used in measurements
    num_FourierPoints=15,
    # enforce symmetry between two spin sectors
    forceSymmetry=true,
    # use low-rank approximation and set the threshold
    isLowrank=true, lrThld=1e-6,
    # debugging flag
    saveRatio=true
)

Random.seed!(1238)

walker = Walker(system, qmc)
sweep!(system, qmc, walker, loop_number=10)