"""
    Job scripts to compute purity on Hubbard model
"""

# include the qmc scripts
include("./qmc_purity_ce.jl")

### Specify job-specific parameters to run on a cluster ###
# extract array ID from the environment
const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))
# for this specific job, we vary β
const β_list = collect(1.0:10.0)
const L_list = collect(10:100)
# assign ID to files
const id = mod(worker_id - 1, length(β_list)) + 1
const file_id = div(worker_id - 1, length(β_list)) + 1
@show file_id
### Specify the model parameter ###
# define the kinetic matrix
const Lx, Ly = 8, 8
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

# specify the lattice parameters
const system_og = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (32, 32),
    # t, U
    T, 2.0,
    # μ
    0.0,
    # β, L
    β_list[id], L_list[id],
    # data type of the system
    sys_type=Float64,
    # if use charge decomposition
    useChargeHST=false,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const c = 8
const qmc_og = QMC(
    system_og,
    # number of warm-ups, samples and measurement interval
    nwarmups=512, nsamples=1024, measure_interval=10,
    # stablization interval
    stab_interval=10,
    # use cluster update, i.e., flipping multiple spins instead of a single spin in every Metropolis test
    useClusterUpdate=true,
    # indices of lattice sites being simultaneously flipped
    cluster_list=[collect(c*(i-1)+1:c*i) for i in 1:div(system.V,c)],
    # number of Fourier points used in measurements
    num_FourierPoints=15,
    # enforce symmetry between two spin sectors
    forceSymmetry=true,
    # use low-rank approximation and set the threshold
    isLowrank=true, lrThld=1e-4,
    # debugging flag
    saveRatio=false
)

const system_ext = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (32, 32),
    # t, U
    T, 2.0,
    # μ
    0.0,
    # β, L
    2*β_list[id], 2*L_list[id],
    # data type of the system
    sys_type=Float64,
    # if use charge decomposition
    useChargeHST=false,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc_ext = QMC(
    system_ext,
    # number of warm-ups, samples and measurement interval
    nwarmups=512, nsamples=1024, measure_interval=10,
    # stablization interval
    stab_interval=10,
    # use cluster update, i.e., flipping multiple spins instead of a single spin in every Metropolis test
    useClusterUpdate=true,
    # indices of lattice sites being simultaneously flipped
    cluster_list=[collect(c*(i-1)+1:c*i) for i in 1:div(system.V,c)],
    # number of Fourier points used in measurements
    num_FourierPoints=15,
    # enforce symmetry between two spin sectors
    forceSymmetry=true,
    # use low-rank approximation and set the threshold
    isLowrank=true, lrThld=1e-4,
    # debugging flag
    saveRatio=false
)

# set seed
seed = 1234 + file_id
@show seed
Random.seed!(seed)

path = "../data/Hubbard_Lx8Ly8/purity"
filename = "FT_U$(system.U)_N$(sum(system.N))_Lx$(system.Ns[1])_Ly$(system.Ns[2])_beta$(system.β)_seed$(seed).jld"

# execute
@time qmcrun_purity(system_og, qmc_og, system_ext, qmc_ext, path, filename, direction=1)
