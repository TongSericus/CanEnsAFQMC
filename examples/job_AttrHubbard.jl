"""
    Job scripts to run finite temperature simulations on negative-U Hubbard model
"""

# include the qmc scripts
include("./qmc_AttrHubbard.jl")

### Specify job-specific parameters to run on a cluster ###
# extract array ID from the environment
const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))
# for this specific job, we vary U
const U_list = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
# assign ID to files
const id = mod(worker_id - 1, length(U_list)) + 1
const file_id = div(worker_id - 1, length(U_list)) + 1
@show file_id

### Specify the model parameter ###
# define the kinetic matrix
const Lx, Ly = 8, 8
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)

# specify the lattice parameters
const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (32, 32),
    # t, U
    T, U_list[id],
    # μ
    0.0,
    # β, L
    5.0, 50,
    # subsystem indices (if one chooses to measure local distributions)
    Aidx=collect(1:32),
    # data type of the system
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=false,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

# specify the simulation parameters
"""
    Important Note: One may want to do some trial runs to specify the values for "cluster_list" and "num_FourierPoints" in qmc variable below

    Typically, a single flip would produce a Metropolis acceptence raio (r) close to 1, which means that a local flip is almost
    alway accepted. This is not ideal (we want acceptence ratio close to 0.5) and one should use heat-bath sampling in such
    cases or flip multiple spins. Flipping multiple spins would not change the mean value of r but would increase its variance,
    meaning that large (r>10) or small (r<0.1) values are more likely to appear. Our goal is to make the median of r around 0.5.
    
    To achieve this, turn on "saveRatio" flag in QMC (saveRatio=true) and initialize a random walker with your system and qmc
    and perform a number of trial sweeps (say 5):

    ```
    julia> walker = Walker(system, qmc)
    julia> sweep!(system, qmc, walker, loop_number=5)
    ```

    Now we have collected a vector of acceptence ratios stored in walker.tmp_r, the median is

    ```
    julia> using Statistics
    julia> median(walker.tmp_r)
    ```

    Choose a suitable "cluster_list" to let this median fall into the range of 0.4-0.6

    For number of Fourier points, it represents the number of Fourier frequencies used in the Fourier transform and is used for measurements. 
    Its default value is system.V+1 which corresponds to no approximation but might be expensive. A smaller "num_FourierPoints"
    introduces errors but can be significantly faster for measurements. A trial test for suitable values can be made by computing
    one-body reduced density matrix (1-RDM) with different "num_FourierPoints":

    ```
    julia> ρ_trial = DensityMatrix(system, Nft=qmc.num_FourierPoints)
    julia> update!(system, walker, ρ_trial, 1)
    julia> rdm_trial = ρ_trial.ρ₁
    ```

    and compare it with the exact value

    ```
    julia> ρ_exact = DensityMatrix(system)  # exact calculation is performed when Nft is not specified
    julia> update!(system, walker, ρ_exact, 1)
    julia> rdm_exact = ρ_exact.ρ₁
    julia> norm(rdm_trial - rdm_exact)
    ```

    A good value would make the difference in norm smaller than 1e-10
"""

if abs(system.U) <= 2.0
    c = 16
elseif 2.0 < abs(system.U) <= 4.0
    c = 8
elseif abs(system.U) > 4.0
    c = 4
end
const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    nwarmups=10, nsamples=5, measure_interval=2,
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
seed = 2224 + file_id
@show seed
Random.seed!(seed)

path = "../data/AttrHubbard_Lx8Ly8/"
filename = "FT_U$(system.U)_N$(sum(system.N))_Lx$(system.Ns[1])_Ly$(system.Ns[2])_LA$(length(system.Aidx))_beta$(system.β)_seed$(seed).jld"

# execute
@time qmc_run_AttrHubbard(system, qmc, path, filename)
