include("./purity.jl")

const worker_id = parse(Int64, get(ENV, "SLURM_ARRAY_TASK_ID", 1))
@show worker_id

const system_og = Hubbard(
    # system size
    (6, 6), 
    # num. of spin-up/down electrons
    (18, 18),
    # t
    1.0,
    # U
    4.0,
    # μ
    2.0,
    # β
    10.0,
    # L = β / Δτ
    150
)

const qmc_og = QMC(
    system_og, 
    # num. of warm-up steps
    200, 
    # num. of samples
    Int64(1e3), 
    # num. of sweeps between two samples
    100,
    # use QRCP?
    true, 
    # stablization interval
    5, 5, 
    # use low-rank truncation? error tolerance
    true, 1e-8, 
    # use the repartition scheme? error tolerance
    false, 0.01
)

const system_ext = Hubbard(
    # system size
    (6, 6), 
    # num. of spin-up/down electrons
    (9, 9),
    # t
    1.0,
    # U
    4.0,
    # μ
    2.0,
    # β
    2 * β[id],
    # L = β / Δτ
    2 * L[id]
)

const qmc_ext = QMC(
    system_ext, 
    # num. of warm-up steps
    200, 
    # num. of samples
    Int64(1e3), 
    # num. of sweeps between two samples
    100,
    # use QRCP?
    true, 
    # stablization interval
    5, 5, 
    # use low-rank truncation? error tolerance
    true, 1e-8, 
    # use the repartition scheme? error tolerance
    false, 0.01
)

const rng_seed = worker_id + 1234
@show rng_seed
Random.seed!(rng_seed)

mcrun_purity_ce(system_og, qmc_og, system_ext, qmc_ext, 1, worker_id)
