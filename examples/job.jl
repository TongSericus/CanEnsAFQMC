#include("mcrun.jl")
include("benchmark_run.jl")

const worker_id = get(ENV, "SLURM_ARRAY_TASK_ID", 1)
@show worker_id

#replica_run(worker_id, system, qmc)
mcrun(worker_id, system, qmc)