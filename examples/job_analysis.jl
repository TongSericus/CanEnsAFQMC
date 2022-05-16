include("data_analysis.jl")

#const worker_id = get(ENV, "SLURM_ARRAY_TASK_ID", 1)
#@show worker_id

const filenames = ["Replica_Lx4_Ly2_U6.0_beta2.0_$(id).jld" for id = 1 : 4]

data_analysis_etgent(filenames)
