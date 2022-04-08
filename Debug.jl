using LinearAlgebra, Statistics, Random, Printf, StaticArrays
const FloatType = Union{Float64, ComplexF64, BigFloat, Complex{BigFloat}}
const MatrixType = Union{Matrix{Float64}, Matrix{ComplexF64}, Diagonal{Float64, Vector{Float64}}, Diagonal{ComplexF64, Vector{ComplexF64}}}

include("Variable.jl")
include("MatrixGenerator.jl")
include("Recursion.jl")
include("Metropolis.jl")
include("ConstraintPath.jl")
include("GrandCanonical.jl")
include("MonteCarlo.jl")
include("Measure.jl")
include("QoL.jl")

system = System(
    ### Model Constants ###
    # number of sites in each dimension (NsX, NsY)
    (4, 4),
    # number of spin-ups/downs (nup, ndn)
    (8, 8),
    # hopping constant t
    1.0,
    # on-site repulsion constant U
    2.0,
    # chemical potential used for the GCE calculations
    1.0,
    ### AFQMC Constants ###
    # imaginary time interval (Δτ)
    0.02,
    # number of imaginary time slices L = β / Δτ
    100
)
qmc = QMC(
    # number of processors (not working for now)
    1,
    ### MCMC (Metropolis) ###
    # number of warm-up runs
    5,
    # number of Metropolis samples per processor
    8,
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
    5,
    # control/calibration interval
    10,
    ### Optimizations ###
    # low-rank truncation?
    false,
    # threshold for low-rank truncation
    1e-3
)

const floatπ = convert(Float64, π)
measure = GeneralMeasure(
    system,
    ### Momentum distribution ###
    # symmetry points
    [[0.0, 0.0], [floatπ, floatπ]],
    # discrete points measured between symmetry points
    30
)


println("System size:", system.V, " U:", system.U)
println("Beta:", system.L * system.Δτ, " tau:", system.Δτ)

nk_array = mc_metropolis(system, qmc, measure)
nk_mean = mean(nk_array, dims = 2)
nk_error = std(nk_array, dims = 2) / sqrt(qmc.nsamples)
for i = 1 : length(measure.DFTmats)
    println(nk_mean[i], "    ", nk_error[i])
end

"""
println("System size:", system.V, " U:", system.U)
println("Beta:", system.L * system.Δτ, " tau:", system.Δτ)

etg = EtgMeasure(6, [i for i = 1:3], [j for j = 0:3])
expS2, expS2n = mc_replica(system, qmc, etg)

expS2_mean = mean(expS2)
expS2_error = std(expS2) / sqrt(qmc.nsamples)
println("Trace of squared RDM: ", expS2_mean, "    ", expS2_error)
S2_mean = -log.(expS2_mean)
S2_error = expS2_error ./ expS2_mean
println("Renyi-2 Entropy: ", S2_mean, "    ", S2_error)

expS2n_mean = mean(expS2n, dims = 3)[:, :, 1]
expS2n_error = std(expS2n, dims = 3)[:, :, 1] / sqrt(qmc.nsamples)
P2n = expS2n_mean / expS2_mean
P2n_error = sqrt.((expS2_error / expS2_mean)^2 .+ (expS2n_error ./ expS2n_mean).^2)
println("Particle sector distribution: ")
for j = 1 : length(etg.k)
    for i = 1 : length(etg.k)
        println(i, "    ", j, "    ", P2n[i, j], "    ", P2n_error[i, j])
    end
end
"""