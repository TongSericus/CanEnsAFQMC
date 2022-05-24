using CanEnsAFQMC, JLD, Measurements
include("../src/analysis/reblocking.jl")

function data_analysis_etgent(filenames::Vector{String})
    expS2_array = [Float64[] for _ = 1 : 6]
    expS2n_array = [Vector{Float64}[] for _ = 1 : 6]
    for filename in filenames
        data = JLD.load("../data/$filename", "sample_list")
        for i = 1 : 6
        expS2_array[i] = [expS2_array[i]; [data[n].expS2[i] for n = 1 : length(data)]]
        expS2n_array[i] = [expS2n_array[i]; [sum_antidiagonal(data[n].expS2n_up[i] * data[n].expS2n_dn[i]') for n = 1 : length(data)]]
        end
    end

    for i = 1 : 6
        println("Subsystem A$i:")
        expS2_avg, expS2_err = reblock(expS2_array[i])
        expS2 = measurement(expS2_avg, expS2_err)
        S2 = -log(expS2)
        println("Renyi-2 EE: $S2")
        expS2n = Measurement{Float64}[]
        for j = 1 : length(expS2n_array[i][1])
            expS2n_avg, expS2n_err = reblock([expS2n_array[i][n][j] for n = 1 : length(expS2n_array[i])])
            push!(expS2n, measurement(expS2n_avg, expS2n_err))
        end
        P2n = expS2n / expS2
        cutoff = (P2n .> 0) .* (P2n .> 1e-10)
        cutoff_ind = findall(x -> x, cutoff)
        Hα = 2 * log(sum(sqrt.(P2n[cutoff_ind])))
        println("Shannon Entropy: $Hα")
        S2op = S2 - Hα
        println("Accessible EE: $S2op")
    end
end
