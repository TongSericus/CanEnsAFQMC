using CanEnsAFQMC, JLD, Measurements
include("../src/analysis/reblocking.jl")

function data_analysis(filenames::Vector{String})
    #U = [1.0, 1.5, 2.0, 3.0, 3.4, 3.8, 4.0, 4.5, 5.0, 5.2, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0]
    #U = collect(2.3:0.1:3.6)
    #i = 1
    for filename in filenames
        sgn_array = Float64[]
	    N_array = Float64[]
	    #Ep_array = Float64[]
        Etot_array = Float64[]
        data = JLD.load("../data/$filename", "sample_list")
        for n = 1 : length(data)
            s = data[n].sgn
            push!(sgn_array, s)
  	        push!(N_array, data[n].N * s)
	        #push!(Ep_array, data[n].Ep * s)
            push!(Etot_array, data[n].Etot * s)
        end

        sgn_avg, sgn_err = mean(sgn_array), std(sgn_array) / sqrt(length(sgn_array))
        println("Average Sign:", sgn_avg, "     ", "Error Bar:", sgn_err)
	    #println(sgn_avg, "     ", sgn_err)	
	
	    #Ep_avg, Ep_err = mean(Ep_array), std(Ep_array) / sqrt(length(Ep_array))
	    #println(Ep_avg / 16 / U[i], "     ", Ep_err / 16 / U[i])	
	    #i += 1	

	    N_avg, N_err = mean(N_array), std(N_array) / sqrt(length(N_array))
	    println("N:", N_avg / sgn_avg, "    ", "Error:", N_err)

        #Etot_avg, Etot_err = mean(Etot_array), std(Etot_array) / sqrt(length(Etot_array))
        Etot_avg, Etot_err = reblock(Etot_array)
        #println("Total Energy:", Etot_avg / sgn_avg / 36, "     ", "Error Bar:", Etot_err / 36, "\n")
	    println(Etot_avg / sgn_avg / 36, "     ", Etot_err / 36, "\n")
    end
end

function data_analysis_fluc(filenames::String)
    data = JLD.load("../data/$filename", "sample_list")
    L = length(data)
    Dupup = zeros(Measurement{Float64}, 36, 36)
    Dupdn = zeros(Measurement{Float64}, 36, 36)
    Ddndn = zeros(Measurement{Float64}, 36, 36)

    for j = 1 : 36
        for i = 1 : 36
            tmp_array = [data[n].Dupup[i, j] for n in 1 : L]
            tmp = [mean(tmp_array), std(tmp_array) / sqrt(L)]
            Dupup[i, j] = measurement(tmp[1], tmp[2])

            tmp_array = [data[n].Ddndn[i, j] for n in 1 : L]
            tmp = [mean(tmp_array), std(tmp_array) / sqrt(L)]
            Dupdn[i, j] = measurement(tmp[1], tmp[2])

            tmp_array = [data[n].Dupdn[i, j] for n in 1 : L]
            tmp = [mean(tmp_array), std(tmp_array) / sqrt(L)]
            Ddndn[i, j] = measurement(tmp[1], tmp[2])
        end
    end

    Cij = zeros(Measurement{Float64}, 36, 36)
    for j = 1 : 36
        for i = 1 : 36
            Nsqd = Dupup[i, j] + Dupdn[i, j] + Dupdn[j, i] + Ddndn[i, j]
            N = (Dupup[i, i] + Ddndn[i, i]) * (Dupup[j, j] + Ddndn[j, j])
            Cij[i, j] = Nsqd - N
        end
    end
end

function data_analysis_etgent(filenames::Vector{String})
    expS2_array = [Float64[] for _ = 1 : 2]
    expS2n_array = [Vector{Float64}[] for _ = 1 : 2]
    for filename in filenames
        data = JLD.load("../data/$filename", "sample_list")
        for i = 1 : 2
        expS2_array[i] = [expS2_array[i]; [data[n].expS2[i] for n = 1 : length(data)]]
        expS2n_array[i] = [expS2n_array[i]; [sum_antidiagonal(data[n].expS2n_up[i] * data[n].expS2n_dn[i]') for n = 1 : length(data)]]
        end
    end

    for i = 1 : 2
        println("Subsystem A$i:")
        expS2_avg, expS2_err = mean(expS2_array[i]), std(expS2_array[i]) / length(expS2_array[i])
        expS2 = measurement(expS2_avg, expS2_err)
        S2 = -log(expS2)
        println("Renyi-2 EE: $S2")
        expS2n = Measurement{Float64}[]
        for j = 1 : length(expS2n_array[i][1])
            tmp_array = [expS2n_array[i][n][j] for n = 1 : length(expS2n_array[i])]
            expS2n_avg, expS2n_err = mean(tmp_array), std(tmp_array) / length(tmp_array)
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
