using CanEnsAFQMC, JLD

function mcrun_fidelity(
    system::System, qmc::QMC,
    dir::Int64, file_id::Int64
)
    """
    MC simulation for fidelity measurements
    """
    β = system.β / 2

    if dir == 1
        walker = Walker(system, qmc)

        p_list = Float64[]
        system.isReal ? sign_p_list = Float64[] : sign_p_list = ComplexF64[]

        for i in 1 : qmc.nwarmups
            reverse_sweep!(system, qmc, walker)
            sweep!(system, qmc, walker)
        end

        for i in 1 : qmc.nsamples

            for j in 1 : qmc.measure_interval
                reverse_sweep!(system, qmc, walker)
                sweep!(system, qmc, walker)
            end
            
            tmp = measure_TransitProb(system, system.μ, walker)
            push!(p_list, tmp[1])
            push!(sign_p_list, tmp[2])
        end

        filename = "Fidelity_denom_Lx$(system.Ns[1])_Ly$(system.Ns[2])_U$(system.U)_beta$(β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end

    elseif dir == 2
        walker = GeneralGCWalker(system, qmc)

        p_list = Float64[]
        system.isReal ? sign_p_list = Float64[] : sign_p_list = ComplexF64[]

        for i in 1 : qmc.nwarmups
            reverse_sweep!(system, qmc, walker)
            sweep!(system, qmc, walker)
        end

        for i in 1 : qmc.nsamples

            for j in 1 : qmc.measure_interval
                reverse_sweep!(system, qmc, walker)
                sweep!(system, qmc, walker)
            end
            
            tmp = measure_TransitProb(system, qmc, system.μ, walker)
            push!(p_list, tmp[1])
            push!(sign_p_list, tmp[2])
        end

        filename = "Fidelity_num_Lx$(system.Ns[1])_Ly$(system.Ns[2])_U$(system.U)_beta$(β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end
    end
end
