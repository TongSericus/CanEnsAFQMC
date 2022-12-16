using CanEnsAFQMC, JLD

function mcrun_purity_ce(
    system_og::System, qmc_og::QMC, 
    system_ext::System, qmc_ext::QMC, 
    direction::Int64, file_id::Int64
)
    """
    MC simulation for purity measurements
    """
    if direction == 1
        walker = Walker(system_ext, qmc_ext)
        p_list = Float64[]
        system_ext.isReal ? sign_p_list = Float64[] : sign_p_list = ComplexF64[]

        for i in 1 : qmc_ext.nwarmups
            reverse_sweep!(system_ext, qmc_ext, walker)
            sweep!(system_ext, qmc_ext, walker)
        end

        for i in 1 : qmc_ext.nsamples

            for j in 1 : qmc_ext.measure_interval
                reverse_sweep!(system_ext, qmc_ext, walker)
                sweep!(system_ext, qmc_ext, walker)
            end
            
            tmp = measure_TransitProb(system_og, qmc_og, walker)
            push!(p_list, tmp[1])
            sgn = prod(walker.sign) * tmp[2]
            system_ext.isReal ? push!(sign_p_list, real(sgn)) : push!(sign_p_list, sgn)
        end

        filename = "PurityCE_denom_Lx$(system_og.Ns[1])_Ly$(system_og.Ns[2])_U$(system_og.U)_beta$(system_og.β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end

    elseif direction == 2
        walker1 = Walker(system_og, qmc_og)
        walker2 = Walker(system_og, qmc_og)
        p_list = Float64[]
        system_og.isReal ? sign_p_list = Float64[] : sign_p_list = ComplexF64[]

        for i in 1 : qmc_og.nwarmups
            reverse_sweep!(system_og, qmc_og, walker1, walker2)
            sweep!(system_og, qmc_og, walker1, walker2)
        end

        for i in 1 : qmc_og.nsamples

            for j in 1 : qmc_og.measure_interval
                reverse_sweep!(system_og, qmc_og, walker1, walker2)
                sweep!(system_og, qmc_og, walker1, walker2)
            end
            
            tmp = measure_TransitProb(system_ext, qmc_ext, walker1, walker2)
            push!(p_list, tmp[1])
            sgn = prod(walker1.sign) * prod(walker2.sign) * tmp[2]
            system_og.isReal ? push!(sign_p_list, real(sgn)) : push!(sign_p_list, sgn)
        end

        filename = "PurityCE_num_Lx$(system_og.Ns[1])_Ly$(system_og.Ns[2])_U$(system_og.U)_beta$(system_og.β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end
    end
end

function mcrun_purity_gce(
    system_og::System, qmc_og::QMC, 
    system_ext::System, qmc_ext::QMC, 
    direction::Int64, file_id::Int64
)
    """
    MC simulation for purity measurements
    """
    if direction == 1
        walker = HubbardGCWalker(system_ext, qmc_ext)
        p_list = Float64[]
        sign_p_list = Float64[]

        for i in 1 : qmc_ext.nwarmups
            sweep!(system_ext, qmc_ext, walker)
        end

        for i in 1 : qmc_ext.nsamples

            for j in 1 : qmc_ext.measure_interval
                sweep!(system_ext, qmc_ext, walker)
            end
            
            tmp = measure_TransitProb(system_og, qmc_og, walker)
            push!(p_list, tmp[1])
            push!(sign_p_list, real(tmp[2]))
        end

        filename = "PurityGCE_denom_Lx$(system_og.Ns[1])_Ly$(system_og.Ns[2])_U$(system_og.U)_beta$(system_og.β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end

    elseif direction == 2
        walker1 = HubbardGCWalker(system_og, qmc_og)
        walker2 = HubbardGCWalker(system_og, qmc_og)
        p_list = Float64[]
        sign_p_list = Float64[]

        for i in 1 : qmc_og.nwarmups
            sweep!(system_og, qmc_og, walker1, walker2)
        end

        for i in 1 : qmc_og.nsamples

            for j in 1 : qmc_og.measure_interval
                sweep!(system_og, qmc_og, walker1, walker2)
            end
            
            tmp = measure_TransitProb(system_ext, qmc_ext, walker1, walker2)
            push!(p_list, tmp[1])
            push!(sign_p_list, real(tmp[2]))
        end

        filename = "PurityGCE_num_Lx$(system_og.Ns[1])_Ly$(system_og.Ns[2])_U$(system_og.U)_beta$(system_og.β)_$(file_id).jld"
        jldopen("../data/$filename", "w") do file
            write(file, "p_list", p_list)
            write(file, "sign_p_list", sign_p_list)
        end
    end
end
