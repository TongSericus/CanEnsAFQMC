"""
    QMC scripts to perform MC runs, measurements, etc.
"""

using CanEnsAFQMC, JLD

function qmc_run_AttrHubbard(system::System, qmc::QMC, path::String, filename::String)

    Nft = qmc.num_FourierPoints

    # initialize walker, density matrix and sampler, subscripts with + and - represent up and down spins respectively
    walker = Walker(system, qmc)
    ρ₊ = DensityMatrix(system, Nft=Nft)
    ρ₋ = DensityMatrix(system, Nft=Nft)
    corr_sampler = CorrFuncSampler(system, qmc)
    #pn_sampler₊ = PnSampler(system, qmc, system.Aidx, Nft=Nft)
    #pn_sampler₋ = PnSampler(system, qmc, system.Aidx, Nft=Nft)

    bins = qmc.measure_interval

    # warm-up steps to thermalize the walker
    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for i in 1 : qmc.nsamples
        sweep!(system, qmc, walker, loop_number=bins)

        # update density matrix
        update!(system, walker, ρ₊, 1)
        update!(system, walker, ρ₋, 2)

        # measure correlation functions
        measure_ChargeCorr(corr_sampler, ρ₊, ρ₋)
        measure_SpinCorr(corr_sampler, ρ₊, ρ₋, addCount=true)

        # measure probability distributions
        #measure_Pn(pn_sampler₊, ρ₊)
        #measure_Pn(pn_sampler₋, ρ₋)
    end

    # save the results
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "ninj", corr_sampler.nᵢ₊ᵣnᵢ)
        write(file, "SxiSxj", corr_sampler.Sˣᵢ₊ᵣSˣᵢ)
        write(file, "SiSj", corr_sampler.Sᵢ₊ᵣSᵢ)
        #write(file, "Pn_up", pn_sampler₊.Pn)
        #write(file, "Pn_dn", pn_sampler₋.Pn)
    end
end
