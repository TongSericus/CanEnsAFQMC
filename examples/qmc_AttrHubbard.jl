"""
    QMC scripts to perform MC runs, measurements, etc.
"""

using CanEnsAFQMC, JLD

function qmc_run_AttrHubbard(system::System, qmc::QMC, path::String, filename::String)

    # initialize walker, density matrix and sampler, subscripts with + and - represent up and down spins respectively
    walker = Walker(system, qmc)
    ρ₊ = DensityMatrix(system, Nft=10)
    ρ₋ = DensityMatrix(system, Nft=10)
    corr_sampler = CorrFuncSampler(system, qmc)
    pn_sampler₊ = PnSampler(system, qmc, system.Aidx, Nft=10)
    pn_sampler₋ = PnSampler(system, qmc, system.Aidx, Nft=10)

    bins = qmc.measure_interval

    # warm-up steps to thermalize the walker
    sweep!(system, qmc, walker, loop_number=qmc.nwarmups)

    for i in 1 : qmc.nsamples
        sweep!(system, qmc, walker, loop_number=bins)

        # update density matrix
        update!(system, walker, ρ₊, 1)
        update!(system, walker, ρ₋, ρ₊)

        # measure correlation functions
        measure_SpinCorr(corr_sampler, ρ₊, ρ₋)

        # measure probability distributions
        measure_Pn(pn_sampler₊, ρ₊)
        measure_Pn(pn_sampler₋, ρ₋)
    end

    # save the results
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "cicj", corr_sampler.cᵢ₊ᵣcᵢ)
        write(file, "SiSj", corr_sampler.Sᵢ₊ᵣSᵢ)
        write(file, "Pn_up", pn_sampler₊.Pn)
        write(file, "Pn_dn", pn_sampler₋.Pn)
    end
end
