using CanEnsAFQMC, JLD

function mcrun_StructFact_ce(
    system::System, qmc::QMC, file_id::Int64
)
    """
    MC simulation for energy measurements
    """
    N = system.N

    nk = Vector{Float64}[]
    Cq = Float64[]
    Sq = Float64[]
    sgn = Float64[]

    DFTmats = generate_DFTmats(system)
    # q = (π, π) point
    DFTmat = DFTmats[end]

    walker = Walker(system, qmc)
    DMup= DensityMatrices(system)
    DMdn= DensityMatrices(system)

    for i in 1 : qmc.nwarmups
        reverse_sweep!(system, qmc, walker)
        sweep!(system, qmc, walker)
    end

    for i in 1 : qmc.nsamples

        for j in 1 : qmc.measure_interval
            reverse_sweep!(system, qmc, walker)
            sweep!(system, qmc, walker)
        end

        fill_DM!(DMup, walker.F[1], N[1])
        fill_DM!(DMdn, walker.F[2], N[2])

        ninj = measure_ChargeCorr(system, DMup, DMdn)
        sisj = measure_SpinCorr(system, DMup, DMdn)

        push!(nk, measure_MomentumDist(DFTmats, DMup.Do, DMdn.Do))
        push!(Cq, real(sum(DFTmat .* ninj)) / system.V)
        push!(Sq, real(sum(DFTmat .* sisj)) / system.V)
        push!(sgn, real(prod(walker.sign)))
    end

    filename = "StructFactCE_Lx$(system.Ns[1])_Ly$(system.Ns[2])_U$(system.U)_beta$(system.β)_$(file_id).jld"
    jldopen("../data/$filename", "w") do file
        write(file, "sgn", sgn)
        write(file, "nk", nk)
        write(file, "Cq", Cq)
        write(file, "Sq", Sq)
    end
end
