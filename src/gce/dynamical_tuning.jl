Base.@kwdef struct MuTuner{T<:FloatType}
    μ::Float64
    μt::Vector{Float64}
    sgn::Vector{T} = []
    Nt::Vector{T} = []
    Nsqdt::Vector{T} = []
end

function measure_occ_squared(
    system::Hubbard, walker_list::Vector{WalkerProfile{W, E, G}}
) where {W<:FloatType, E<:FloatType, G<:FloatType}
    """
    Compute <N^2>
    """
    n = [walker_list[1].G[i, i] + walker_list[2].G[i, i] for i in 1 : system.V]
    N = sum(n)
    Nsqd = 0
    for i in 1 : system.V
        for j in i + 1 : system.V
            Nsqd += 2 * n[i] * n[j]
        end
    end
    return Nsqd + N, N
end

function dynamical_tuning(
    system::System, qmc::QMC, μ0::Float64, T::Int64;
    tuner = MuTuner{Float64}(μ0, [μ0], [], [], [])
)
    α = system.V / system.U
    walker = GCEWalker(system, qmc, tuner.μ)
    μt = tuner.μ
    for t in 1 : T
        walker = sweep!(system, qmc, walker, μt)
        walker_profile = [WalkerProfile(system, walker, μt, 1), WalkerProfile(system, walker, μt, 2)]
        sgn = prod(walker.sgn)
        push!(tuner.sgn, sgn)
        Nsqd, N = measure_occ_squared(system, walker_profile)
        push!(tuner.Nt, N * sgn)
        push!(tuner.Nsqdt, Nsqd * sgn)

        t_half = Int64(ceil(length(tuner.μt) / 2))
        sgn_avg = mean(@view tuner.sgn[t_half : end])
        μt_avg = mean(@view tuner.μt[t_half : end])
        Nt_avg = mean(@view tuner.Nt[t_half : end]) / sgn_avg
        Nsqdt_avg = mean(@view tuner.Nsqdt[t_half : end]) / sgn_avg

        varμt = @views mean(tuner.μt[t_half : end].^2) - μt_avg^2
        varNt = Nsqdt_avg - Nt_avg^2
        κ_fluc = system.β * varNt
        κ_min = α / sqrt(length(tuner.μt) + 1)
        if varμt <= 0
            κ_max = system.V
        else
            κ_max = sqrt(abs(varNt / varμt))
        end
        κt = max(κ_min, min(κ_max, κ_fluc))

        μt = μt_avg + (sum(system.N) - Nt_avg) / κt
        push!(tuner.μt, μt)
    end

    return MuTuner{Float64}(μt, tuner.μt, tuner.sgn, tuner.Nt, tuner.Nsqdt)
end