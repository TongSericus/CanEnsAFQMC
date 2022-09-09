"""
    Tuning the chemical potential dynamically using the method
    described in doi/10.1103/PhysRevE.105.045311
"""

Base.@kwdef struct MuTuner{T<:FloatType}
    μ::Float64
    μt::Vector{Float64}
    sgn::Vector{T} = []
    Nt::Vector{T} = []
    Nsqdt::Vector{T} = []
end

function dynamical_tuning(
    system::System, qmc::QMC, μ0::Float64, T::Int64, mT::Int64;
    tuner = MuTuner{Float64}(μ0, [μ0], [], [], [])
)
    α = system.V / system.U
    walker = GCEWalker(system, qmc, μ=tuner.μ)
    μt = tuner.μ
    for t in 1 : T
        # reconstruct the walker
        expβμ = exp(system.β * μt)
        walker = GCEWalker{Float64, eltype(walker.cluster.B)}(walker.α, expβμ, walker.auxfield, walker.G, walker.cluster)
        # in case the sign problem is severe, collect multiple samples before averaging
        for i = 1 : mT
            walker = sweep!(system, qmc, walker)
            sgn = sign(det(walker.G[1])) * sign(det(walker.G[2]))
            push!(tuner.sgn, sgn)

            n = [walker.G[1][i, i] + walker.G[2][i, i] for i in 1 : system.V]
            N = sum(n)
            Nsqd = sum(n * n') - sum(n.^2) + N
            push!(tuner.Nt, real(N * sgn))
            push!(tuner.Nsqdt, real(Nsqd * sgn))
        end

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
