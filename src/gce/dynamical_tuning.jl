"""
    Tuning the chemical potential dynamically using the method
    described in doi/10.1103/PhysRevE.105.045311
"""

Base.@kwdef struct MuTuner{T<:Number}
    μ::Base.RefValue{Float64}
    μt::Vector{Float64}
    sgn::Vector{T} = []
    Nt::Vector{T} = []
    Nt_avg::Base.RefValue{Float64}
    N²t::Vector{T} = []
end

function dynamical_tuning(
    system::Hubbard, qmc::QMC, μ0::Float64, T::Int64, mT::Int64;
    tuner = MuTuner{Float64}(Ref(μ0), [μ0], [], [], Ref(0.0), []),
    # energy scale
    α::E = system.V / system.U,
    # lower bound for the average sign
    sgn_min::E = 0.1
) where {E<:AbstractFloat}
    # target particle number
    Nₒ = sum(system.N)

    walker = HubbardGCWalker(system, qmc, μ=tuner.μ[])
    μt = tuner.μ[]

    DM = [DensityMatrices(system), DensityMatrices(system)]

    for t in 1 : T
        # change mu of the walker to the new average
        walker.expβμ[] = exp(system.β * μt)
        # in case the sign problem is severe, collect multiple samples before averaging
        for i = 1 : mT
            sweep!(system, qmc, walker)
            
            sgn = prod(walker.sign)
            push!(tuner.sgn, sgn)

            fill_DM!(DM[1], walker.G[1])
            fill_DM!(DM[2], walker.G[2])

            n = tr(DM[1].Do) + tr(DM[2].Do)
            N = sum(n)
            N²t = sum(n * n') - sum(n.^2) + N
            push!(tuner.Nt, real(N * sgn))
            push!(tuner.N²t, real(N²t * sgn))
        end

        t_half = Int64(ceil(length(tuner.μt) / 2))

        sgn_avg = mean(@view tuner.sgn[t_half : end])
        sgn_avg = max(sgn_avg, sgn_min)

        μt_avg = mean(@view tuner.μt[t_half : end])

        Nt_avg = mean(@view tuner.Nt[t_half : end]) / sgn_avg
        tuner.Nt_avg[] = Nt_avg
        N²t_avg = mean(@view tuner.N²t[t_half : end]) / sgn_avg

        μt_var = @views mean(tuner.μt[t_half : end].^2) - μt_avg^2
        Nt_var = N²t_avg - Nt_avg^2

        κ_fluc = system.β * Nt_var
        κ_min = α / sqrt(length(tuner.μt) + 1)

        if μt_var <= 0
            κ_max = system.V
        else
            κ_max = sqrt(abs(Nt_var / μt_var))
        end
        κt = max(κ_min, min(κ_max, κ_fluc))

        μt = μt_avg + (Nₒ - Nt_avg) / κt
        push!(tuner.μt, μt)
        tuner.μ[] = μt
    end

    return tuner
end
