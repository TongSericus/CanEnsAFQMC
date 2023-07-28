"""
    Correlation function measurements
"""

struct CorrFuncSampler
    # counters
    s_counter::Base.RefValue{Int}

    # all possible δr vectors
    δr::Vector{Tuple{Int, Int}}

    ipδr::Matrix{Int}

    ## observables ##
    # charge correlation
    nᵢ₊ᵣnᵢ::Matrix{ComplexF64}
    # spin correlation
    # z-direction spin-spin correlation
    Sᵢ₊ᵣSᵢ::Matrix{ComplexF64}
    # x/y-direction spin-spin correlation
    Sˣᵢ₊ᵣSˣᵢ::Matrix{ComplexF64}
    # single-pair correlation (it is averaged over nᵢ₊ᵣꜛnᵢꜜ and nᵢ₊ᵣꜜnᵢꜛ)
    nᵢ₊ᵣꜛnᵢꜜ::Matrix{ComplexF64}
    # pair correlation
    Pₛ::Matrix{ComplexF64}
end

function CorrFuncSampler(system::System, qmc::QMC; nsamples::Int = qmc.nsamples)
    δr = Tuple{Int, Int}[]
    δrx_max = div(system.Ns[1],2)
    δry_max = div(system.Ns[2],2)

    δr = [(δrx, δry) for δrx in 0 : δrx_max for δry in 0 : δry_max]

    Lx, Ly, _ = system.Ns
    L = Lx * Ly
    x = collect(0:L-1) .% Lx       # x positions for sites
    y = div.(collect(0:L-1), Lx)   # y positions for sites

    ipδr = zeros(Int, L, length(δr))
    for (n, i) in enumerate(δr)
        δrx, δry = i
        @views copyto!(ipδr[:, n], (x .+ δrx) .% Lx .+ Lx * ((y .+ δry) .% Ly) .+ 1)
    end

    nᵢ₊ᵣnᵢ = zeros(ComplexF64, length(δr), nsamples)
    Sᵢ₊ᵣSᵢ = zeros(ComplexF64, length(δr), nsamples)
    Sˣᵢ₊ᵣSˣᵢ = zeros(ComplexF64, length(δr), nsamples)
    nᵢ₊ᵣꜛnᵢꜜ = zeros(ComplexF64, length(δr), nsamples)
    Pₛ = zeros(ComplexF64, length(δr), nsamples)

    return CorrFuncSampler(Ref(1), δr, ipδr, nᵢ₊ᵣnᵢ, Sᵢ₊ᵣSᵢ, Sˣᵢ₊ᵣSˣᵢ, nᵢ₊ᵣꜛnᵢꜜ, Pₛ)
end

"""
    measure_ChargeCorr(system::System)

    Compute charge-charge correlation function:
    ⟨nᵢ₊ᵣnᵢ⟩ = N⁻¹∑ᵢ⟨(nᵢ₊ᵣ↑+nᵢ₊ᵣ↓)(nᵢ↑+nᵢ↓)⟩
"""
function measure_ChargeCorr(
    sampler::CorrFuncSampler, ρ₊::DensityMatrix, ρ₋::DensityMatrix;
    addCount::Bool = false
)
    s = sampler.s_counter[]
    ρ₁₊ = ρ₊.ρ₁
    ρ₁₋ = ρ₋.ρ₁
    nᵢ₊ᵣnᵢ = sampler.nᵢ₊ᵣnᵢ

    Ns⁻¹ = 1 / length(sampler.ipδr[:, 1])
    @inbounds for n in 1:length(sampler.δr)
        for (i,ipδr) in enumerate(@view sampler.ipδr[:, n])
                nᵢ₊ᵣnᵢ[n, s] += ρ₂(ρ₊, ipδr, ipδr, i, i) + ρ₂(ρ₋, ipδr, ipδr, i, i) + 
                                ρ₁₊[ipδr, ipδr] * ρ₁₋[i, i] + ρ₁₊[i, i] * ρ₁₋[ipδr, ipδr]
        end
        nᵢ₊ᵣnᵢ[n, s] *= Ns⁻¹
    end

    addCount && (sampler.s_counter[] += 1)

    return nothing
end

"""
    measure_SpinCorr(system::System)

    Compute second-order spin-order in z-direction:
    ⟨Sᵢ₊ᵣSᵢ⟩ = N⁻¹∑ᵢ⟨(nᵢ₊ᵣ↑-nᵢ₊ᵣ↓)(nᵢ↑-nᵢ↓)⟩
"""
function measure_SpinCorr(
    sampler::CorrFuncSampler, ρ₊::DensityMatrix, ρ₋::DensityMatrix;
    addCount::Bool = false
)
    s = sampler.s_counter[]
    ρ₁₊ = ρ₊.ρ₁
    ρ₁₋ = ρ₋.ρ₁
    Sᵢ₊ᵣSᵢ = sampler.Sᵢ₊ᵣSᵢ
    Sˣᵢ₊ᵣSˣᵢ = sampler.Sˣᵢ₊ᵣSˣᵢ

    Ns⁻¹ = 1 / length(sampler.ipδr[:, 1])
    @inbounds for n in 1:length(sampler.δr)
        for (i,ipδr) in enumerate(@view sampler.ipδr[:, n])
                Sᵢ₊ᵣSᵢ[n, s] += ρ₂(ρ₊, ipδr, ipδr, i, i) + ρ₂(ρ₋, ipδr, ipδr, i, i) - 
                                ρ₁₊[ipδr, ipδr] * ρ₁₋[i, i] - ρ₁₊[i, i] * ρ₁₋[ipδr, ipδr]
                Sˣᵢ₊ᵣSˣᵢ[n, s] += (1 - ρ₁₊[i, ipδr]) * (1 - ρ₁₋[ipδr, i])
        end
        Sᵢ₊ᵣSᵢ[n, s] *= Ns⁻¹
        Sˣᵢ₊ᵣSˣᵢ[n, s] *= Ns⁻¹
    end

    addCount && (sampler.s_counter[] += 1)

    return nothing
end

"""
    measure_PairCorr(system::System)

    Compute the single pair correlation
    ⟨Sᵢ₊ᵣSᵢ⟩ = N⁻¹∑ᵢ⟨(nᵢ₊ᵣ↑-nᵢ₊ᵣ↓)(nᵢ↑-nᵢ↓)⟩

    and the s-wave pair correlation
    Pₛ(i+r, i) = ⟨c⁺ᵢ₊ᵣ↑c⁺ᵢ₊ᵣ↓cᵢ↓cᵢ↑ + c⁺ᵢ↑c⁺ᵢ↓cᵢ₊ᵣ↓cᵢ₊ᵣ↑⟩ = ρꜛᵢ₊ᵣ,ᵢρ↓ᵢ₊ᵣ,ᵢ + ρꜛᵢ,ᵢ₊ᵣρ↓ᵢ,ᵢ₊ᵣ
"""
function measure_PairCorr(
    sampler::CorrFuncSampler, ρ₊::DensityMatrix, ρ₋::DensityMatrix;
    addCount::Bool = false
)
    s = sampler.s_counter[]
    ρ₁₊ = ρ₊.ρ₁
    ρ₁₋ = ρ₋.ρ₁
    nᵢ₊ᵣꜛnᵢꜜ = sampler.nᵢ₊ᵣꜛnᵢꜜ
    Pₛ = sampler.Pₛ

    Ns⁻¹ = 1 / length(sampler.ipδr[:, 1])
    @inbounds for n in 1:length(sampler.δr)
        for (i,ipδr) in enumerate(@view sampler.ipδr[:, n])
            nᵢ₊ᵣꜛnᵢꜜ[n, s] += (ρ₁₊[ipδr, ipδr]*ρ₁₋[i, i] + ρ₁₋[ipδr, ipδr]*ρ₁₊[i, i]) / 2
            Pₛ[n, s] += ρ₁₊[ipδr, i]*ρ₁₋[ipδr, i] + ρ₁₊[i, ipδr]*ρ₁₋[i, ipδr]
        end
        nᵢ₊ᵣꜛnᵢꜜ[n, s] *= Ns⁻¹
        Pₛ[n, s] *= Ns⁻¹
    end

    addCount && (sampler.s_counter[] += 1)

    return nothing
end
