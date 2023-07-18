"""
    Measure particle/spin number distribution in a subsystem
"""

struct PnSampler
    # partition
    Aidx::Vector{Int}

    # observables
    Pn::Matrix{ComplexF64}  # probability distribution

    # counters
    s_counter::Base.RefValue{Int}   # count the number of collected samples

    # eigendecomposition
    λ::Vector{ComplexF64}
    P::Matrix{ComplexF64}
    P⁻¹::Matrix{ComplexF64}
    t::Base.RefValue{UnitRange{Int}}

    # Fourier transform
    Nft::Int
    iφ::Vector{ComplexF64}
    expiφ::Vector{ComplexF64}
    expiφμ::Vector{ComplexF64}
    Z̃ₘ::Vector{ComplexF64}   # Fourier weights

    # temporal data
    P̃n::Matrix{ComplexF64}
    tmpPn::Matrix{ComplexF64}
end

function PnSampler(
    system::System, qmc::QMC, Aidx::Vector{Int}; 
    Nft::Int = system.V+1, nsamples::Int = qmc.nsamples
)
    Ns = system.V
    L = length(Aidx)

    # observables
    Pn = zeros(ComplexF64, L+1, nsamples)

    # eigendecomposition
    λ = zeros(ComplexF64, Ns)
    P = zeros(ComplexF64, Ns, Ns)
    P⁻¹ = zeros(ComplexF64, Ns, Ns)
    t = Ref(1:Ns)

    # Fourier transform
    iφ = im * [2*π*m / Nft for m = 1 : Nft]
    expiφ = exp.(iφ)
    expiφμ = zeros(ComplexF64, Nft)
    Z̃ₘ = zeros(ComplexF64, Nft)

    # temporal data
    P̃n = zeros(ComplexF64, L+1, Nft)
    tmpPn = zeros(ComplexF64, L+1, L)

    return PnSampler(
        Aidx,
        Pn, Ref(1),
        λ, P, P⁻¹, t,
        Nft, iφ, expiφ, expiφμ, Z̃ₘ,
        P̃n, tmpPn
    )
end

function update!(
    sampler::PnSampler, walker::Walker{T, LDRLowRank{T,E}}, spin::Int
) where {T,E}
    λ, P, P⁻¹ = eigen(walker.F[spin], walker.ws)
    sampler.t[] = walker.F[spin].t[]
    t = sampler.t[]
    @views copyto!(sampler.λ[t], λ)
    @views copyto!(sampler.P[:, t], P)
    @views copyto!(sampler.P⁻¹[t, :], P⁻¹)

    return nothing
end

function update!(
    sampler::PnSampler, walker::Walker{T, LDR{T,E}}, spin::Int
) where {T,E}
    λ, P, P⁻¹ = eigen(walker.F[spin], walker.ws)
    copyto!(sampler.λ, λ)
    copyto!(sampler.P, P)
    copyto!(sampler.P⁻¹, P⁻¹)

    return nothing
end

"""
    compute_Fourier_weights(...)

    Compute the Fourier coefficients. Results are in the logrithmic form
    to avoid numerical overflows
"""
function compute_Fourier_weights(
    system::System, sampler::PnSampler, spin::Int
)
    N = system.N[spin]
    Ns = sampler.Nft
    expiφ = sampler.expiφ
    Z̃ₘ = sampler.Z̃ₘ

    expβμ = fermilevel(sampler.λ, N)
    βμN = N * log(expβμ)
    for i in 1:Ns
        sampler.expiφμ[i] = expiφ[i] / expβμ
    end

    λ = sampler.λ / expβμ
    for m in 1:Ns
        Z̃ₘ[m] = sum(log.(1 .+ expiφ[m]*λ)) + βμN - N*sampler.iφ[m]
    end

    return Z̃ₘ
end

function Pn_estimator(
    sampler::PnSampler, ws::LDRWorkspace{T,E}
) where {T,E}

    Ns = sampler.Nft
    P̃n = sampler.P̃n
    tmpPn = sampler.tmpPn

    Aidx = sampler.Aidx
    t = sampler.t[]
    λ = @view sampler.λ[t]
    P = @view sampler.P[:, t]
    P⁻¹ = @view sampler.P⁻¹[t, :]
    expiφμ = sampler.expiφμ

    for m in 1 : Ns
        nₖφₘ = @view ws.M[t, 1]
        @inbounds for i in eachindex(λ)
            nₖφₘ[i] = expiφμ[m]*λ[i] / (1 + expiφμ[m]*λ[i])
        end

        GₐP⁻¹ = @view ws.M′[t, Aidx]
        @views mul!(GₐP⁻¹, Diagonal(nₖφₘ), P⁻¹[:, Aidx])
        Gₐ = @view ws.M″[Aidx, Aidx]
        @views mul!(Gₐ, P[Aidx, :], GₐP⁻¹)

        ϵ = eigvals(Gₐ)
        # compute the eigenvalues of (GA⁻¹ - I)⁻¹
        ϵ = 1 ./ (1 ./ ϵ .- 1)
        sort!(ϵ, by=abs)

        # apply Poisson binomial iterator
        poissbino(ϵ, P=tmpPn)

        @views copyto!(P̃n[:, m], tmpPn[:, end])
    end

    return P̃n
end

function measure_Pn(
    system::System, walker::Walker, sampler::PnSampler, spin::Int
)

    s = sampler.s_counter[]
    Lₐ = length(sampler.Aidx)

    # update the eigendecomposition
    update!(sampler, walker, spin)
    # compute the Fourier weights
    compute_Fourier_weights(system, sampler, spin)
    # compute frequency-dependent probabilities
    Pn_estimator(sampler, walker.ws)

    # compute the Fourier coefficients
    Ns = sampler.Nft
    Z̃ₘ = sampler.Z̃ₘ
    logZ = walker.weight[spin] + log(walker.sign[spin])
    for m in eachindex(Z̃ₘ)
        Z̃ₘ[m] = exp(Z̃ₘ[m] - logZ)
    end
    # reverse Fourier transform
    for i = 1 : Lₐ+1
        sampler.Pn[i, s] = sum(sampler.P̃n[i, :] .* Z̃ₘ) / Ns
    end
    
    sampler.s_counter[] += 1

    return nothing
end
