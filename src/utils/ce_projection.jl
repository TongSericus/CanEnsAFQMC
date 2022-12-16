"""
    Canonical Ensemble Projections

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number
    λ -> exponentiated spectrum, i.e., λ := exp(-βϵ)
    expiφ -> discrete Fourier frequencies
"""
function pf_projection(
    λ::Vector{T}, N::Int; 
    returnFull::Bool = false,
    # number of quadrature points
    Nft = length(λ),
    iφ = im * [2 * π * m / Nft for m = 1 : Nft],
    expiφ = exp.(iφ)
) where T
    expβμ = fermilevel(λ, N)
    expiφmβμ = expiφ / expβμ

    η = [sum(log.(1 .+ expiφmβμ[m] * λ)) for m = 1 : Nft]

    logZm = N*(-iφ .+ log(expβμ)) .+ η

    # rescale all components
    logZm_max = maximum(real(logZm))
    Z = sum(exp.(logZm .- logZm_max))

    logZ = log(Z) + logZm_max - log(Nft)

    returnFull || return logZ
    return expβμ, expiφmβμ, η, logZ
end

function occ_projection(
    λ::Vector{T}, N::Int;
    Ns = length(λ),
    Nft = length(λ),
    iφ = im * [2 * π * m / Nft for m = 1 : Nft],
    expiφ = exp.(iφ),
    n = zeros(ComplexF64, Ns)
) where T
    expβμ, expiφmβμ, η, logZ = pf_projection(λ, N, returnFull=true, Nft=Nft, iφ=iφ, expiφ=expiφ)

    @inbounds for i in 1 : Ns
        ñk = expiφmβμ * λ[i] ./ (1 .+ expiφmβμ * λ[i])
        logñ = -iφ * N .+ log.(ñk) .+ η
        logn = logñ .+ log(expβμ) * N .- logZ
        n[i] = sum(exp.(logn)) / Nft
    end

    return n
end
