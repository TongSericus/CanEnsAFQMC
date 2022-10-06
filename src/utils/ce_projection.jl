"""
    Canonical Ensemble Projections

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number
    expβϵ -> exponentiated spectrum, i.e., exp(-βϵ)
    expiφ -> discrete Fourier frequencies
"""

function pf_projection(
    Ns::Int64, N::Int64, expβϵ::Vector{T}; 
    returnFull::Bool = false,
    iφ = im * [2 * π * m / Ns for m = 1 : Ns],
    expiφ = exp.(iφ)
) where {T<:FloatType}
    expβμ = fermilevel(expβϵ, N)
    expiφβμ = expiφ / expβμ

    η = [prod(1 .+ expiφβμ[m] * expβϵ) for m = 1 : Ns]
    Z = sum((conj(expiφ) * expβμ).^N .* η) / Ns

    returnFull || return Z
    return expβμ, η, Z
end

function occ_projection(
    Ns::Int64, N::Int64, expβϵ::Vector{T};
    iφ = im * [2 * π * m / Ns for m = 1 : Ns],
    expiφ = exp.(iφ)
) where {T<:FloatType}
    expβμ, η, Z = pf_projection(Ns, N, expβϵ, returnFull=true, iφ=iφ, expiφ=expiφ)
    ñ = [expiφ[m] / expβμ * expβϵ[i] / (1 + expiφ[m] / expβμ * expβϵ[i]) for m = 1 : Ns, i = 1 : Ns]
    n = [sum((conj(expiφ) * expβμ).^N .* ñ[:, i] .* η) / Ns / Z for i = 1 : Ns]
    return n
end


function pf_projection_stable(
    Ns::Int64, N::Int64, expβϵ::Vector{T}; 
    returnFull::Bool = false,
    iφ = im * [2 * π * m / Ns for m = 1 : Ns],
    expiφ = exp.(iφ)
) where {T<:FloatType}
    expβμ = fermilevel(expβϵ, N)
    expiφβμ = expiφ / expβμ

    η = [sum(log.(1 .+ expiφβμ[m] * expβϵ)) for m = 1 : Ns]

    logZm = N*(-iφ .+ log(expβμ)) .+ η
    logZm_max = maximum(real(logZm))

    Z = sum(exp.(logZm .- logZm_max))
    logZ = log(Z) + logZm_max - log(Ns)

    returnFull || return logZ
    return expβμ, η, logZ
end

function occ_projection_stable(
    Ns::Int64, N::Int64, expβϵ::Vector{T};
    iφ = im * [2 * π * m / Ns for m = 1 : Ns],
    expiφ = exp.(iφ)
) where {T<:FloatType}
    expβμ, η, Z = pf_projection_stable(Ns, N, expβϵ, returnFull=true, iφ=iφ, expiφ=expiφ)
    ñ = [expiφ[m] / expβμ * expβϵ[i] / (1 + expiφ[m] / expβμ * expβϵ[i]) for m = 1 : Ns, i = 1 : Ns]
    n = [sum((conj(expiφ) * expβμ).^N .* ñ[:, i] .* η) / Ns / Z for i = 1 : Ns]
    return n
end
