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
    expiφ = exp.(im * [2 * π * m / (prod(Ns) + 1) for m = 1 : prod(Ns) + 1])
) where {T<:FloatType}
    expβμ = fermilevel(expβϵ, N)
    η = [prod(1 .+ expiφ[m] / expβμ * expβϵ) for m = 1 : Ns + 1]
    Z = sum(quick_rotation(expiφ, N, true) .* η) / (Ns + 1) * expβμ ^ N

    returnFull || return Z
    return expβμ, η, Z
end

function occ_projection(
    Ns::Int64, N::Int64, expβϵ::Vector{T};
    expiφ = exp.(im * [2 * π * m / (prod(Ns) + 1) for m = 1 : prod(Ns) + 1])
) where {T<:FloatType}
    expβμ, η, Z = pf_projection(Ns, N, expβϵ, returnFull=true, expiφ=expiφ)
    ñ = [expiφ[m] / expβμ * expβϵ[i] / (1 + expiφ[m] / expβμ * expβϵ[i]) for m = 1 : Ns + 1, i = 1 : Ns]
    n = [sum(quick_rotation(expiφ, N, true) .* ñ[:, i] .* η) * expβμ ^ N / (Ns + 1) / Z for i = 1 : Ns]
    return n
end

function doubleocc_projection()
end