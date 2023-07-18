"""
    Canonical Ensemble Recursions

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number (number of recursions)
    λ -> exponentiated spectrum, i.e., exp(-βϵ)
"""

"""
    compute_pf_recursion(λ::AbstractVector{T}, N::Int)

    Given a exponentiated spectrum λ = exp(-βϵ), compute the N-particle
    partition function recursively
"""
function compute_pf_recursion(
    λ::AbstractVector{T}, N::Int;
    isReal::Bool = false,
    Ns = length(λ),
    P::AbstractMatrix{Tp} = zeros(eltype(λ), N + 1, Ns)
) where {T, Tp}
    isReal || (λ = complex(λ))

    N == 0 && return convert(T, 0.0), 1.0
    if N == 1 
        logZ = log(sum(λ))
        abslogZ = real(logZ)
        sgnlogZ = exp(imag(logZ)im)

        return abslogZ, sgnlogZ
    elseif N == Ns
        logZ = sum(log.(λ))
        abslogZ = real(logZ)
        sgnlogZ = exp(imag(logZ)im)

        return abslogZ, sgnlogZ
    elseif N == Ns - 1
        logZ = sum(log.(λ)) - log(sum(λ))
        abslogZ = real(logZ)
        sgnlogZ = exp(imag(logZ)im)

        return abslogZ, sgnlogZ
    end
    
    # rescale spectrum
    expβμ = fermilevel(λ, N)
    expβϵμ = λ / expβμ
    
    poissbino(expβϵμ, N, P=P)

    logP0 = -sum(log.(1 .+ expβϵμ))
    logZ = log(P[N+1, Ns]) - logP0 + N*log(expβμ)

    abslogZ = real(logZ)
    sgnlogZ = exp(imag(logZ)im)

    return abslogZ, sgnlogZ
end

function compute_pf_recursion(
    λ::AbstractVector{T}, N::Int, ϵ::Float64;
    isReal::Bool = false,
    Ns = length(λ),
    P::AbstractMatrix{Tp} = zeros(eltype(λ), N + 1, Ns)
) where {T, Tp}
    """
    Recursive calculation of the partition function with low-temperature approximation

    ϵ -> tolerance for the partition function. Empirically, it can be directly used as
        the truncation threshold for the level occupancy, p.
    """
    isReal || (λ = complex(λ))

    expβμ = fermilevel(λ, N)
    expβϵμ = λ / expβμ

    p = expβϵμ ./ (1 .+ expβϵμ)
    Imp = 1 ./ (1 .+ expβϵμ)

    Nsu = 1
    while real(p[Nsu]) < ϵ
        Nsu += 1
    end
    Nsl = Ns
    while real(1 - p[Nsl]) < ϵ
        Nsl -= 1
    end

    λocc = @view λ[Nsl + 1 : Ns]
    λf = @view expβϵμ[Nsu : Nsl]
    Nsf = length(λf)
    Nf = N - (Ns - Nsl)
    poissbino(λf, Nf, P=P, ν1 = (@view p[Nsu : Nsl]), ν2= (@view Imp[Nsu : Nsl]))

    logP0 = -sum(log.(1 .+ λf))
    logZ = log(P[Nf+1, Nsf]) - logP0 + Nf*log(expβμ) + sum(log.(λocc))

    abslogZ = real(logZ)
    sgnlogZ = exp(imag(logZ)im)

    return abslogZ, sgnlogZ
end

function compute_occ_recursion(
    λ::AbstractVector{T}, N::Int64;
    isReal::Bool = false,
    Ns::Int = length(λ),
    P::AbstractMatrix{Tp} = zeros(eltype(λ), Ns + 1, Ns)
) where {T<:Number, Tp<:Number}
    """
    Recursive calculation of the occupation number
    """
    isReal || (λ = complex(λ))

    N == 0 && return zeros(T, Ns)
    N == Ns && return ones(T, Ns)
    N == 1 && return λ / sum(λ)

    expβμ = fermilevel(λ, N)
    expβϵμ = λ / expβμ
    poissbino(expβϵμ, P=P)

    # num of energy levels below the Fermi level
    # use this formula to ensure complex conjugate pairs are in the same section
    N_below = sum(abs.(expβϵμ) .> 1)

    # separate recursions for occupancies of energy levels above/below the Fermi level
    @views n_above = compute_occ_recursion_rescaled(Ns, N, expβϵμ[1 : Ns - N_below], P[:, Ns])
    @views n_below = compute_occ_recursion_rescaled(Ns, N, expβϵμ[Ns - N_below + 1 : Ns], P[:, Ns], isReverse=true)
    # then concatenate
    n = vcat(n_above, n_below)

    return n
end

function compute_occ_recursion_rescaled(
    Ns::Int64, N::Int64,
    expβϵμ::AbstractArray{Te}, P::AbstractArray{Tp};
    isReverse::Bool = false
) where {Te, Tp}
    """
    Level occupancy recursion using the rescaled spectrum
    """
    Ñs = length(expβϵμ)
    if !isReverse
        n = zeros(ComplexF64, N + 1, Ñs)
        @inbounds for i = 2 : N
            n[i + 1, :] = (P[i] / P[i + 1]) * expβϵμ .* (1 .- n[i, :])
            # Truncate the values that are smaller than 10^-10
            n[i + 1, :] = n[i + 1, :] .* (abs.(real(n[i + 1, :])) .> 1e-10)
        end
        return n[N + 1, :]
    else
        # num. of reverse recursions
        N_rev = Ns - N
        n = ones(ComplexF64, N_rev, Ñs)
        @inbounds for i = 1 : N_rev - 1
            n[i + 1, :] = (P[Ns - i + 1] / P[Ns - i]) * n[i, :] ./ expβϵμ
            # Truncate the values that are smaller than 10^-10
            n[i + 1, :] = n[i + 1, :] .* (abs.(real(n[i + 1, :])) .> 1e-10)
            n[i + 1, :] = 1 .- n[i + 1, :]
        end
        return n[N_rev, :]
    end
end

function second_order_corr(
    λ::Array{T,1}, ni::Array{Tn,1};
    Ns = length(λ), ninj = zeros(Tn, Ns, Ns)
) where {T<:Number, Tn<:Number}
    """
    Generate second-order correlation matrix, i.e., ⟨n_{i} n_{j}⟩ using the formula:
    ⟨n_{i} n_{j}⟩ = (n_{i}/expβϵ_{i} - n_{j}/expβϵ_{j}) / (1/expβϵ_{i} - 1/expβϵ_{j}).

    Practically, degenerate levels (expβϵ_{i} = expβϵ_{j}) should be super rare so we dont consider it here.
    """
    @inbounds for i in 1 : Ns
        ninj[i, i] = ni[i]
    end

    λ⁻¹ = 1 ./ λ
    @inbounds for i in 2 : Ns
        niλ⁻¹ = ni[i] * λ⁻¹[i]
        for j in 1 : i - 1
            njλ⁻¹ = ni[j] * λ⁻¹[j]
            ninj[j, i] = (njλ⁻¹ - niλ⁻¹) / (λ⁻¹[j] - λ⁻¹[i])
            ninj[i, j] = ninj[j, i]
        end
    end

    return ninj
end
