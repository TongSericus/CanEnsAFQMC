"""
    Canonical Ensemble Recursions

    # General Arguments
    Ns -> number of energy levels
    N -> desired particle number (number of recursions)
    expβϵ -> exponentiated spectrum, i.e., exp(-βϵ)
"""

function recursion(
    Ns::Int64, N::Int64, expβϵ::Array{T,1}, OCCUPANCY::Bool
) where {T<:FloatType}
    """
    Main function to do the particle number recursion

    # Arguments
    OCCUPANCY -> if true, return the level occupancies instead of partition functions (PFs)

    # Returns
    Z -> statistical weights of the configuration
    ni -> ⟨n_{i}⟩, first-order correlation matrix (mean occupancy)
    """

    ### recursion setup ###
    # exponentiated chemical potential
    expβμ = (abs(expβϵ[Ns - N + 1]) +
            abs(expβϵ[Ns - N])) / 2
    # rescaled spectrum
    expβϵμ = expβϵ / expβμ

    # recursions for PFs
    Z̃ = APF_recursion(Ns, expβϵμ)
    # use the Z_1 instead of Z_0 as Z̃_1 is more stable numerically
    Z₁ = real(sum(expβϵ))
    Z = Z̃[N + 1] / Z̃[2] * (expβμ) ^ (N - 1) * Z₁

    # skip the occupancy number recursion
    OCCUPANCY || return Z

    # num of energy levels below the Fermi level
    # use this formula to ensure complex conjugate pairs are in the same section
    # so that the particle number recursion is stable
    N_below = sum(abs.(expβϵμ) .> 1)

    # separate recursions for occupancies of energy levels above/below the Fermi level
    n₁ = expβϵ / Z₁
    ni_above = occupancy_recursion(N, expβϵμ[1 : Ns - N_below], Z̃, n₁)
    ni_below = occupancy_recursion_reverse(Ns, N, expβϵμ[Ns - N_below + 1 : Ns], Z̃)
    # then concatenate
    ni = vcat(ni_above, ni_below)

    ni

end

function APF_recursion(
    Ns::Int64, expβϵμ::Array{T,1}
) where {T<:FloatType}
    """
    Auxiliary partition function recursion with the rescaled spectrum
        that solves the numerical instabilities while running fermion recursions
    See doi.org/10.1103/PhysRevResearch.2.043206
    """
    # occupancy distribution
    ν1 = expβϵμ ./ (1 .+ expβϵμ)
    # 1 - ν1 (hole distribution)
    ν2 = 1 ./ (1 .+ expβϵμ)
    # APF
    Z̃ = complex(zeros(Ns + 1, Ns))

    # Initialization
    Z̃[1, 1] = ν2[1]
    Z̃[2, 1] = ν1[1]
    # iteration over energy levels
    for i = 2 : Ns
        Z̃[1, i] = ν2[i] * Z̃[1, i - 1]
        # iteration over particles
        for j = 2 : i + 1
            Z̃[j, i] = ν2[i] * Z̃[j, i - 1] + ν1[i] * Z̃[j - 1, i - 1]
        end
    end

    # PF can be proven to be strictly real
    return real(Z̃[:, Ns])

end

function occupancy_recursion(
    N::Int64, expβϵμ::Array{T1,1}, Z̃::Array{T2,1}, n₁::Array{T3,1}
) where {T1<:FloatType, T2<:FloatType, T3<:FloatType}
    """
    Level occupancy recursion using the rescaled spectrum and APF
    for levels above the Fermi level
    """
    # Initialization
    Ñs = length(expβϵμ)
    ni = complex(zeros(N + 1, Ñs))
    # use n₁ as a correction
    ni[2, :] = n₁[1 : Ñs]

    for i = 2 : N
        ni[i + 1, :] = real(Z̃[i] / Z̃[i + 1]) * expβϵμ .* (1 .- ni[i, :])
        # Truncate the values that are smaller than 10^-10
        ni[i + 1, :] = ni[i + 1, :] .* (abs.(real(ni[i + 1, :])) .> 1e-10)
    end

    return ni[N + 1, :]

end

function occupancy_recursion_reverse(
    Ns::Int64, N::Int64, expβϵμ::Array{T1,1}, Z̃::Array{T2,1}
) where {T1<:FloatType, T2<:FloatType}
    """
    Reversed level occupancy recursion using rescaled spectrum and APF
    for levels below the Fermi level
    """
    # Initialization
    Ñs = length(expβϵμ)
    N_rev = Ns - N                  # num. of reverse recursions
    ni = complex(ones(N_rev, Ñs))

    for i = 1 : N_rev - 1
        ni[i + 1, :] = real(Z̃[Ns - i + 1] / Z̃[Ns - i]) * ni[i, :] ./ expβϵμ
        # Truncate the values that are smaller than 10^-10
        ni[i + 1, :] = ni[i + 1, :] .* (abs.(real(ni[i + 1, :])) .> 1e-10)
        ni[i + 1, :] = 1 .- ni[i + 1, :]
    end

    return ni[N_rev, :]

end


function second_order_corr(
    Ns::Int64, expβϵ::Array{T,1}, ni::Array{ComplexF64,1}
    ) where {T<:FloatType}
    """
    Generate second-order correlation matrix, i.e., ⟨n_{i} n_{j}⟩ using the formula:
        ⟨n_{i} n_{j}⟩ = (n_{i}/expβϵ_{i} - n_{j}/expβϵ_{j}) / (1/expβϵ_{i} - 1/expβϵ_{j}).
    Practically, degenerate levels (expβϵ_{i} = expβϵ_{j}) should be super rare so we dont consider it here.
    """
    nij = complex(zeros(Ns, Ns))
    expβϵ_inv = 1 ./ expβϵ
    for i = 2 : Ns
        niexpβϵ = ni[i] * expβϵ_inv[i]
        for j = 1 : i - 1
            njexpβϵ = ni[j] * expβϵ_inv[j]
            nij[i, j] = (niexpβϵ - njexpβϵ) / (expβϵ_inv[i] - expβϵ_inv[j])
            nij[j, i] = nij[i, j]
        end
    end

    return nij

end
