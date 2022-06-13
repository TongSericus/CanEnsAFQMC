"""
    Measure Renyi-2 and accessible entanglement entropies
"""
function eigvals_squaredEtgHam(
    eigenGA1::Eigen{T, T, Matrix{T}, Vector{T}},
    eigenGA2::Eigen{T, T, Matrix{T}, Vector{T}}
) where {T<:FloatType}
    """
    Compute the eigenvalues of exp(-HA1) * exp(-HA2) in the form
    of GA1 * (I - GA1)^-1 * GA2 * (I - GA2)^-1
    """
    nkA1, P1 = eigenGA1
    nkA2, P2 = eigenGA2

    nkAc1 = regularized_complement.(nkA1)
    HA1 = UDT(nkA1 ./ nkAc1, P1, inv(P1))

    nkAc2 = regularized_complement.(nkA2)
    HA2 = UDT(nkA2 ./ nkAc2, P2, inv(P2))

    HA = QRCP_merge(HA1, HA2)

    return eigvals(HA)

end

function measure_renyi2_entropy(
    system::System, Aidx::Vector{Int64}, spin::Int64,
    walker1::WalkerProfile, walker2::WalkerProfile
)
    """
    Compute the regular and particle-number-resolved Renyi-2 entropies 
    for a pair of replica samples
    """
    N = system.N[spin]
    LA = length(Aidx)

    expβμ1 = fermilevel(walker1.expβϵ, N)
    expiφβμ1 = system.expiφ / expβμ1
    P1 = walker1.P[Aidx, :]
    invP1 = walker1.invP[:, Aidx]

    expβμ2 = fermilevel(walker2.expβϵ, N)
    expiφβμ2 = system.expiφ / expβμ2
    P2 = walker2.P[Aidx, :]
    invP2 = walker2.invP[:, Aidx]

    # initialize observables
    expS2 = 0
    nU = min(LA, N)
    nL = max(0, N - system.V + LA)
    expS2n = SizedVector{N + 1}(zeros(ComplexF64, N + 1))
    
    # measure through a 2D Fourier transform
    for m = 1 : system.V + 1
        for n = 1 : system.V + 1
            # regular Renyi-2 entropy calculation
            nk1 = expiφβμ1[m] * walker1.expβϵ ./ (1 .+ expiφβμ1[m] * walker1.expβϵ)
            GA1 = UDT(nk1, P1, invP1)
            nkc1 = 1 ./ (1 .+ expiφβμ1[m] * walker1.expβϵ)
            GAc1 = UDT(nkc1, P1, invP1)

            nk2 = expiφβμ2[n] * walker2.expβϵ ./ (1 .+ expiφβμ2[n] * walker2.expβϵ)
            GA2 = UDT(nk2, P2, invP2)
            nkc2 = 1 ./ (1 .+ expiφβμ2[n] * walker2.expβϵ)
            GAc2 = UDT(nkc2, P2, invP2)

            GA = QRCP_merge(GA1, GA2)
            GAc = QRCP_merge(GAc1, GAc2)
            γmn = det(QRCP_sum(GA, GAc))
            ηmn = prod(1 .+ expiφβμ1[m] * walker1.expβϵ) * prod(1 .+ expiφβμ2[n] * walker2.expβϵ)
            expS2_temp = conj(quick_rotation(system.expiφ[m], N)) * conj(quick_rotation(system.expiφ[n], N)) * ηmn * γmn
            expS2 += expS2_temp

            # particle-number-resolved Renyi-2 entropy calculation
            eigenGA1 = eigen(GA1)
            eigenGA2 = eigen(GA2)
            ν2mn = eigvals_squaredEtgHam(eigenGA1, eigenGA2)
            Pmn = poissbino(LA, ν2mn, false)[nL + 1 : nU + 1]
            expS2n[nL + 1 : nU + 1] += expS2_temp * Pmn
        end
    end

    r = expβμ1^N * expβμ2^N / walker1.weight / walker2.weight / (system.V + 1)^2    # rescale factor
    system.isReal && return real(expS2 * r), real(expS2n * r)
    return expS2 * r, expS2n * r
end
