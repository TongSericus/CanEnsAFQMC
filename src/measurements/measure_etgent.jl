"""
    Measure Renyi-2 and accessible entanglement entropies
"""
function eigvals_squaredEtgHam(
    GA1::UDT, GAc1::UDT, GA2::UDT, GAc2::UDT
)
    """
    Compute the eigenvalues of exp(-HA1) * exp(-HA2) in the form
    of GA1 * (I - GA1)^-1 * GA2 * (I - GA2)^-1
    """
    invGAc1 = UDT(inv(GAc1))
    HA1 = QRCP_merge(GA1, invGAc1)

    invGAc2 = UDT(inv(GAc2))
    HA2 = QRCP_merge(GA2, invGAc2)

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
    nU = min(LA, system.N[spin])
    nL = max(0, system.N[spin] - system.V + LA)
    Ln = nU - nL + 1
    expS2n = zeros(ComplexF64, Ln)
    
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
            expS2_temp = conj(system.expiφ[m])^N * conj(system.expiφ[n])^N * ηmn * γmn
            expS2 += expS2_temp

            # particle-number-resolved Renyi-2 entropy calculation
            ν2mn = eigvals_squaredEtgHam(GA1, GAc1, GA2, GAc2)
            Pmn = poissbino(LA, ν2mn, false)[nL + 1 : nU + 1]
            expS2n += expS2_temp * Pmn
        end
    end

    # rescale factor
    r = expβμ1^N * expβμ2^N / walker1.weight / walker2.weight / (system.V + 1)^2
    return expS2 * r , expS2n * r
    
end
