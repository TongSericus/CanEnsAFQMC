"""
    Measure Renyi-2 and accessible entanglement entropies
"""
function eigvals_etgHam(
    nk::Vector{T1}, P::Matrix{T2}, invP::Matrix{T2}
) where {T1<:FloatType, T2<:FloatType}
    """
    Stable eigen decomposion of GA using SVD
    """
    UDV_GA = svd(Diagonal(nk) * invP)
    U = P * UDV_GA.U
    eigen_GA = eigen(Diagonal(UDV_GA.S) * (UDV_GA.Vt * U), sortby = abs)
    nAk = eigen_GA.values
    PA = U * eigen_GA.vectors
    invPA = inv(PA)

    return nAk, PA, invPA

end

function eigvals_squaredEtgHam(
    nk1::Vector{T1}, P1::Matrix{T2}, invP1::Matrix{T2},
    nk2::Vector{T3}, P2::Matrix{T4}, invP2::Matrix{T4}
) where {T1<:FloatType, T2<:FloatType, T3<:FloatType, T4<:FloatType}
    """
    Compute the eigenvalues of exp(-HA1) * exp(-HA2) in the form
    of GA1 * (I - GA1)^-1 * GA2 * (I - GA2)^-1
    """
    # Compute the eigen decomposion for GA1 and GA2 first
    nAk1, PA1, invPA1 = eigvals_etgHam(nk1, P1, invP1)
    nAk2, PA2, invPA2 = eigvals_etgHam(nk2, P2, invP2)

    nAkc1 = regularized_complement.(nAk1)
    nAkc2 = regularized_complement.(nAk2)

    # Compute SVD decomposions of GAi * (I - GAi)^-1
    UDV_HA1 = svd(Diagonal(nAk1 ./ nAkc1) * invPA1)
    U1 = PA1 * UDV_HA1.U
    UDV_HA2 = svd(Diagonal(nAk2 ./ nAkc2) * invPA2)
    U2 = PA2 * UDV_HA2.U

    # Merge SVD results, i.e, UDV = U1*D1*V1 * U2*D2*V2
    UDV = svd(Diagonal(UDV_HA1.S) * (UDV_HA1.Vt * U2) * Diagonal(UDV_HA2.S))
    U = U1 * UDV.U
    Vt = UDV.Vt * UDV_HA2.Vt

    return eigvals(Diagonal(UDV.S) * Vt * U, sortby = abs)

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
    expβμ1 = fermilevel(walker1.expβϵ, N)
    P1 = walker1.P[Aidx, :]
    invP1 = walker1.invP[:, Aidx]
    expβμ2 = fermilevel(walker2.expβϵ, N)
    P2 = walker2.P[Aidx, :]
    invP2 = walker2.invP[:, Aidx]

    # initialize observables
    expS2 = 0
    LA = length(Aidx)
    nU = min(LA, system.N[spin])
    nL = max(0, system.N[spin] - system.V + LA)
    Ln = nU - nL + 1
    expS2n = zeros(ComplexF64, Ln)
    
    # measure through a 2D Fourier transform
    for m = 1 : system.V + 1
        for n = 1 : system.V + 1
            # regular Renyi-2 entropy calculation
            nk1 = (system.expiφ[m] / expβμ1) * walker1.expβϵ ./ (1 .+ (system.expiφ[m] / expβμ1) * walker1.expβϵ)
            GA1 = P1 * Diagonal(nk1) * invP1
            nkc1 = 1 ./ (1 .+ (system.expiφ[m] / expβμ1) * walker1.expβϵ)
            GAc1 = P1 * Diagonal(nkc1) * invP1  # namely, I - GA1

            nk2 = (system.expiφ[n] / expβμ2) * walker2.expβϵ ./ (1 .+ (system.expiφ[n] / expβμ2) * walker2.expβϵ)
            GA2 = P2 * Diagonal(nk2) * invP2
            nkc2 = 1 ./ (1 .+ (system.expiφ[n] / expβμ2) * walker2.expβϵ)
            GAc2 = P2 * Diagonal(nkc2) * invP2  # namely, I - GA2

            γmn = det(GA1 * GA2 + GAc1 * GAc2)
            ηmn = prod(1 .+ system.expiφ[m] / expβμ1 * walker1.expβϵ) *
                prod(1 .+ system.expiφ[n] / expβμ2 * walker2.expβϵ)
            expS2_temp = conj(system.expiφ[m])^N * conj(system.expiφ[n])^N * expβμ1^N * expβμ2^N *
                ηmn * γmn
            expS2 += expS2_temp

            # particle-number-resolved Renyi-2 entropy calculation
            ν2mn = eigvals_squaredEtgHam(nk1, P1, invP1, nk2, P2, invP2)
            Pmn = poissbino(LA, ν2mn, false)[nL + 1 : nU + 1]
            expS2n += expS2_temp * Pmn
        end
    end

    return expS2 / (system.V + 1)^2 / walker1.weight / walker2.weight, expS2n / (system.V + 1)^2 / walker1.weight / walker2.weight
    
end
