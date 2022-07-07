"""
    Measure Renyi-2 and accessible entanglement entropies
"""
function eigvals_squaredEtgHam(
    eigenGA1::Eigen{T1, T1, Matrix{T1}, Vector{T1}},
    eigenGA2::Eigen{T2, T2, Matrix{T2}, Vector{T2}}
) where {T1<:FloatType, T2<:FloatType}
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

    HA = QR_merge(HA1, HA2)

    return eigvals(HA)

end

function Fourier_weights(
    Ns::Int64, N::Int64, iφ::Vector{ComplexF64}, expβϵ::Vector{T}, Z::E
) where {T<:FloatType, E<:FloatType}
    """
    Compute the Fourier coefficients in the logrithmic form
    to avoid numerical overflows
    """
    sgnZ = sign(Z)
    logZ = log(abs(Z))

    iφN = iφ * N
    expβμ = fermilevel(expβϵ, N)
    βμN = N * log(expβμ)
    expβϵμ = expβϵ / expβμ
    t = sum(abs.(expβϵμ) .< 1e-8) + 1 : length(expβϵμ)
    λ = [exp(iφ[m]) * expβϵμ for m = 1 : Ns]

    @views f = [sum(log.(1 .+ λ[m][t])) + βμN - N * iφ[m] for m = 1 : Ns]
    return exp.(f .- logZ) * sgnZ, λ
end

function measure_renyi2_entropy(
    system::System, Aidx::Vector{Int64}, spin::Int64,
    walker1::WalkerProfile{W, E1, G1}, walker2::WalkerProfile{W, E2, G2}
) where {W, E1, G1, E2, G2}
    """
    Compute the regular and particle-number-resolved Renyi-2 entropies 
    for a pair of replica samples
    """
    N = system.N[spin]
    iφ = system.iφ
    LA = length(Aidx)

    f1, λ1 = Fourier_weights(system.V, N, iφ, walker1.expβϵ, walker1.weight)
    P1 = walker1.P[Aidx, :]
    invP1 = walker1.invP[:, Aidx]

    f2, λ2 = Fourier_weights(system.V, N, iφ, walker2.expβϵ, walker2.weight)
    P2 = walker2.P[Aidx, :]
    invP2 = walker2.invP[:, Aidx]

    # initialize observables
    expS2 = 0
    nU = min(LA, N)
    nL = max(0, N - system.V + LA)
    expS2n = SizedVector{N + 1}(zeros(ComplexF64, N + 1))
    
    # measure through a 2D Fourier transform
    for m = 1 : system.V

        nk1 = λ1[m] ./ (1 .+ λ1[m])
        GA1 = UDT(nk1, P1, invP1)
        nkc1 = 1 ./ (1 .+ λ1[m])
        GAc1 = UDT(nkc1, P1, invP1)
        eigenGA1 = eigen(GA1)

        for n = 1 : system.V

            nk2 = λ2[n] ./ (1 .+ λ2[n])
            GA2 = UDT(nk2, P2, invP2)
            nkc2 = 1 ./ (1 .+ λ2[n])
            GAc2 = UDT(nkc2, P2, invP2)

            GA = QR_merge(GA1, GA2)
            GAc = QR_merge(GAc1, GAc2)
            γmn = det(QR_sum(GA, GAc))
            expS2_temp = f1[m] * f2[n] * γmn
            expS2 += expS2_temp

            # particle-number-resolved Renyi-2 entropy calculation
            eigenGA2 = eigen(GA2)
            ν2mn = eigvals_squaredEtgHam(eigenGA1, eigenGA2)
            Pmn = poissbino(LA, ν2mn, false)[nL + 1 : nU + 1]
            expS2n[nL + 1 : nU + 1] += expS2_temp * Pmn
        end
    end

    r = system.V^2    # rescale factor
    system.isReal && return real(expS2 / r), real(expS2n / r)
    return expS2 / r, expS2n / r
end
