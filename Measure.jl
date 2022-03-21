"""
    Measure Observables
"""
function construct_walker_profile(system::System, walker::Walker)
    walker_eigen = (
            eigen(walker.D[1] * walker.T[1] * walker.Q[1], sortby = abs),
            eigen(walker.D[2] * walker.T[2] * walker.Q[2], sortby = abs)
        )
    P = (
            walker.Q[1] * walker_eigen[1].vectors,
            walker.Q[2] * walker_eigen[2].vectors
        )
    return (
            WalkerProfile(walker.weight[1], walker_eigen[1].values, P[1], inv(P[1])),
            WalkerProfile(walker.weight[2], walker_eigen[2].values, P[2], inv(P[2]))
        )

end

function measurement_mcmc(system::System, walker::Walker)
    """
    Measurements in MCMC
    """
    # construct the density matrix
    walker_eigen = (
            eigen(walker.D[1] * walker.T[1] * walker.Q[1], sortby = abs),
            eigen(walker.D[2] * walker.T[2] * walker.Q[2], sortby = abs)
        )
    ni = (
        recursion(system.V, system.N[1], walker_eigen[1].values, true),
        recursion(system.V, system.N[2], walker_eigen[2].values, true)
        )
    P = (
        walker.Q[1] * walker_eigen[1].vectors,
        walker.Q[2] * walker_eigen[2].vectors
        )
    G = (
        transpose(inv(P[1])) * Diagonal(ni[1]) * transpose(P[1]),
        transpose(inv(P[2])) * Diagonal(ni[2]) * transpose(P[2])
    )
end

###### Energy ######
function measure_energy(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    Measure the kinetic (one-body), the potential (two-body) energy and total energy
    """
    Ek, Ep = 0, 0

    for i in eachindex(view(system.kinetic_matrix, 1 : system.V, 1 : system.V))
        if system.kinetic_matrix[i] != 0
            Ek += -system.t * (G[1][i[1], i[2]] + G[2][i[1], i[2]])
        end
    end

    for i = 1 : system.V
        Ep += system.U * (G[1][i, i] * G[2][i, i])
    end

    return real(Ek), real(Ep), real(Ek + Ep)

end

###### Momentum Distribution ######
function DFT_matrix(k::Vector{Float64}, r_matrix::Matrix{Vector{Float64}})
    """
    Discrete Fourier transform matrix
    """
    DFT_matrix = similar(r_matrix, ComplexF64)
    fill!(DFT_matrix, 0)
    for (i, r) in enumerate(r_matrix)
        DFT_matrix[i] = exp(im * dot(k, r))
    end

    return DFT_matrix

end

function measure_momentum_dist(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}},
    kpath::Vector{Vector{Float64}}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    Measure the momentum distribution along a given symmetry path
    kpath -> a symmetry path in the reciprocal lattice. For instance,
        [(0,0), (π,π), (π,0), (0,0)]
    """
    nk = (
        zeros(ComplexF64, length(kpath)), 
        zeros(ComplexF64, length(kpath))
        )
    for (i, k) in enumerate(kpath)
        nk[1][i] = sum(DFT_matrix(k, system.r_matrix) .* G[1])
        nk[2][i] = sum(DFT_matrix(k, system.r_matrix) .* G[2])
    end

    return nk

end

###### Spin-spin Correlation ######
function add_vector(system::System,
    rx::Int64, ry::Int64, lx::Int64, ly::Int64)
    """
    Calculate (rx + lx, ry + ly) with PBC
    """
    return (rx + lx - 1) % system.Ns[1] + 1, (ry + ly - 1) % system.Ns[2] + 1
end

function spin_corr(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}}, 
    r1::Tuple{Int64,Int64}, r2::Tuple{Int64,Int64}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    calculate spin correlation between two points
    """
    # convert to matrix indices
    d1 = (r1[2] - 1) * system.Ns[1] + r1[1]
    d2 = (r2[2] - 1) * system.Ns[1] + r2[1]
    return (
        wick2_converter(d2, d2, d1, d1, G[1]) -
        G[1][d2, d2] * G[2][d1, d1] -
        G[2][d2, d2] * G[1][d1, d1] +
        wick2_converter(d2, d2, d1, d1, G[2])
        )
end

function measure_spincorr_func(
    system::System, G::Tuple{Matrix{T1}, Matrix{T2}}, 
    path::Array{Tuple{Int64,Int64},1}
    ) where {T1<:FloatType, T2<:FloatType}
    """
    Measure the spin-spin correlation function along a given path

    # Argument
    path -> a triangular path in the lattice. For instance,
        [(0,0), (1,0), (2,0), (2, 1), (2, 2), (1, 1)] traces out
        such a path
    """
    Css = @MVector zeros(ComplexF64, length(path))
    for r1x = 1 : system.Ns[1]
        for r1y = 1 : system.Ns[2]
            for (i, l) in enumerate(path)
                r2 = add_vector(system, r1x, r1y, l[1], l[2])
                # spin correlation between r1 and r2
                Css[i] += spin_corr(system, G, (r1x, r1y), r2) / system.V
            end
        end
    end

    return real(Css)

end

###### Entanglement Entropy ######
function Poissbino_recursion(
    Ns::Int64, ϵ::Array{T,1}, NORMALIZED::Bool
) where {T<:FloatType}
    #) where {T<:FloatType}
    """
    A regularized version of the recursive relation for the Poisson binomial
    distribution

    # Argument
    ϵ -> eigenvalues
    """
    ν1 = zeros(Complex{Float64}, Ns)
    ν2 = zeros(Complex{Float64}, Ns)
    # occupancy & hole distribution
    if NORMALIZED
        ν1 = ϵ .+ (abs.(ϵ) .< 1e-10) * 1e-10
        ν2 = 1 .- ϵ .+ (abs.(1 .- ϵ) .< 1e-10) * 1e-10
    else
        ν1 = ϵ ./ (1 .+ ϵ)
        ν2 = 1 ./ (1 .+ ϵ)
    end
    # culmulative probability distribution
    P = complex(zeros(Ns + 1, Ns))

    # Initialization
    P[1, 1] = ν2[1]
    P[2, 1] = ν1[1]
    # iteration over trials
    for i = 2 : Ns
        P[1, i] = ν2[i] * P[1, i - 1]
        # iteration over number of successes
        for j = 2 : i + 1
            P[j, i] = ν2[i] * P[j, i - 1] + ν1[i] * P[j - 1, i - 1]
        end
    end

    return P[:, Ns]

end

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

    nAkc1 = 1 .- nAk1
    nAkc2 = 1 .- nAk2

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
    system::System, etg::EtgMeasure, spin::Int64,
    walker1::WalkerProfile, walker2::WalkerProfile
    ) where {T1<:FloatType, T2<:FloatType}
    """
    Compute the regular and particle-number-resolved Renyi-2 entropies 
    for a pair of replica samples
    """
    N = system.N[spin]
    expβμ1 = pinpoint_Fermilevel(walker1.expβϵ, N)
    P1 = walker1.P[etg.Aidx, :]
    invP1 = walker1.invP[:, etg.Aidx]
    expβμ2 = pinpoint_Fermilevel(walker2.expβϵ, N)
    P2 = walker2.P[etg.Aidx, :]
    invP2 = walker2.invP[:, etg.Aidx]

    # initialize observables
    expS2 = 0
    expS2n = zeros(ComplexF64, length(etg.k))
    
    # measure through a 2D Fourier transform
    for m = 1 : system.V + 1
        for n = 1 : system.V + 1
            # regular Renyi-2 entropy calculation
            nk1 = (etg.expiφ[m] / expβμ1) * walker1.expβϵ ./ (1 .+ (etg.expiφ[m] / expβμ1) * walker1.expβϵ)
            GA1 = P1 * Diagonal(nk1) * invP1
            nkc1 = 1 ./ (1 .+ (etg.expiφ[m] / expβμ1) * walker1.expβϵ)
            GAc1 = P1 * Diagonal(nkc1) * invP1  # namely, I - GA1

            nk2 = (etg.expiφ[n] / expβμ2) * walker2.expβϵ ./ (1 .+ (etg.expiφ[n] / expβμ2) * walker2.expβϵ)
            GA2 = P2 * Diagonal(nk2) * invP2
            nkc2 = 1 ./ (1 .+ (etg.expiφ[n] / expβμ2) * walker2.expβϵ)
            GAc2 = P2 * Diagonal(nkc2) * invP2  # namely, I - GA2

            γmn = det(GA1 * GA2 + GAc1 * GAc2)
            ηmn = prod(1 .+ etg.expiφ[m] / expβμ1 * walker1.expβϵ) *
                prod(1 .+ etg.expiφ[n] / expβμ2 * walker2.expβϵ)
            expS2_temp = conj(etg.expiφ[m])^N * conj(etg.expiφ[n])^N * expβμ1^N * expβμ2^N *
                ηmn * γmn
            expS2 += expS2_temp

            # particle-number-resolved Renyi-2 entropy calculation
            ν2mn = eigvals_squaredEtgHam(nk1, P1, invP1, nk2, P2, invP2)
            Pmn = Poissbino_recursion(etg.LA, ν2mn, false)[etg.k .+ 1]
            expS2n += expS2_temp * Pmn
        end
    end

    return expS2 / (system.V + 1)^2 / walker1.weight / walker2.weight, expS2n / (system.V + 1)^2 / walker1.weight / walker2.weight
    
end
