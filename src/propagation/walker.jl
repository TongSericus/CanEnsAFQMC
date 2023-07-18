"""
    QMC Walker Definitions
"""
### Random Walker in the Canonical Ensemble ###
struct Walker{T<:Number, Fact<:Factorization{T}, E, C}
    # weights of the walker in the logarithmic form
    weight::Vector{Float64}
    # signs/phases of the weights
    sign::Vector{ComplexF64}
    # temporal weights and signs
    weight′::Vector{Float64}
    sign′::Vector{ComplexF64}

    auxfield::Matrix{Int64}
    F::Vector{Fact}
    ws::LDRWorkspace{T, E}

    ### Temporal data to avoid memory allocations ###
    # All partial factorizations
    FC::Cluster{Fact}
    # Factorization of two unit matrices for spin-up and spin-down
    Fτ::Vector{Fact}
    # Preallocated multiplication of FC[i] and Fτ[i]
    FM::Vector{Fact}
    # Temporal array of matrices with the ith element B̃_i being
    # B̃_i = B_{(cidx-1)k + i}, where cidx is the index of the current imaginary time slice to be updated
    # Note that the spin-up and spin-down matrices are strored as the first and the second half of the array, respectively
    Bl::Cluster{C}
    # Temporal array of matrices with the ith element B̃_i being
    # B̃_i = B_{(i-1)k + 1} ⋯ B_{ik}, where k is the stablization interval defined as qmc.stab_interval
    Bc::Cluster{C}
    # Temporal matrices to store trial imaginary-time propagators
    Bτ::Cluster{C}
    # Temporal matrix for recursive calculation
    P::Matrix{ComplexF64}

    ### Data for debugging ###
    tmp_r::Vector{T}
end

function Walker(
    system::Hubbard, qmc::QMC;
    # Initialize a walker with a random configuration
    auxfield = 2 * (rand(system.V, system.L) .< 0.5) .- 1,
    T::DataType = eltype(system.auxfield)
)
    Ns = system.V
    N = system.N

    k = qmc.stab_interval
    K = qmc.K
    ϵ = qmc.lrThld

    weight = zeros(Float64, 2)
    weight′ = zeros(Float64, 2)
    sgn = zeros(ComplexF64, 2)
    sgn′ = zeros(ComplexF64, 2)

    Bl = Cluster(Ns, 2 * k, T = T)
    Bτ = Cluster(Ns, 2, T = T)
    ws = ldr_workspace(Bl.B[1])
    Fτ = ldrs(Bl.B[1], 2)
    FM = ldrs(Bl.B[1], 2)

    F, Bc, FC = build_propagator(auxfield, system, qmc, ws)
    if qmc.isLowrank
        F = [LDRLowRank(F[1], N[1], ϵ), LDRLowRank(F[2], N[2], ϵ)]
        Fτ = [LDRLowRank(Fτ[1], N[1], ϵ), LDRLowRank(Fτ[2], N[2], ϵ)]
        FM = [LDRLowRank(FM[1], N[1], ϵ), LDRLowRank(FM[2], N[2], ϵ)]

        FC_lowrank = Cluster(B = LDRLowRank{T, eltype(ws.v)}[])
        LDRLowRank(FC.B[1:K], N[1], ϵ, F_lowrank=FC_lowrank)
        LDRLowRank(FC.B[K+1:end], N[2], ϵ, F_lowrank=FC_lowrank)
        FC = FC_lowrank

        weight[1], sgn[1] = compute_pf(F[1], ws)
        weight[2], sgn[2] = compute_pf(F[2], ws)
    else
        weight[1], sgn[1] = compute_pf(F[1], N[1], ws)
        weight[2], sgn[2] = compute_pf(F[2], N[2], ws)
    end

    P = zeros(ComplexF64, Ns+1, Ns)
    tmp_r = Vector{T}()

    return Walker{T, eltype(F), eltype(ws.v), eltype(Bl.B)}(
                weight, sgn, weight′, sgn′, 
                auxfield, F, ws, 
                FC, Fτ, FM, Bl, Bc, Bτ, 
                P, tmp_r
            )
end
