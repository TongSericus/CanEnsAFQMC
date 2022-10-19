"""
    Metropolis Sampling (MCMC) (for different factorizations)
"""
### UDR(deleted) ###

### UDT ###
function calc_pf(
    F::UDT, N::Int64;
    Ns = length(F.D),
    PMat = zeros(ComplexF64, Ns + 1, Ns)
)
    λ = eigvals(F)
    return pf_recursion(λ, N, P=PMat)
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, P::Matrix{Tp}, F1::UDT{T}, F2::UDT{T}
) where {T<:Number, Tp<:Number}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    weight_new = Vector{Tp}()
    Btmp = singlestep_matrix(σ, system)

    Utmp = Btmp[1] * F1.U
    Ftmp = UDT(Utmp, F1.D, F1.T)
    push!(weight_new, calc_pf(Ftmp, system.N[1], PMat=P))
    
    Utmp = Btmp[2] * F2.U
    Ftmp = UDT(Utmp, F2.D, F2.T)
    push!(weight_new, calc_pf(Ftmp, system.N[2], PMat=P))

    return weight_new, Btmp
end

function global_flip!(
    system::System, weight_old::Vector{W},
    σ::AbstractArray, P::Matrix{Tp},
    F1::UDT{T}, Bl1::AbstractMatrix{Tm}, 
    F2::UDT{T}, Bl2::AbstractMatrix{Tm}
) where {W<:Number, Tp<:Number, T<:Number, Tm<:Number}
    """
    Select a fraction of sites and flip their spins
    """
    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, Bltmp = calc_trial(σ, system, P, F1, F2)

    # accept ratio
    r = abs(exp(sum(weight_new) - sum(weight_old)))
    if rand() < min(1, r)
        weight_old .= weight_new
        Bl1 .= Bltmp[1]
        Bl2 .= Bltmp[2]
    else
        σ .*= σ_flip
    end

    return nothing
end

function update_cluster!(
    walker::Walker{Tw, Tf, UDT{Tf}, Tp, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDT{T}, F2::UDT{T}
) where {Tw, Tf, Tp, C, T}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    P = walker.tempdata.P
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    U1, D1, R1 = F1
    U2, D2, R2 = F2
    R1 *= cluster.B[cidx]
    R2 *= cluster.B[K + cidx]

    for i in 1 : k
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        global_flip!(
            system, walker.weight, σ, P,
            UDT(U1, D1, R1), Bl[i],
            UDT(U2, D2, R2), Bl[k + i]
        )

        U1 = Bl[i] * U1
        U2 = Bl[k + i] * U2
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{Tw, Tf, UDT{Tf}, Tp, C}
) where {Tw<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    Ns = system.V
    K = qmc.K

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpL = tempdata.FC.B
    tmpR = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in 1 : K
        QR_merge!(tmpM[1], tmpR[1], tmpL[cidx])
        QR_merge!(tmpM[2], tmpR[2], tmpL[K + cidx])
        update_cluster!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpL[cidx], tmpR[1])
        copyto!(tmpL[K + cidx], tmpR[2])

        tmpR[1] = QR_lmul(cluster.B[cidx], tmpR[1])
        tmpR[2] = QR_lmul(cluster.B[K + cidx], tmpR[2])
    end

    # save the propagation results
    copyto!.(walker.F, tmpR)
    # then reset Fτ to unit matrix
    tmpR .= [UDT(Ns), UDT(Ns)]

    return nothing
end

function update_cluster_reverse!(
    walker::Walker{Tw, Tf, UDT{Tf}, Tp, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDT{T}, F2::UDT{T}
) where {Tw, Tf, Tp, C, T}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    P = walker.tempdata.P
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    U1, D1, R1 = F1
    U2, D2, R2 = F2
    U1 = cluster.B[cidx] * U1
    U2 = cluster.B[K + cidx] * U2

    for i in k : -1 : 1
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        U1 = inv(Bl[i]) * U1
        U2 = inv(Bl[k + i]) * U2

        global_flip!(
            system, walker.weight, σ, P,
            UDT(U1, D1, R1), Bl[i],
            UDT(U2, D2, R2), Bl[k + i]
        )

        R1 *= Bl[i]
        R2 *= Bl[k + i]
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function reverse_sweep!(
    system::System, qmc::QMC, 
    walker::Walker{Tw, Tf, UDT{Tf}, Tp, C}
) where {Tw<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep the walker over the entire space-time lattice in the reverse order
    """
    Ns = system.V
    K = qmc.K

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpR = tempdata.FC.B
    tmpL = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in K : -1 : 1
        QR_merge!(tmpM[1], tmpR[cidx], tmpL[1])
        QR_merge!(tmpM[2], tmpR[cidx + K], tmpL[2])
        update_cluster_reverse!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpR[cidx], tmpL[1])
        copyto!(tmpR[cidx + K], tmpL[2])

        tmpL[1] = QR_rmul(tmpL[1], cluster.B[cidx])
        tmpL[2] = QR_rmul(tmpL[2], cluster.B[K + cidx])
    end

    # save the propagation results
    copyto!.(walker.F, tmpL)
    # tnen reset Fτ to unit matrix
    tmpL .= [UDT(Ns), UDT(Ns)]

    return nothing
end

### UDT with low-rank truncation ###
function calc_pf(
    F::UDTlr, N::Int64;
    Ns = length(F.t[]),
    PMat = zeros(ComplexF64, Ns + 1, Ns),
    isRepart::Bool = false, rpThld::Float64 = 1e-4
)
    if isRepart
        λocc, λf = repartition(F, rpThld)
        λ = vcat(λf, λocc)
        return pf_recursion(λ, N, P=PMat)
    else
        λ = eigvals(F)
        return pf_recursion(λ, N, P=PMat)
    end
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, P::Matrix{Tp}, F1::UDTlr{T}, F2::UDTlr{T}
) where {T<:Number, Tp<:Number}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    weight_new = Vector{Tp}()
    Btmp = singlestep_matrix(σ, system)

    @views Utmp = Btmp[1] * F1.U[:, F1.t[]]
    Ftmp = UDTlr(Utmp, F1.D, F1.T, F1.t)
    push!(weight_new, calc_pf(Ftmp, system.N[1], PMat=P))
    
    @views Utmp = Btmp[2] * F2.U[:, F2.t[]]
    Ftmp = UDTlr(Utmp, F2.D, F2.T, F2.t)
    push!(weight_new, calc_pf(Ftmp, system.N[2], PMat=P))

    return weight_new, Btmp
end

function local_flip!()
    """
    Flip the spins site by site
    """
end

function global_flip!(
    system::System, weight_old::Vector{W},
    σ::AbstractArray, P::AbstractMatrix{Tp},
    F1::UDTlr{T}, Bl1::AbstractMatrix{Tm}, 
    F2::UDTlr{T}, Bl2::AbstractMatrix{Tm}
) where {W<:Number, Tp<:Number, T<:Number, Tm<:Number}
    """
    Select a fraction of sites and flip their spins
    """
    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, Bltmp = calc_trial(σ, system, P, F1, F2)

    # accept ratio
    r = abs(exp(sum(weight_new) - sum(weight_old)))
    if rand() < min(1, r)
        weight_old .= weight_new
        copyto!(Bl1, Bltmp[1])
        copyto!(Bl2, Bltmp[2])
    else
        σ .*= σ_flip
    end

    return nothing
end

function update_cluster!(
    walker::Walker{Tw, Tf, UDTlr{Tf}, Tp, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {Tw, Tf, Tp, C, T}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    P = walker.tempdata.P
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    U1, D1, R1 = F1
    U2, D2, R2 = F2
    R1 *= cluster.B[cidx]
    R2 *= cluster.B[K + cidx]

    for i in 1 : k
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        global_flip!(
            system, walker.weight, σ, P,
            UDTlr(U1, D1, R1, F1.t), Bl[i],
            UDTlr(U2, D2, R2, F2.t), Bl[k + i]
        )

        U1 = Bl[i] * U1
        U2 = Bl[k + i] * U2
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker{Tw, Tf, UDTlr{Tf}, Tp, C}
) where {Tw<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    N = system.N
    K = qmc.K

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpL = tempdata.FC.B
    tmpR = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in 1 : K
        QR_merge!(tmpM[1], tmpR[1], tmpL[cidx], N[1], qmc.lrThld)
        QR_merge!(tmpM[2], tmpR[2], tmpL[K + cidx], N[2], qmc.lrThld)
        update_cluster!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpL[cidx], tmpR[1])
        copyto!(tmpL[K + cidx], tmpR[2])

        tmpR[1] = QR_lmul(cluster.B[cidx], tmpR[1], N[1], qmc.lrThld)
        tmpR[2] = QR_lmul(cluster.B[K + cidx], tmpR[2], N[2], qmc.lrThld)
    end

    # save the propagation results
    copyto!.(walker.F, tmpR)
    # then reset Fτ to unit matrix
    reset!.(tmpR)

    return nothing
end

function update_cluster_reverse!(
    walker::Walker{Tw, Tf, UDTlr{Tf}, Tp, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {Tw, Tf, Tp, C, T}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    P = walker.tempdata.P
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    U1, D1, R1 = F1
    U2, D2, R2 = F2
    U1 = cluster.B[cidx] * U1
    U2 = cluster.B[K + cidx] * U2

    for i in k : -1 : 1
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        U1 = inv(Bl[i]) * U1
        U2 = inv(Bl[k + i]) * U2

        global_flip!(
            system, walker.weight, σ, P,
            UDTlr(U1, D1, R1, F1.t), Bl[i],
            UDTlr(U2, D2, R2, F2.t), Bl[k + i]
        )

        R1 *= Bl[i]
        R2 *= Bl[k + i]
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function reverse_sweep!(
    system::System, qmc::QMC, 
    walker::Walker{Tw, Tf, UDTlr{Tf}, Tp, C}
) where {Tw<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep the walker over the entire space-time lattice in the reverse order
    """
    N = system.N
    K = qmc.K

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpR = tempdata.FC.B
    tmpL = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in K : -1 : 1
        QR_merge!(tmpM[1], tmpR[cidx], tmpL[1], N[1], qmc.lrThld)
        QR_merge!(tmpM[2], tmpR[cidx + K], tmpL[2], N[2], qmc.lrThld)
        update_cluster_reverse!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpR[cidx], tmpL[1])
        copyto!(tmpR[cidx + K], tmpL[2])

        tmpL[1] = QR_rmul(tmpL[1], cluster.B[cidx], N[1], qmc.lrThld)
        tmpL[2] = QR_rmul(tmpL[2], cluster.B[K + cidx], N[2], qmc.lrThld)
    end

    # save the propagation results
    copyto!.(walker.F, tmpL)
    # then reset Fτ to unit matrix
    reset!.(tmpL)

    return nothing
end
