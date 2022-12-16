"""
    Metropolis Sampling (MCMC) (for different factorizations)
"""
### UDR(deleted) ###
function compute_PF(
    F::LDR{T, E}, N::Int64;
    Ns = length(F.d),
    PMat = zeros(ComplexF64, Ns + 1, Ns)
) where {T, E}
    λ = eigvals(F)
    return pf_recursion(λ, N, P=PMat)
end


### UDT ###
function compute_PF(
    F::UDT{T}, N::Int64;
    Ns = length(F.D),
    PMat = zeros(ComplexF64, Ns + 1, Ns)
) where T
    λ = eigvals(F)
    return @views pf_recursion(λ, N, P = PMat[N + 1, :])
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, P::Matrix{Tp}, F1::UDT{T}, F2::UDT{T}
) where {T<:Number, Tp<:Number}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    weight_new = zeros(T, 2)
    sgn_new = zeros(Tp, 2)
    Btmp = singlestep_matrix(σ, system)

    Utmp = Btmp[1] * F1.U
    Ftmp = UDT(Utmp, F1.D, F1.T)
    weight_new[1], sgn_new[1] = compute_PF(Ftmp, system.N[1], PMat=P)
    
    Utmp = Btmp[2] * F2.U
    Ftmp = UDT(Utmp, F2.D, F2.T)
    weight_new[2], sgn_new[2] = compute_PF(Ftmp, system.N[2], PMat=P)

    return weight_new, sgn_new, Btmp
end

function global_flip!(
    system::System, walker::Walker,
    σ::AbstractArray, P::Matrix{Tp},
    F1::UDT{T}, Bl1::AbstractMatrix{Tm}, 
    F2::UDT{T}, Bl2::AbstractMatrix{Tm}
) where {Tp<:Number, T<:Number, Tm<:Number}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old = walker.weight
    sgn_old = walker.sign

    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, sgn_new, Bltmp = calc_trial(σ, system, P, F1, F2)

    # accept ratio
    r = exp(sum(weight_new) - sum(weight_old))
    if rand() < min(1, r)
        copyto!(weight_old, weight_new)
        copyto!(sgn_old, sgn_new)

        Bl1 .= Bltmp[1]
        Bl2 .= Bltmp[2]
    else
        σ .*= σ_flip
    end

    return nothing
end

function update_cluster!(
    walker::Walker{Tw, Ts, Tf, UDT{Tf}, E, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDT{T}, F2::UDT{T}
) where {Tw, Ts, Tf, E, C, T}
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
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        global_flip!(
            system, walker, σ, P,
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
    walker::Walker{Tw, Ts, Tf, E, UDT{Tf}, C}
) where {Tw, Ts, Tf, E, C}
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
    walker::Walker{Tw, Ts, Tf, UDT{Tf}, E, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDT{T}, F2::UDT{T}
) where {Tw, Ts, Tf, E, C, T}
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
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        U1 = inv(Bl[i]) * U1
        U2 = inv(Bl[k + i]) * U2

        global_flip!(
            system, walker, σ, P,
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
    walker::Walker{Tw, Ts, Tf, UDT{Tf}, E, C}
) where {Tw, Ts, Tf, E, C}
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
function compute_PF(
    F::UDTlr{T}, N::Int64;
    Ns = length(F.t[]),
    PMat = zeros(ComplexF64, N + 1, Ns),
    isRepart::Bool = false, rpThld::Float64 = 1e-4
) where T
    if isRepart
        λocc, λf = repartition(F, rpThld)
        λ = vcat(λf, λocc)
        return pf_recursion(λ, N, P = PMat)
    else
        λ = eigvals(F)
        return pf_recursion(λ, N, P = PMat)
    end
end

function calc_trial(
    σ::AbstractArray{Int64}, system::System, P::Matrix{Tp}, F1::UDTlr{T}, F2::UDTlr{T}
) where {T<:Number, Tp<:Number}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    weight_new = zeros(T, 2)
    sgn_new = zeros(Tp, 2)

    Btmp = singlestep_matrix(σ, system)

    @views Utmp = Btmp[1] * F1.U[:, F1.t[]]
    Ftmp = UDTlr(Utmp, F1.D, F1.T, F1.t)
    weight_new[1], sgn_new[1] = compute_PF(Ftmp, system.N[1], PMat=P)
    
    @views Utmp = Btmp[2] * F2.U[:, F2.t[]]
    Ftmp = UDTlr(Utmp, F2.D, F2.T, F2.t)
    weight_new[2], sgn_new[2] = compute_PF(Ftmp, system.N[2], PMat=P)

    return weight_new, sgn_new, Btmp
end

function local_flip!(
    system::System, walker::Walker,
    σ::AbstractArray, P::AbstractMatrix{Tp},
    F1::UDTlr{T}, Bl1::AbstractMatrix{Tm}, 
    F2::UDTlr{T}, Bl2::AbstractMatrix{Tm}
) where {Tp<:Number, T<:Number, Tm<:Number}
    """
    Flip the spins site by site
    """
    weight_old = walker.weight
    sgn_old = walker.sign

    for i in 1 : system.V
        σ[i] *= -1
        weight_new, sgn_new, Bltmp = calc_trial(σ, system, P, F1, F2)

        # accept ratio
        r = exp(sum(weight_new) - sum(weight_old))
        if rand() < min(1, r)
            copyto!(weight_old, weight_new)
            copyto!(sgn_old, sgn_new)

            copyto!(Bl1, Bltmp[1])
            copyto!(Bl2, Bltmp[2])
        else
            # revert the change
            σ[i] *= -1
        end
    end

    return nothing
end

function global_flip!(
    system::System, walker::Walker,
    σ::AbstractArray, P::AbstractMatrix{Tp},
    F1::UDTlr{T}, Bl1::AbstractMatrix{Tm}, 
    F2::UDTlr{T}, Bl2::AbstractMatrix{Tm}
) where {Tp<:Number, T<:Number, Tm<:Number}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old = walker.weight
    sgn_old = walker.sign

    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, sgn_new, Bltmp = calc_trial(σ, system, P, F1, F2)

    # accept ratio
    r = exp(sum(weight_new) - sum(weight_old))
    if rand() < min(1, r)
        copyto!(weight_old, weight_new)
        copyto!(sgn_old, sgn_new)

        copyto!(Bl1, Bltmp[1])
        copyto!(Bl2, Bltmp[2])
    else
        # revert the change
        σ .*= σ_flip
    end

    return nothing
end

function update_cluster!(
    walker::Walker{Tw, Ts, Tf, UDTlr{Tf}, E, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {Tw, Ts, Tf, E, C, T}
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
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        local_flip!(
            system, walker, σ, P,
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
    walker::Walker{Tw, Ts, Tf, UDTlr{Tf}, E, C}
) where {Tw, Ts, Tf, E, C}
    """
    Sweep the walker over the entire space-time lattice
    """
    N = system.N
    K = qmc.K
    ϵ = qmc.lrThld

    ws = walker.ws

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpL = tempdata.FC.B
    tmpR = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in 1 : K
        mul!(tmpM[1], tmpR[1], tmpL[cidx], N[1], ϵ, ws)
        mul!(tmpM[2], tmpR[2], tmpL[K + cidx], N[2], ϵ, ws)
        update_cluster!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpL[cidx], tmpR[1])
        copyto!(tmpL[K + cidx], tmpR[2])

        lmul!(cluster.B[cidx], tmpR[1], N[1], ϵ, ws)
        lmul!(cluster.B[K + cidx], tmpR[2], N[2], ϵ, ws)
    end

    # save the propagation results
    copyto!.(walker.F, tmpR)
    # then reset Fτ to unit matrix
    reset!.(tmpR)

    return nothing
end

function update_cluster_reverse!(
    walker::Walker{Tw, Ts, Tf, UDTlr{Tf}, E, C}, system::System, qmc::QMC, 
    cidx::Int64, F1::UDTlr{T}, F2::UDTlr{T}
) where {Tw, Ts, Tf, E, C, T}
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
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        U1 = inv(Bl[i]) * U1
        U2 = inv(Bl[k + i]) * U2

        local_flip!(
            system, walker, σ, P,
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
    walker::Walker{Tw, Ts, Tf, UDTlr{Tf}, E, C}
) where {Tw<:Number, Ts<:Number, Tf<:Number, E, C}
    """
    Sweep the walker over the entire space-time lattice in the reverse order
    """
    N = system.N
    K = qmc.K
    ϵ = qmc.lrThld

    ws = walker.ws

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpR = tempdata.FC.B
    tmpL = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in K : -1 : 1
        mul!(tmpM[1], tmpR[cidx], tmpL[1], N[1], ϵ, ws)
        mul!(tmpM[2], tmpR[cidx + K], tmpL[2], N[2], ϵ, ws)
        update_cluster_reverse!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpR[cidx], tmpL[1])
        copyto!(tmpR[cidx + K], tmpL[2])

        rmul!(tmpL[1], cluster.B[cidx], N[1], ϵ, ws)
        rmul!(tmpL[2], cluster.B[K + cidx], N[2], ϵ, ws)
    end

    # save the propagation results
    copyto!.(walker.F, tmpL)
    # then reset Fτ to unit matrix
    reset!.(tmpL)

    return nothing
end
