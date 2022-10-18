"""
    MCMC with two copies of walker for Renyi-2 Entropy / Purity
"""
function heatbath_sampling(weights::Vector{Float64})
    """
    Simple heatbath sampling
    """
    norm = sum(weights)
    cum_prob = cumsum(weights / norm)

    u = rand()
    idx = 1
    while u > cum_prob[idx]
        idx += 1
    end

    return idx
end

function global_flip!(
    system::System,
    weight_old_1::Vector{W}, weight_old_2::Vector{W},
    σ1::AbstractArray{Ts}, σ2::AbstractArray{Ts},
    P::AbstractMatrix{Tp},
    F1::UDTlr{T}, F2::UDTlr{T}, Bl1::AbstractVector{Tb},
    F3::UDTlr{T}, F4::UDTlr{T}, Bl2::AbstractVector{Tb}
) where {W<:Number, Ts<:Number, Tp<:Number, T<:Number, Tb}
    """
    Select a fraction of sites and flip their spins
    """
    σ_flip_1 = 2 * (rand(system.V) .< 0.5) .- 1
    σ1 .*= σ_flip_1
    weight_new_1, Bl1_tmp = calc_trial(σ1, system, P, F1, F2)

    σ_flip_2 = 2 * (rand(system.V) .< 0.5) .- 1
    σ2 .*= σ_flip_2
    weight_new_2, Bl2_tmp = calc_trial(σ2, system, P, F3, F4)

    r1 = sum(weight_new_1) - sum(weight_old_1)
    r1 = abs(exp(r1))
    r2 = sum(weight_new_2) - sum(weight_old_2)
    r2 = abs(exp(r2))

    idx = heatbath_sampling([1, r1, r2, r1 * r2])

    if idx == 1
        # reject both trials
        σ1 .*= σ_flip_1
        
        σ2 .*= σ_flip_2
    elseif idx == 2
        # accept trial 1 but reject trial 2
        copyto!(weight_old_1, weight_new_1)
        copyto!.(Bl1, Bl1_tmp)

        σ2 .*= σ_flip_2
    elseif idx == 3
        # reject trial 1 but accept trial 1
        σ1 .*= σ_flip_1
        
        copyto!(weight_old_2, weight_new_2)
        copyto!.(Bl2, Bl2_tmp)
    elseif idx == 4
        # accept both trials
        copyto!(weight_old_1, weight_new_1)
        copyto!.(Bl1, Bl1_tmp)

        copyto!(weight_old_2, weight_new_2)
        copyto!.(Bl2, Bl2_tmp)
    end

    return nothing
end

function update_cluster!(
    walker1::Walker{Tw1, Tf, UDTlr{Tf}, Tp, C},
    walker2::Walker{Tw2, Tf, UDTlr{Tf}, Tp, C}, 
    system::System, qmc::QMC, 
    cidx::Int64, F1::Vector{UDTlr{Tf}}, F2::Vector{UDTlr{Tf}}
) where {Tw1<:Number, Tw2<:Number, Tf<:Number, Tp<:Number, C}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.stab_interval
    K = qmc.K
    P = walker1.tempdata.P

    Bl1 = walker1.tempdata.cluster
    cluster1 = walker1.cluster

    Bl2 = walker2.tempdata.cluster
    cluster2 = walker2.cluster

    U1, D1, R1 = collectU(F1), collectD(F1), collectT(F1)
    R1 .*= [cluster1.B[cidx], cluster1.B[K + cidx]]

    U2, D2, R2 = collectU(F2), collectD(F2), collectT(F2)
    R2 .*= [cluster2.B[cidx], cluster2.B[K + cidx]]

    for i in 1 : k
        σ1 = @view walker1.auxfield[:, (cidx - 1) * k + i]
        singlestep_matrix!(Bl1.B[i], Bl1.B[k + i], σ1, system)
        R1 .*= [inv(Bl1.B[i]), inv(Bl1.B[k + i])]

        σ2 = @view walker2.auxfield[:, (cidx - 1) * k + i]
        singlestep_matrix!(Bl2.B[i], Bl2.B[k + i], σ2, system)
        R2 .*= [inv(Bl2.B[i]), inv(Bl2.B[k + i])]

        global_flip!(
            system, 
            walker1.weight, walker2.weight,
            σ1, σ2, P,
            UDTlr(U1[1], D1[1], R1[1], F1[1].t), UDTlr(U1[2], D1[2], R1[2], F1[2].t),
            [Bl1.B[i], Bl1.B[k + i]],
            UDTlr(U2[1], D2[1], R2[1], F2[1].t), UDTlr(U2[2], D2[2], R2[2], F2[2].t),
            [Bl2.B[i], Bl2.B[k + i]]
        )
        
        U1 = [Bl1.B[i], Bl1.B[k + i]] .* U1
        U2 = [Bl2.B[i], Bl2.B[k + i]] .* U2
    end

    cluster1.B[cidx] = prod(Bl1, collect(k:-1:1))
    cluster1.B[K + cidx] = prod(Bl1, collect(2*k:-1:k+1))

    cluster2.B[cidx] = prod(Bl2, collect(k:-1:1))
    cluster2.B[K + cidx] = prod(Bl2, collect(2*k:-1:k+1))

    return nothing
end

function sweep!(
    system::System, qmc::QMC, 
    walker1::Walker{Tw1, Tf, UDTlr{Tf}, Tp, C},
    walker2::Walker{Tw2, Tf, UDTlr{Tf}, Tp, C}
) where {Tw1<:Number, Tw2<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep two copies of walker over the entire space-time lattice
    """
    N = system.N
    K = qmc.K

    cluster1 = walker1.cluster
    tempdata1 = walker1.tempdata
    tmpL1 = tempdata1.FC.B
    tmpR1 = tempdata1.Fτ
    tmpM1 = tempdata1.FM

    cluster2 = walker2.cluster
    tempdata2 = walker2.tempdata
    tmpL2 = tempdata2.FC.B
    tmpR2 = tempdata2.Fτ
    tmpM2 = tempdata2.FM

    for cidx in 1 : K
        QR_merge!(tmpM1[1], tmpR1[1], tmpL1[cidx], N[1], qmc.lrThld)
        QR_merge!(tmpM1[2], tmpR1[2], tmpL1[K + cidx], N[2], qmc.lrThld)

        QR_merge!(tmpM2[1], tmpR2[1], tmpL2[cidx], N[1], qmc.lrThld)
        QR_merge!(tmpM2[2], tmpR2[2], tmpL2[K + cidx], N[2], qmc.lrThld)

        update_cluster!(
            walker1, walker2, system, qmc, cidx, tmpM1, tmpM2
        )
        
        copyto!(tmpL1[cidx], tmpR1[1])
        copyto!(tmpL1[K + cidx], tmpR1[2])

        tmpR1[1] = QR_lmul(cluster1.B[cidx], tmpR1[1], N[1], qmc.lrThld)
        tmpR1[2] = QR_lmul(cluster1.B[K + cidx], tmpR1[2], N[2], qmc.lrThld)

        copyto!(tmpL2[cidx], tmpR2[1])
        copyto!(tmpL2[K + cidx], tmpR2[2])

        tmpR2[1] = QR_lmul(cluster2.B[cidx], tmpR2[1], N[1], qmc.lrThld)
        tmpR2[2] = QR_lmul(cluster2.B[K + cidx], tmpR2[2], N[2], qmc.lrThld)
    end
    
    # save the propagation results
    copyto!.(walker1.F, tmpR1)
    copyto!.(walker2.F, tmpR2)
    # then reset
    reset!.(tmpR1)
    reset!.(tmpR2)

    return nothing
end

function reverse_sweep!(
    system::System, qmc::QMC, 
    walker1::Walker{Tw1, Tf, UDTlr{Tf}, Tp, C},
    walker2::Walker{Tw2, Tf, UDTlr{Tf}, Tp, C}
) where {Tw1<:Number, Tw2<:Number, Tf<:Number, Tp<:Number, C}
    """
    Sweep two copies of walker over the entire space-time lattice in the reverse order
    """
    N = system.N
    K = qmc.K

    cluster1 = walker1.cluster
    tempdata1 = walker1.tempdata
    tmpR1 = tempdata1.FC.B
    tmpL1 = tempdata1.Fτ
    tmpM1 = tempdata1.FM

    cluster2 = walker2.cluster
    tempdata2 = walker2.tempdata
    tmpR2 = tempdata2.FC.B
    tmpL2 = tempdata2.Fτ
    tmpM2 = tempdata2.FM

    for cidx in K : -1 : 1
        QR_merge!(tmpM1[1], tmpR1[cidx], tmpL1[1], N[1], qmc.lrThld)
        QR_merge!(tmpM1[2], tmpR1[cidx + K], tmpL1[2], N[2], qmc.lrThld)

        QR_merge!(tmpM2[1], tmpR2[cidx], tmpL2[1], N[1], qmc.lrThld)
        QR_merge!(tmpM2[2], tmpR2[cidx + K], tmpL2[2], N[2], qmc.lrThld)

        update_cluster!(
            walker1, walker2, system, qmc, cidx, tmpM1, tmpM2
        )
        
        copyto!(tmpR1[cidx], tmpL1[1])
        copyto!(tmpR1[cidx + K], tmpL1[2])

        tmpL1[1] = QR_rmul(tmpL1[1], cluster1.B[cidx], N[1], qmc.lrThld)
        tmpL1[2] = QR_rmul(tmpL1[2], cluster1.B[K + cidx], N[2], qmc.lrThld)

        copyto!(tmpR2[cidx], tmpL2[1])
        copyto!(tmpR2[cidx + K], tmpL2[2])

        tmpL2[1] = QR_rmul(tmpL2[1], cluster2.B[cidx], N[1], qmc.lrThld)
        tmpL2[2] = QR_rmul(tmpL2[2], cluster2.B[K + cidx], N[2], qmc.lrThld)
    end

    # save the propagation results
    copyto!.(walker1.F, tmpL1)
    copyto!.(walker2.F, tmpL2)
    # then reset
    reset!.(tmpL1)
    reset!.(tmpL2)

    return nothing
end
