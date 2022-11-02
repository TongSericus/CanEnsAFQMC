"""
    Replica sampling in GCE
"""
function global_flip!(
    system::System, walker1::GeneralGCWalker, walker2::GeneralGCWalker, 
    σ1::AbstractArray, σ2::AbstractArray,
    F1::LDR{T,E}, F2::LDR{T,E}, Bl1::AbstractVector{Tb}, 
    F3::LDR{T,E}, F4::LDR{T,E}, Bl2::AbstractVector{Tb}
) where {Tb, T<:Number, E<:Real}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old_1 = walker1.weight
    weight_old_2 = walker2.weight

    σ_flip_1 = 2 * (rand(system.V) .< 0.5) .- 1
    σ1 .*= σ_flip_1
    weight_new_1, sign_new_1, Bl1_tmp = calc_trial(σ1, system, walker1, F1, F2)

    σ_flip_2 = 2 * (rand(system.V) .< 0.5) .- 1
    σ2 .*= σ_flip_2
    weight_new_2, sign_new_2, Bl2_tmp = calc_trial(σ2, system, walker2, F3, F4)

    # accept ratio
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
        copyto!(walker1.sign, sign_new_1)
        copyto!.(Bl1, Bl1_tmp)

        σ2 .*= σ_flip_2
    elseif idx == 3
        # reject trial 1 but accept trial 1
        σ1 .*= σ_flip_1
        
        copyto!(weight_old_2, weight_new_2)
        copyto!(walker2.sign, sign_new_2)
        copyto!.(Bl2, Bl2_tmp)
    elseif idx == 4
        # accept both trials
        copyto!(weight_old_1, weight_new_1)
        copyto!(walker1.sign, sign_new_1)
        copyto!.(Bl1, Bl1_tmp)

        copyto!(weight_old_2, weight_new_2)
        copyto!(walker2.sign, sign_new_2)
        copyto!.(Bl2, Bl2_tmp)
    end

    return nothing
end

function update_cluster!(
    walker1::GeneralGCWalker, walker2::GeneralGCWalker, system::System, qmc::QMC, 
    cidx::Int64, F1::Vector{LDR{T,E}}, F2::Vector{LDR{T,E}}
) where {T<:Number, E<:Real}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    Bl1 = walker1.tempdata.cluster.B
    cluster1 = walker1.cluster

    Bl2 = walker2.tempdata.cluster.B
    cluster2 = walker2.cluster

    L1, d1, R1 = collectL(F1), collectd(F1), collectR(F1)
    R1 .*= [cluster1.B[cidx], cluster1.B[K + cidx]]
    
    L2, d2, R2 = collectL(F2), collectd(F2), collectR(F2)
    R2 .*= [cluster2.B[cidx], cluster2.B[K + cidx]]

    for i in 1 : k
        σ1 = @view walker1.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl1[i], Bl1[k + i], σ1, system, tmpmat = walker1.ws.M)
        R1 .*= [inv(Bl1[i]), inv(Bl1[k + i])]

        σ2 = @view walker2.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl2[i], Bl2[k + i], σ2, system, tmpmat = walker2.ws.M)
        R2 .*= [inv(Bl2[i]), inv(Bl2[k + i])]

        global_flip!(
            system, walker1, walker2, σ1, σ2,
            LDR(L1[1], d1[1], R1[1]), LDR(L1[2], d1[2], R1[2]), [Bl1[i], Bl1[k + i]],
            LDR(L2[1], d2[1], R2[1]), LDR(L2[2], d2[2], R2[2]), [Bl2[i], Bl2[k + i]]
        )

        L1 = [Bl1[i], Bl1[k + i]] .* L1
        L2 = [Bl2[i], Bl2[k + i]] .* L2
    end

    @views copyto!(cluster1.B[cidx], prod(Bl1[k:-1:1]))
    @views copyto!(cluster1.B[K + cidx], prod(Bl1[2*k:-1:k+1]))

    @views copyto!(cluster2.B[cidx], prod(Bl2[k:-1:1]))
    @views copyto!(cluster2.B[K + cidx], prod(Bl2[2*k:-1:k+1]))

    return nothing
end

function sweep!(system::System, qmc::QMC, walker1::GeneralGCWalker, walker2::GeneralGCWalker)
    """
    Sweep the walker over the entire space-time lattice
    """
    K = qmc.K

    ws = walker1.ws

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
        mul!(tmpM1[1], tmpR1[1], tmpL1[cidx], ws)
        mul!(tmpM1[2], tmpR1[2], tmpL1[K + cidx], ws)

        mul!(tmpM2[1], tmpR2[1], tmpL2[cidx], ws)
        mul!(tmpM2[2], tmpR2[2], tmpL2[K + cidx], ws)

        update_cluster!(
            walker1, walker2, system, qmc, cidx, tmpM1, tmpM2
        )

        copyto!(tmpL1[cidx], tmpR1[1])
        copyto!(tmpL1[K + cidx], tmpR1[2])

        lmul!(cluster1.B[cidx], tmpR1[1], ws)
        lmul!(cluster1.B[K + cidx], tmpR1[2], ws)

        copyto!(tmpL2[cidx], tmpR2[1])
        copyto!(tmpL2[K + cidx], tmpR2[2])

        lmul!(cluster2.B[cidx], tmpR2[1], ws)
        lmul!(cluster2.B[K + cidx], tmpR2[2], ws)
    end

    # save the propagation results
    copyto!.(walker1.F, tmpR1)
    copyto!.(walker2.F, tmpR2)
    # then reset Fτ to unit matrix
    ldr!(tmpR1[1], I)
    ldr!(tmpR1[2], I)
    ldr!(tmpR2[1], I)
    ldr!(tmpR2[2], I)

    return nothing
end

function update_cluster_reverse!(
    walker1::GeneralGCWalker, walker2::GeneralGCWalker, system::System, qmc::QMC, 
    cidx::Int64, F1::Vector{LDR{T,E}}, F2::Vector{LDR{T,E}}
) where {T<:Number, E<:Real, C}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    Bl1 = walker1.tempdata.cluster.B
    cluster1 = walker1.cluster

    Bl2 = walker2.tempdata.cluster.B
    cluster2 = walker2.cluster

    L1, d1, R1 = collectL(F1), collectd(F1), collectR(F1)
    L1 = [cluster1.B[cidx], cluster1.B[K + cidx]] .* L1

    L2, d2, R2 = collectL(F2), collectd(F2), collectR(F2)
    L2 = [cluster2.B[cidx], cluster2.B[K + cidx]] .* L2

    for i in k : -1 : 1
        σ1 = @view walker1.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl1[i], Bl1[k + i], σ1, system, tmpmat = walker1.ws.M)
        L1 = [inv(Bl1[i]), inv(Bl1[k + i])] .* L1

        σ2 = @view walker2.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl2[i], Bl2[k + i], σ2, system, tmpmat = walker2.ws.M)
        L2 = [inv(Bl2[i]), inv(Bl2[k + i])] .* L2

        global_flip!(
            system, walker1, walker2, σ1, σ2,
            LDR(L1[1], d1[1], R1[1]), LDR(L1[2], d1[2], R1[2]), [Bl1[i], Bl1[k + i]],
            LDR(L2[1], d2[1], R2[1]), LDR(L2[2], d2[2], R2[2]), [Bl2[i], Bl2[k + i]]
        )

        R1 .*= [Bl1[i], Bl1[k + i]]
        R2 .*= [Bl2[i], Bl2[k + i]]
    end

    @views copyto!(cluster1.B[cidx], prod(Bl1[k:-1:1]))
    @views copyto!(cluster1.B[K + cidx], prod(Bl1[2*k:-1:k+1]))

    @views copyto!(cluster2.B[cidx], prod(Bl2[k:-1:1]))
    @views copyto!(cluster2.B[K + cidx], prod(Bl2[2*k:-1:k+1]))

    return nothing
end

function reverse_sweep!(system::System, qmc::QMC, walker1::GeneralGCWalker, walker2::GeneralGCWalker)
    """
    Sweep the walker over the entire space-time lattice in the reverse order
    """
    K = qmc.K

    ws = walker1.ws

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
        mul!(tmpM1[1], tmpR1[cidx], tmpL1[1], ws)
        mul!(tmpM1[2], tmpR1[cidx + K], tmpL1[2], ws)

        mul!(tmpM2[1], tmpR2[cidx], tmpL2[1], ws)
        mul!(tmpM2[2], tmpR2[cidx + K], tmpL2[2], ws)

        update_cluster_reverse!(
            walker1, walker2, system, qmc, cidx, tmpM1, tmpM2
        )

        copyto!(tmpR1[cidx], tmpL1[1])
        copyto!(tmpR1[cidx + K], tmpL1[2])

        rmul!(tmpL1[1], cluster1.B[cidx], ws)
        rmul!(tmpL1[2], cluster1.B[K + cidx], ws)

        copyto!(tmpR2[cidx], tmpL2[1])
        copyto!(tmpR2[cidx + K], tmpL2[2])

        rmul!(tmpL2[1], cluster2.B[cidx], ws)
        rmul!(tmpL2[2], cluster2.B[K + cidx], ws)
    end

    # save the propagation results
    copyto!.(walker1.F, tmpL1)
    copyto!.(walker2.F, tmpL2)
    # tnen reset Fτ to unit matrix
    ldr!(tmpL1[1], I)
    ldr!(tmpL1[2], I)
    ldr!(tmpL2[1], I)
    ldr!(tmpL2[2], I)

    return nothing
end
