convert_auxidx(σ::Int64) = Int64((σ + 1) / 2 + 1)

function update_cluster!(
    walker::GCEWalker{T, C}, 
    system::System, qmc::QMC, cidx::Int64
) where {T<:FloatType, C}
    k = qmc.stab_interval
    Bl = Cluster(system.V, 2 * k)
    G = walker.G
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * k + i
        @views σ = convert_auxidx.(walker.auxfield[:, l])
        for j in 1 : system.V
            # ratio of determinants
            d_up = 1 + α[1, σ[j]] * (1 - G[1][j, j])
            d_dn = 1 + α[2, σ[j]] * (1 - G[2][j, j])

            r = abs(d_up * d_dn)
            p = r / (1 + r)

            if rand() < p
                walker.auxfield[j, l] *= -1
                G[1] = updateG(G[1], α[1, σ[j]], d_up, j)
                G[2] = updateG(G[2], α[2, σ[j]], d_dn, j)
            end
        end
        Bl.B[i], Bl.B[k + i] = singlestep_matrix(walker.auxfield[:, l], system)

        # wrap the Green's function
        B = (system.Bk) * Bl.B[i] * inv(system.Bk)
        G[1] = B * G[1] * inv(B)
        B = (system.Bk) * Bl.B[k + i] * inv(system.Bk)
        G[2] = B * G[2] * inv(B)
    end
    walker.cluster.B[cidx] = prod(Bl, k : -1 : 1)
    walker.cluster.B[qmc.K + cidx] = prod(Bl, 2*k : -1 : k+1)

    # wrap is not stable and G needs to be periodically recomputed
    recomputeG(system, qmc, walker, cidx)
end

function sweep!(
    system::System, qmc::QMC, 
    walker::GCEWalker{T, C}
) where {T<:FloatType, C}
    for cidx in 1 : qmc.K
        walker = update_cluster!(walker, system, qmc, cidx)
    end

    return walker
end

### Replica Sampling ###
function update_cluster!(
    walker1::GCEWalker{T1, C}, walker2::GCEWalker{T2, C},
    system::System, qmc::QMC, cidx::Int64
) where {T1<:FloatType, T2<:FloatType, C}
    """
        Update the whole cluster for two copies of walker
    """
    k = qmc.stab_interval

    Bl1 = Cluster(system.V, 2 * k)
    G1 = walker1.G
    α1 = walker1.α

    Bl2 = Cluster(system.V, 2 * k)
    G2 = walker2.G
    α2 = walker2.α

    for i in 1 : k
        l = (cidx - 1) * k + i
        @views σ1 = convert_auxidx.(walker1.auxfield[:, l])
        @views σ2 = convert_auxidx.(walker2.auxfield[:, l])
        for j in 1 : system.V
            # ratio of determinants
            d1_up = 1 + α1[1, σ1[j]] * (1 - G1[1][j, j])
            d1_dn = 1 + α1[2, σ1[j]] * (1 - G1[2][j, j])
            r1 = abs(d1_up * d1_dn)
            
            d2_up = 1 + α2[1, σ2[j]] * (1 - G2[1][j, j])
            d2_dn = 1 + α2[2, σ2[j]] * (1 - G2[2][j, j])
            r2 = abs(d2_up * d2_dn)

            idx = heatbath_sampling([1, r1, r2, r1 * r2])
            
            if idx == 2
                # accept trial 1 but reject trial 2
                walker1.auxfield[j, l] *= -1
                G1[1] = updateG(G1[1], α1[1, σ1[j]], d1_up, j)
                G1[2] = updateG(G1[2], α1[2, σ1[j]], d1_dn, j)
            elseif idx == 3
                # reject trial 1 but accept trial 1
                walker2.auxfield[j, l] *= -1
                G2[1] = updateG(G2[1], α2[1, σ2[j]], d2_up, j)
                G2[2] = updateG(G2[2], α2[2, σ2[j]], d2_dn, j)
            elseif idx == 4
                # accept both trials
                walker1.auxfield[j, l] *= -1
                G1[1] = updateG(G1[1], α1[1, σ1[j]], d1_up, j)
                G1[2] = updateG(G1[2], α1[2, σ1[j]], d1_dn, j)

                walker2.auxfield[j, l] *= -1
                G2[1] = updateG(G2[1], α2[1, σ2[j]], d2_up, j)
                G2[2] = updateG(G2[2], α2[2, σ2[j]], d2_dn, j)
            end
        end
        Bl1.B[i], Bl1.B[k + i] = singlestep_matrix(walker1.auxfield[:, l], system)
        Bl2.B[i], Bl2.B[k + i] = singlestep_matrix(walker2.auxfield[:, l], system)

        # wrap the Green's function
        B = (system.Bk) * Bl1.B[i] * inv(system.Bk)
        G1[1] = B * G1[1] * inv(B)
        B = (system.Bk) * Bl1.B[k + i] * inv(system.Bk)
        G1[2] = B * G1[2] * inv(B)

        B = (system.Bk) * Bl2.B[i] * inv(system.Bk)
        G2[1] = B * G2[1] * inv(B)
        B = (system.Bk) * Bl2.B[k + i] * inv(system.Bk)
        G2[2] = B * G2[2] * inv(B)
    end
    walker1.cluster.B[cidx] = prod(Bl1, k : -1 : 1)
    walker1.cluster.B[qmc.K + cidx] = prod(Bl1, 2*k : -1 : k+1)

    walker2.cluster.B[cidx] = prod(Bl2, k : -1 : 1)
    walker2.cluster.B[qmc.K + cidx] = prod(Bl2, 2*k : -1 : k+1)

    # wrap is not stable and G needs to be periodically recomputed
    walker1 = recomputeG(system, qmc, walker1, cidx)
    walker2 = recomputeG(system, qmc, walker2, cidx)

    return walker1, walker2
end

function sweep!(
    system::System, qmc::QMC, 
    walker1::GCEWalker{T1, C}, walker2::GCEWalker{T2, C}
) where {T1<:FloatType, T2<:FloatType, C}
    for cidx in 1 : qmc.K
        walker1, walker2 = update_cluster!(walker1, walker2, system, qmc, cidx)
    end

    return walker1, walker2
end

### A New Scheme using StableLinearAlgebra Package ###
function calc_trial(
    σ::AbstractArray{Int64}, system::System, walker::GCWalker, F1::LDR{T,E}, F2::LDR{T,E}
) where {T<:Number, E<:Real, C}
    """
    Compute the weight of a trial configuration using a potential repartition scheme
    """
    expβμ = walker.expβμ[]
    G = walker.G
    ws = walker.ws

    weight = similar(walker.weight)
    sign = similar(walker.sign)
    Ltmp = similar(F1.L)

    Btmp = singlestep_matrix(σ, system)

    mul!(Ltmp, Btmp[1], F1.L)
    Ftmp = LDR(Ltmp, F1.d, F1.R)
    weight[1], sign[1] = inv_IpμA!(G[1], Ftmp, expβμ, ws)

    mul!(Ltmp, Btmp[2], F2.L)
    Ftmp = LDR(Ltmp, F2.d, F2.R)
    weight[2], sign[2] = inv_IpμA!(G[2], Ftmp, expβμ, ws)

    return -weight, sign, Btmp
end

function global_flip!(
    system::System, walker::GCWalker, σ::AbstractArray,
    F1::LDR{T,E}, Bl1::AbstractMatrix{Tm}, 
    F2::LDR{T,E}, Bl2::AbstractMatrix{Tm}
) where {Tm<:Number, T<:Number, E<:Real, C}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old = walker.weight

    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, sign_new, Bltmp = calc_trial(σ, system, walker, F1, F2)

    # accept ratio
    r = abs(exp(sum(weight_new) - sum(weight_old)))
    if rand() < min(1, r)
        weight_old .= weight_new
        walker.sign .= sign_new

        copyto!(Bl1, Bltmp[1])
        copyto!(Bl2, Bltmp[2])
    else
        σ .*= σ_flip
    end

    return nothing
end

function update_cluster!(
    walker::GCWalker, system::System, qmc::QMC, 
    cidx::Int64, F1::LDR{T,E}, F2::LDR{T,E}
) where {T<:Number, E<:Real}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    L1, d1, R1 = F1
    L2, d2, R2 = F2
    R1 *= cluster.B[cidx]
    R2 *= cluster.B[K + cidx]

    for i in 1 : k
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        global_flip!(
            system, walker, σ,
            LDR(L1, d1, R1), Bl[i],
            LDR(L2, d2, R2), Bl[k + i]
        )

        L1 = Bl[i] * L1
        L2 = Bl[k + i] * L2
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function sweep!(system::System, qmc::QMC, walker::GCWalker)
    """
    Sweep the walker over the entire space-time lattice
    """
    K = qmc.K

    ws = walker.ws

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpL = tempdata.FC.B
    tmpR = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in 1 : K
        mul!(tmpM[1], tmpR[1], tmpL[cidx], ws)
        mul!(tmpM[2], tmpR[2], tmpL[K + cidx], ws)
        update_cluster!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpL[cidx], tmpR[1])
        copyto!(tmpL[K + cidx], tmpR[2])

        lmul!(cluster.B[cidx], tmpR[1], ws)
        lmul!(cluster.B[K + cidx], tmpR[2], ws)
    end

    # save the propagation results
    copyto!.(walker.F, tmpR)
    # then reset Fτ to unit matrix
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

function update_cluster_reverse!(
    walker::GCWalker, system::System, qmc::QMC, 
    cidx::Int64, F1::LDR{T,E}, F2::LDR{T,E}
) where {T<:Number, E<:Real, C}
    """
    Update the propagation matrices of size equaling to the stablization interval
    """
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster

    L1, d1, R1 = F1
    L2, d2, R2 = F2
    L1 = cluster.B[cidx] * L1
    L2 = cluster.B[K + cidx] * L2

    for i in k : -1 : 1
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system)
        L1 = inv(Bl[i]) * L1
        L2 = inv(Bl[k + i]) * L2

        global_flip!(
            system, walker, σ,
            LDR(L1, d1, R1), Bl[i],
            LDR(L2, d2, R2), Bl[k + i]
        )

        R1 *= Bl[i]
        R2 *= Bl[k + i]
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function reverse_sweep!(system::System, qmc::QMC, walker::GCWalker)
    """
    Sweep the walker over the entire space-time lattice in the reverse order
    """
    K = qmc.K

    ws = walker.ws

    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpR = tempdata.FC.B
    tmpL = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in K : -1 : 1
        mul!(tmpM[1], tmpR[cidx], tmpL[1], ws)
        mul!(tmpM[2], tmpR[cidx + K], tmpL[2], ws)
        update_cluster_reverse!(
            walker, system, qmc, cidx, tmpM[1], tmpM[2]
        )

        copyto!(tmpR[cidx], tmpL[1])
        copyto!(tmpR[cidx + K], tmpL[2])

        rmul!(tmpL[1], cluster.B[cidx], ws)
        rmul!(tmpL[2], cluster.B[K + cidx], ws)
    end

    # save the propagation results
    copyto!.(walker.F, tmpL)
    # tnen reset Fτ to unit matrix
    ldr!(tmpL[1], I)
    ldr!(tmpL[2], I)

    return nothing
end
