"""
    Monte Carlo Propagation in the GC
"""

flip_HSField(σ::Int) = Int64((σ + 1) / 2 + 1)

function update_G!(G::AbstractMatrix{T}, α::Float64, d::Float64, sidx::Int64) where T
    """
    Fast update the Green's function
    """
    ImG = I - G
    @views dG = α / d * ImG[:, sidx] * (G[sidx, :])'
    G .-= dG
end

function wrap_G!(G::AbstractMatrix{T}, B::AbstractMatrix{T}, ws::LDRWorkspace{T, E}) where {T, E}
    """
    Compute G' = B * G * B⁻¹
    """
    mul!(ws.M, B, G)
    # B⁻¹ calculation could be further optimized
    mul!(G, ws.M, inv(B))
end

function update_cluster!(
    walker::HubbardGCWalker, 
    system::System, qmc::QMC, cidx::Int64
)
    k = qmc.K_interval[cidx]
    K = qmc.K
    

    Bl = walker.tempdata.cluster.B
    cluster = walker.cluster
    G = walker.G
    α = walker.α

    for i in 1 : k
        l = (cidx - 1) * k + i
        @views σ = flip_HSField.(walker.auxfield[:, l])
        for j in 1 : system.V
            # compute ratios of determinants through G
            d_up = 1 + α[1, σ[j]] * (1 - G[1][j, j])
            d_dn = 1 + α[2, σ[j]] * (1 - G[2][j, j])

            r = abs(d_up * d_dn)

            if rand() < r
                walker.auxfield[j, l] *= -1
                update_G!(G[1], α[1, σ[j]], d_up, j)
                update_G!(G[2], α[2, σ[j]], d_dn, j)
            end
        end
        @views σ = walker.auxfield[:, l]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)

        # rank-1 update of the Green's function
        wrap_G!(G[1], Bl[i], walker.ws)
        wrap_G!(G[2], Bl[k + i], walker.ws)
    end

    @views copyto!(cluster.B[cidx], prod(Bl[k:-1:1]))
    @views copyto!(cluster.B[K + cidx], prod(Bl[2*k:-1:k+1]))

    return nothing
end

function sweep!(system::System, qmc::QMC, walker::HubbardGCWalker)
    """
    Sweep the walker over the entire space-time lattice
    """
    K = qmc.K

    ws = walker.ws
    weight = walker.weight
    sgn = walker.sign
    
    G = walker.G
    cluster = walker.cluster
    tempdata = walker.tempdata
    tmpL = tempdata.FC.B
    tmpR = tempdata.Fτ
    tmpM = tempdata.FM

    for cidx in 1 : K
        update_cluster!(walker, system, qmc, cidx)

        lmul!(cluster.B[cidx], tmpR[1], ws)
        lmul!(cluster.B[K + cidx], tmpR[2], ws)

        mul!(tmpM[1], tmpR[1], tmpL[cidx], ws)
        mul!(tmpM[2], tmpR[2], tmpL[K + cidx], ws)

        
        # G needs to be periodically recomputed
        weight[1], sgn[1] = inv_IpμA!(G[1], tmpM[1], walker.expβμ[], ws)
        weight[2], sgn[2] = inv_IpμA!(G[2], tmpM[2], walker.expβμ[], ws)
        weight .*= -1
    end

    # At the end of the simulation, recompute all partial factorizations
    run_full_propagation_reverse(walker.auxfield, system, qmc, ws, FC = walker.tempdata.FC)

    # reset Fτs to unit matrices
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

### General GC Sweep, No Rank-1 Update Scheme ###
function calc_trial(
    σ::AbstractArray{Int64}, system::System, walker::GeneralGCWalker, F1::LDR{T,E}, F2::LDR{T,E}
) where {T<:Number, E<:Real}
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

function local_flip!(
    system::System, walker::GeneralGCWalker, σ::AbstractArray,
    F1::LDR{T,E}, Bl1::AbstractMatrix{Tm}, 
    F2::LDR{T,E}, Bl2::AbstractMatrix{Tm}
) where {Tm<:Number, T<:Number, E<:Real}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old = walker.weight

    for i in 1 : system.V
        σ[i] *= -1
        weight_new, sign_new, Bltmp = calc_trial(σ, system, walker, F1, F2)

        # accept ratio
        r = exp(sum(weight_new) - sum(weight_old))
        if rand() < min(1, r)
            weight_old .= weight_new
            walker.sign .= sign_new

            copyto!(Bl1, Bltmp[1])
            copyto!(Bl2, Bltmp[2])
        else
            σ[i] *= -1
        end
    end

    return nothing
end

function global_flip!(
    system::System, walker::GeneralGCWalker, σ::AbstractArray,
    F1::LDR{T,E}, Bl1::AbstractMatrix{Tm}, 
    F2::LDR{T,E}, Bl2::AbstractMatrix{Tm}
) where {Tm<:Number, T<:Number, E<:Real}
    """
    Select a fraction of sites and flip their spins
    """
    weight_old = walker.weight

    σ_flip = 2 * (rand(system.V) .< 0.5) .- 1
    σ .*= σ_flip
    weight_new, sign_new, Bltmp = calc_trial(σ, system, walker, F1, F2)

    # accept ratio
    r = exp(sum(weight_new) - sum(weight_old))
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
    walker::GeneralGCWalker, system::System, qmc::QMC, 
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
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        R1 *= inv(Bl[i])
        R2 *= inv(Bl[k + i])

        local_flip!(
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

function sweep!(system::System, qmc::QMC, walker::GeneralGCWalker)
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

    # compute Green's function after sweep
    inv_IpμA!(walker.G[1], tmpR[1], walker.expβμ[], walker.ws)
    inv_IpμA!(walker.G[2], tmpR[2], walker.expβμ[], walker.ws)
    # save the propagation results
    copyto!.(walker.F, tmpR)
    # then reset Fτ to unit matrix
    ldr!(tmpR[1], I)
    ldr!(tmpR[2], I)

    return nothing
end

function update_cluster_reverse!(
    walker::GeneralGCWalker, system::System, qmc::QMC, 
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
    L1 = cluster.B[cidx] * L1
    L2 = cluster.B[K + cidx] * L2

    for i in k : -1 : 1
        σ = @view walker.auxfield[:, (cidx - 1) * qmc.stab_interval + i]
        singlestep_matrix!(Bl[i], Bl[k + i], σ, system, tmpmat = walker.ws.M)
        L1 = inv(Bl[i]) * L1
        L2 = inv(Bl[k + i]) * L2

        local_flip!(
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

function reverse_sweep!(system::System, qmc::QMC, walker::GeneralGCWalker)
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

    # compute Green's function after sweep
    inv_IpμA!(walker.G[1], tmpL[1], walker.expβμ[], walker.ws)
    inv_IpμA!(walker.G[2], tmpL[2], walker.expβμ[], walker.ws)
    # save the propagation results
    copyto!.(walker.F, tmpL)
    # tnen reset Fτ to unit matrix
    ldr!(tmpL[1], I)
    ldr!(tmpL[2], I)

    return nothing
end
