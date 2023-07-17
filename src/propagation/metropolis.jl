"""
    Metropolis Sampling (MCMC) (for different factorizations)
"""

function sweep!(system::System, qmc::QMC, walker::Walker; loop_number::Int = 1)
    if system.useChargeHST || qmc.forceSymmetry # charge decomposition
        for i in 1 : loop_number
            sweep!_symmetric(system, qmc, walker, direction=1)
            sweep!_symmetric(system, qmc, walker, direction=2)
        end

        return nothing
    else                                        # spin decomposition
        for i in 1 : loop_number
            sweep!_asymmetric(system, qmc, walker, direction=1)
            sweep!_asymmetric(system, qmc, walker, direction=2)
        end

        return nothing
    end
end

##################################################
##### Symmetric Sweep for Charge HS Transform ####
##################################################
"""
    local_flip!(...)

    Metropolis test by flipping a single spin
"""
function local_flip!(
    system::System, qmc::QMC, walker::Walker,
    σ::AbstractArray, Bl::AbstractMatrix
)
    weight = walker.weight
    weight′ = walker.weight′
    sgn = walker.sign
    sgn′ = walker.sign′
    M = walker.FM[1]
    Bτ = walker.Bτ.B[1]

    for i in 1 : system.V
        σ[i] *= -1
        r = compute_Metropolis_ratio(system, walker, σ, M)
        qmc.saveRatio && push!(walker.tmp_r, r)
        u = qmc.useHeatbath ? r/(1+r) : r

        if rand() < u
            copyto!(weight, weight′)
            copyto!(sgn, sgn′)

            copyto!(Bl, Bτ)
        else
            # revert the change
            σ[i] *= -1
        end
    end

    return nothing
end

"""
    cluster_flip!(...)

    Metropolis test by flipping a cluster of sites
"""
function cluster_flip!(
    system::System, qmc::QMC, walker::Walker,
    σ::AbstractArray, Bl::AbstractMatrix
)
    weight = walker.weight
    weight′ = walker.weight′
    sgn = walker.sign
    sgn′ = walker.sign′
    M = walker.FM[1]
    Bτ = walker.Bτ.B[1]

    for i in qmc.cluster_list
        σ_flip = 2 * (rand(length(i)) .< 0.5) .- 1
        @views @. σ[i] *= σ_flip

        r = compute_Metropolis_ratio(system, walker, σ, M)
        qmc.saveRatio && push!(walker.tmp_r, r)
        u = qmc.useHeatbath ? r/(1+r) : r

        if rand() < u
            copyto!(weight, weight′)
            copyto!(sgn, sgn′)

            copyto!(Bl, Bτ)
        else
            # revert the change
            @views @. σ[i] *= σ_flip
        end
    end

    return nothing
end

"""
    update_cluster!_symmetric(...)

    Update the propagation matrices of size equaling to the stablization interval
"""
function update_cluster!_symmetric(
    system::System, qmc::QMC, walker::Walker{T, LDR{T,E}}, cidx::Int64;
    direction::Int = 1
) where {T,E}
    k = qmc.K_interval[cidx]
    
    ws = walker.ws
    F = walker.FM[1]
    Bl = walker.Bl.B
    Bc = walker.Bc.B

    L, _, R = F

    direction == 1 ? begin
            slice = collect(1:k)
            # compute R <- R * Bc in-place
            rmul!(R, Bc[cidx], ws.M)
        end :
        begin
            slice = collect(k:-1:1)
            # compute L <- Bc * L in-place
            lmul!(Bc[cidx], L, ws.M)
        end

    for i in slice
        σ = @view walker.auxfield[:, (cidx-1) * qmc.stab_interval + i]
        imagtime_propagator!(Bl[i], σ, system, tmpmat=ws.M)
        direction == 1 ? begin
            # compute R <- R * Bl⁻¹ in-place
            rmul_inv!(R, Bl[i], ws)
        end :
        begin
            # compute L <- Bl⁻¹ * L in-place
            lmul_inv!(Bl[i], L, ws)
        end

        qmc.useClusterUpdate ? 
            cluster_flip!(
                system, qmc, walker, σ, Bl[i]
            ) :
            local_flip!(
                system, qmc, walker, σ, Bl[i]
            )

        direction == 1 ? begin
            # compute L <- Bl * L in-place
            lmul!(Bl[i], L, ws.M)
        end :
        begin
            # compute R <- R * Bl in-place
            rmul!(R, Bl[i], ws.M)
        end
    end

    @views prod_cluster!(Bc[cidx], Bl[k:-1:1], ws.M)

    return nothing
end

"""
    update_cluster!_symmetric(...)

    Update the propagation matrices of size equaling to the stablization interval
"""
function update_cluster!_symmetric(
    system::System, qmc::QMC, walker::Walker{T, LDRLowRank{T,E}}, cidx::Int64;
    direction::Int = 1
) where {T,E}
    k = qmc.K_interval[cidx]
    
    ws = walker.ws
    M = walker.FM[1]
    Bl = walker.Bl.B
    Bc = walker.Bc.B

    L, _, R = M.F

    direction == 1 ? begin
            slice = collect(1:k)
            # compute R <- R * Bc in-place
            rmul!(R, Bc[cidx], ws.M)
        end :
        begin
            slice = collect(k:-1:1)
            # compute L <- Bc * L in-place
            lmul!(Bc[cidx], L, ws.M)
        end

    for i in slice
        σ = @view walker.auxfield[:, (cidx-1) * qmc.stab_interval + i]
        imagtime_propagator!(Bl[i], σ, system, tmpmat=ws.M)
        direction == 1 ? begin
            # compute R <- R * Bl⁻¹ in-place
            rmul_inv!(R, Bl[i], ws)
        end :
        begin
            # compute L <- Bl⁻¹ * L in-place
            lmul_inv!(Bl[i], L, ws)
        end

        qmc.useClusterUpdate ? 
            cluster_flip!(
                system, qmc, walker, σ, Bl[i]
            ) :
            local_flip!(
                system, qmc, walker, σ, Bl[i]
            )

        direction == 1 ? begin
            # compute L <- Bl * L in-place
            lmul!(Bl[i], L, ws.M)
        end :
        begin
            # compute R <- R * Bl in-place
            rmul!(R, Bl[i], ws.M)
        end
    end

    @views prod_cluster!(Bc[cidx], Bl[k:-1:1], ws.M)

    return nothing
end

"""
    sweep!_symmetric(...)
"""
function sweep!_symmetric(
    system::System, qmc::QMC, walker::Walker;
    direction::Int = 1
)
    K = qmc.K
    ws = walker.ws

    direction == 1 ? (
            tmpL = walker.FC.B;
            tmpR = walker.Fτ[1]
        ) : 
        (
            tmpL = walker.Fτ[1];
            tmpR = walker.FC.B
        )
    tmpM = walker.FM[1]
    Bc = walker.Bc.B

    # propagate from 0 to β
    direction == 1 && begin
        for cidx in 1:K
            mul!(tmpM, tmpR, tmpL[cidx], ws)
            update_cluster!_symmetric(
                system, qmc, walker, cidx, direction=1
            )

            copyto!(tmpL[cidx], tmpR)

            lmul!(Bc[cidx], tmpR, ws)
        end

        # save the propagation results
        copyto!(walker.F[1], tmpR)
        # then reset Fτ to unit matrix
        reset!(tmpR)

        return nothing 
    end

    # propagate from β to 0
    for cidx in K:-1:1
        mul!(tmpM, tmpR[cidx], tmpL, ws)
        update_cluster!_symmetric(
            system, qmc, walker, cidx, direction=2
        )

        copyto!(tmpR[cidx], tmpL)

        rmul!(tmpL, Bc[cidx], ws)
    end

    # save the propagation results
    copyto!(walker.F[1], tmpL)
    # then reset Fτ to unit matrix
    reset!(tmpL)

    return nothing
end

##################################################
##### Asymmetric Sweep for Spin HS Transform #####
##################################################
"""
    local_flip!(...)

    Metropolis test by flipping a single spin
"""
function local_flip!(
    system::System, qmc::QMC, walker::Walker,
    σ::AbstractArray, Bl::Vector{Tb}
) where {Tb<:AbstractMatrix}
    weight = walker.weight
    weight′ = walker.weight′
    sgn = walker.sign
    sgn′ = walker.sign′
    M = walker.FM
    Bτ = walker.Bτ.B

    for i in 1 : system.V
        σ[i] *= -1
        r = compute_Metropolis_ratio(system, walker, σ, M)
        qmc.saveRatio && push!(walker.tmp_r, r)
        u = qmc.useHeatbath ? r/(1+r) : r

        if rand() < u
            copyto!(weight, weight′)
            copyto!(sgn, sgn′)

            copyto!(Bl[1], Bτ[1])
            copyto!(Bl[2], Bτ[2])
        else
            # revert the change
            σ[i] *= -1
        end
    end

    return nothing
end

"""
    cluster_flip!(...)

    Metropolis test by flipping a cluster of sites
"""
function cluster_flip!(
    system::System, qmc::QMC, walker::Walker,
    σ::AbstractArray, Bl::Vector{Tb}
) where {Tb<:AbstractMatrix}
    weight = walker.weight
    weight′ = walker.weight′
    sgn = walker.sign
    sgn′ = walker.sign′
    M = walker.FM
    Bτ = walker.Bτ.B

    for i in qmc.cluster_list
        σ_flip = 2 * (rand(length(i)) .< 0.5) .- 1
        @views @. σ[i] *= σ_flip

        r = compute_Metropolis_ratio(system, walker, σ, M)
        qmc.saveRatio && push!(walker.tmp_r, r)
        u = qmc.useHeatbath ? r/(1+r) : r

        if rand() < u
            copyto!(weight, weight′)
            copyto!(sgn, sgn′)

            copyto!(Bl[1], Bτ[1])
            copyto!(Bl[2], Bτ[2])
        else
            # revert the change
            @views @. σ[i] *= σ_flip
        end
    end

    return nothing
end

"""
    update_cluster!_asymmetric(...)

    Update the propagation matrices of size equaling to the stablization interval
"""
function update_cluster!_asymmetric(
    system::System, qmc::QMC, walker::Walker{T, LDR{T,E}}, cidx::Int64;
    direction::Int = 1
) where {T,E}
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    ws = walker.ws
    F = walker.FM
    Bl = walker.Bl.B
    Bc = walker.Bc.B

    L₊, _, R₊ = F[1]
    L₋, _, R₋ = F[2]

    direction == 1 ? begin
            slice = collect(1:k)
            # compute R <- R * Bc in-place
            rmul!(R₊, Bc[cidx], ws.M)
            rmul!(R₋, Bc[K+cidx], ws.M)
        end :
        begin
            slice = collect(k:-1:1)
            # compute L <- Bc * L in-place
            lmul!(Bc[cidx], L₊, ws.M)
            lmul!(Bc[K+cidx], L₋, ws.M)
        end

    for i in slice
        σ = @view walker.auxfield[:, (cidx-1) * qmc.stab_interval + i]
        imagtime_propagator!(Bl[i], Bl[k+i], σ, system; tmpmat=ws.M)
        direction == 1 ? begin
            # compute R <- R * Bl⁻¹ in-place
            rmul_inv!(R₊, Bl[i], ws)
            rmul_inv!(R₋, Bl[k+i], ws)
        end :
        begin
            # compute L <- Bl⁻¹ * L in-place
            lmul_inv!(Bl[i], L₊, ws)
            lmul_inv!(Bl[k+i], L₋, ws)
        end

        qmc.useClusterUpdate ? 
            cluster_flip!(
                system, qmc, walker, σ, [Bl[i], Bl[k+i]]
            ) :
            local_flip!(
                system, qmc, walker, σ, [Bl[i], Bl[k+i]]
            )

        direction == 1 ? begin
            # compute L <- Bl * L in-place
            lmul!(Bl[i], L₊, ws.M)
            lmul!(Bl[k+i], L₋, ws.M)
        end :
        begin
            # compute R <- R * Bl in-place
            rmul!(R₊, Bl[i], ws.M)
            rmul!(R₋, Bl[k+i], ws.M)
        end
    end

    @views prod_cluster!(Bc[cidx], Bl[k:-1:1], ws.M)
    @views prod_cluster!(Bc[K+cidx], Bl[2*k:-1:k+1], ws.M)

    return nothing
end

"""
    update_cluster!_asymmetric(...)

    Update the propagation matrices of size equaling to the stablization interval
"""
function update_cluster!_asymmetric(
    system::System, qmc::QMC, walker::Walker{T, LDRLowRank{T,E}}, cidx::Int64;
    direction::Int = 1
) where {T,E}
    k = qmc.K_interval[cidx]
    K = qmc.K
    
    ws = walker.ws
    M = walker.FM
    Bl = walker.Bl.B
    Bc = walker.Bc.B

    L₊, _, R₊ = M[1].F
    L₋, _, R₋ = M[2].F

    direction == 1 ? begin
            slice = collect(1:k)
            # compute R <- R * Bc in-place
            rmul!(R₊, Bc[cidx], ws.M)
            rmul!(R₋, Bc[K+cidx], ws.M)
        end :
        begin
            slice = collect(k:-1:1)
            # compute L <- Bc * L in-place
            lmul!(Bc[cidx], L₊, ws.M)
            lmul!(Bc[K+cidx], L₋, ws.M)
        end

    for i in slice
        σ = @view walker.auxfield[:, (cidx-1) * qmc.stab_interval + i]
        imagtime_propagator!(Bl[i], Bl[k+i], σ, system, tmpmat=ws.M)
        direction == 1 ? begin
            # compute R <- R * Bl⁻¹ in-place
            rmul_inv!(R₊, Bl[i], ws)
            rmul_inv!(R₋, Bl[k+i], ws)
        end :
        begin
            # compute L <- Bl⁻¹ * L in-place
            lmul_inv!(Bl[i], L₊, ws)
            lmul_inv!(Bl[k+i], L₋, ws)
        end

        qmc.useClusterUpdate ? 
            cluster_flip!(
                system, qmc, walker, σ, [Bl[i], Bl[k+i]]
            ) :
            local_flip!(
                system, qmc, walker, σ, [Bl[i], Bl[k+i]]
            )

        direction == 1 ? begin
            # compute L <- Bl * L in-place
            lmul!(Bl[i], L₊, ws.M)
            lmul!(Bl[k+i], L₋, ws.M)
        end :
        begin
            # compute R <- R * Bl in-place
            rmul!(R₊, Bl[i], ws.M)
            rmul!(R₋, Bl[k+i], ws.M)
        end
    end

    @views prod_cluster!(Bc[cidx], Bl[k:-1:1], ws.M)
    @views prod_cluster!(Bc[K+cidx], Bl[2*k:-1:k+1], ws.M)

    return nothing
end

"""
    sweep!_asymmetric(...)
"""
function sweep!_asymmetric(
    system::System, qmc::QMC, walker::Walker;
    direction::Int = 1
)
    K = qmc.K
    ws = walker.ws

    direction == 1 ? (
            tmpL = walker.FC.B;
            tmpR = walker.Fτ
        ) : 
        (
            tmpL = walker.Fτ;
            tmpR = walker.FC.B
        )
    tmpM = walker.FM
    Bc = walker.Bc.B

    # propagate from 0 to β
    direction == 1 && begin
        for cidx in 1:K
            mul!(tmpM[1], tmpR[1], tmpL[cidx], ws)
            mul!(tmpM[2], tmpR[2], tmpL[K+cidx], ws)
            update_cluster!_asymmetric(
                system, qmc, walker, cidx, direction=1
            )

            copyto!(tmpL[cidx], tmpR[1])
            copyto!(tmpL[K+cidx], tmpR[2])

            lmul!(Bc[cidx], tmpR[1], ws)
            lmul!(Bc[K+cidx], tmpR[2], ws)
        end

        # save the propagation results
        copyto!(walker.F[1], tmpR[1])
        copyto!(walker.F[2], tmpR[2])
        # then reset Fτ to unit matrix
        reset!(tmpR[1])
        reset!(tmpR[2])

        return nothing 
    end

    # propagate from β to 0
    for cidx in K:-1:1
        mul!(tmpM[1], tmpR[cidx], tmpL[1], ws)
        mul!(tmpM[2], tmpR[cidx+K], tmpL[2], ws)
        update_cluster!_asymmetric(
            system, qmc, walker, cidx, direction=2
        )

        copyto!(tmpR[cidx], tmpL[1])
        copyto!(tmpR[cidx+K], tmpL[2])

        rmul!(tmpL[1], Bc[cidx], ws)
        rmul!(tmpL[2], Bc[K+cidx], ws)
    end

    # save the propagation results
    copyto!(walker.F[1], tmpL[1])
    copyto!(walker.F[2], tmpL[2])
    # then reset Fτ to unit matrix
    reset!(tmpL[1])
    reset!(tmpL[2])

    return nothing
end
