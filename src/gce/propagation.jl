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
