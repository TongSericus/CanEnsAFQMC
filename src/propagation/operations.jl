###################################
##### Basic operations in QMC #####
###################################

"""
    compute_pf(...)
"""
function compute_pf(
    F::LDR{T, E}, N::Int64, ws::LDRWorkspace{T,E};
    Ns = length(F.d),
    P = zeros(ComplexF64, Ns+1, Ns)
) where {T, E}
    λ = eigvals(F, ws)
    return compute_pf_recursion(λ, N, P=P)
end

"""
    compute_pf(...)
"""
function compute_pf(
    M::LDRLowRank{T, E}, ws::LDRWorkspace{T,E};
    Ns = length(M.F.d),
    P = zeros(ComplexF64, Ns+1, Ns)
) where {T, E}
    λ = eigvals(M, ws)
    return compute_pf_recursion(λ, M.N, P=P)
end

"""
    compute_Metropolis_ratio(...)
"""
function compute_Metropolis_ratio(
    system::System, walker::Walker, σ::AbstractArray{Int}, 
    F::LDR{T,E}
) where {T,E}
    weight = walker.weight
    weight′ = walker.weight′
    sgn′ = walker.sign′

    ws = walker.ws
    Bτ = walker.Bτ.B[1]

    imagtime_propagator!(Bτ, σ, system, tmpmat=ws.M)

    L = ws.M′
    mul!(L, Bτ, F.L)
    M = LDR(L, F.d, F.R)
    weight′[1], sgn′[1] = compute_pf(M, system.N[1], ws, P=walker.P)
    
    weight′[2] = weight′[1]
    sgn′[2] = conj(sgn′[1])

    r = exp(sum(weight′) - sum(weight))

    return r
end

"""
    compute_Metropolis_ratio(...)
"""
function compute_Metropolis_ratio(
    system::System, walker::Walker, σ::AbstractArray{Int}, M::LDRLowRank{T,E}
) where {T,E}
    # perform low-rank truncation
    ws = walker.ws
    lowrank_truncation!(M, ws=ws)

    weight = walker.weight
    weight′ = walker.weight′
    sgn′ = walker.sign′

    Bτ = walker.Bτ.B[1]
    F = M.F

    imagtime_propagator!(Bτ, σ, system, tmpmat=ws.M)

    L = ws.M′
    @views mul!(L[:, M.t[]], Bτ, F.L[:, M.t[]])
    M = LDRLowRank(LDR(L, F.d, F.R), M.N, M.ϵ, M.t)
    weight′[1], sgn′[1] = compute_pf(M, ws, P=walker.P)
    
    weight′[2] = weight′[1]
    sgn′[2] = conj(sgn′[1])

    r = exp(sum(weight′) - sum(weight))

    return r
end

"""
    compute_Metropolis_ratio(...)
"""
function compute_Metropolis_ratio(
    system::System, walker::Walker, σ::AbstractArray{Int}, 
    F::Vector{LDR{T,E}}
) where {T,E}
    weight = walker.weight
    weight′ = walker.weight′
    sgn′ = walker.sign′

    ws = walker.ws
    Bτ = walker.Bτ.B
    F₊, F₋ = F

    imagtime_propagator!(Bτ[1], Bτ[2], σ, system, tmpmat=ws.M)

    L₊ = ws.M′
    mul!(L₊, Bτ[1], F₊.L)
    M₊ = LDR(L₊, F₊.d, F₊.R)
    weight′[1], sgn′[1] = compute_pf(M₊, system.N[1], ws, P=walker.P)
    
    L₋ = ws.M″
    mul!(L₋, Bτ[2], F₋.L)
    M₋ = LDR(L₋, F₋.d, F₋.R)
    weight′[2], sgn′[2] = compute_pf(M₋, system.N[2], ws, P=walker.P)

    r = exp(sum(weight′) - sum(weight))

    return r
end

"""
    compute_Metropolis_ratio(...)
"""
function compute_Metropolis_ratio(
    system::System, walker::Walker, σ::AbstractArray{Int}, M::Vector{LDRLowRank{T,E}}
) where {T,E}
    # perform low-rank truncation
    ws = walker.ws
    lowrank_truncation!(M[1], ws=ws)
    lowrank_truncation!(M[2], ws=ws)

    weight = walker.weight
    weight′ = walker.weight′
    sgn′ = walker.sign′

    Bτ = walker.Bτ.B
    F₊ = M[1].F
    F₋ = M[2].F

    imagtime_propagator!(Bτ[1], Bτ[2], σ, system, tmpmat=ws.M)

    L₊ = ws.M′
    @views mul!(L₊[:, M[1].t[]], Bτ[1], F₊.L[:, M[1].t[]])
    M₊ = LDRLowRank(LDR(L₊, F₊.d, F₊.R), M[1].N, M[1].ϵ, M[1].t)
    weight′[1], sgn′[1] = compute_pf(M₊, ws, P=walker.P)
    
    L₋ = ws.M″
    @views mul!(L₋[:, M[2].t[]], Bτ[2], F₋.L[:, M[2].t[]])
    M₋ = LDRLowRank(LDR(L₋, F₋.d, F₋.R), M[2].N, M[2].ϵ, M[2].t)
    weight′[2], sgn′[2] = compute_pf(M₋, ws, P=walker.P)

    r = exp(sum(weight′) - sum(weight))

    return r
end

"""
    prod_cluster!(B::AbstractMatrix, Bl::AbstractArray{T}, C::AbstractMatrix)

    In-place calculation of prod(Bl) and overwrite the result to B, with an auxiliary matrix C
"""
function prod_cluster!(B::AbstractMatrix, Bl::AbstractArray{T}, C::AbstractMatrix) where {T<:AbstractMatrix}
    size(B) == size(Bl[1]) == size(C) || throw(BoundsError())
    k = length(Bl)
    k == 1 && (copyto!(B, Bl[1]); return nothing)
    k == 2 && (mul!(B, Bl[1], Bl[2]); return nothing)

    mul!(C, Bl[1], Bl[2])
    @inbounds for i in 3:k
        mul!(B, C, Bl[i])
        copyto!(C, B)
    end

    return nothing
end

############################################
##### Full Imaginary-time Propagations #####
############################################
"""
    build_propagator(auxfield, system, qmc, ws)

    Propagate over the full space-time lattice given the auxiliary field configuration
"""
function build_propagator(
    auxfield::AbstractMatrix{Int64}, system::System, qmc::QMC, ws::LDRWorkspace{T,E}; 
    isReverse::Bool = true, K = qmc.K, K_interval = qmc.K_interval
) where {T, E}
    V = system.V
    si = qmc.stab_interval

    # initialize partial matrix products
    Tb = eltype(system.auxfield)
    B = [Matrix{Tb}(I, V, V), Matrix{Tb}(I, V, V)]
    MatProd = Cluster(V, 2 * K, T = Tb)
    F = ldrs(B[1], 2)
    FC = Cluster(B = ldrs(B[1], 2 * K))

    Bm = MatProd.B
    Bf = FC.B

    isReverse && begin
        for i in K:-1:1
            for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat=ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        # save all partial products
        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        rmul!(F[1], Bm[i], ws)
        rmul!(F[2], Bm[K + i], ws)
    end

        return F, MatProd, FC
    end

    for i in 1:K
        for j = 1 : K_interval[i]
            @views σ = auxfield[:, (i - 1) * si + j]
            imagtime_propagator!(B[1], B[2], σ, system, tmpmat=ws.M)
            Bm[i] = B[1] * Bm[i]            # spin-up
            Bm[K + i] = B[2] * Bm[K + i]    # spin-down
        end

        copyto!(Bf[i], F[1])
        copyto!(Bf[K + i], F[2])

        lmul!(Bm[i], F[1], ws)
        lmul!(Bm[K + i], F[2], ws)
    end

    return F, MatProd, FC
end

"""
    build_propagator!(Fc, MatProd, ws)

    Propagate over the full space-time lattice given the matrix clusters
"""
function build_propagator!(
    Fc::Vector{Fact}, MatProd::Cluster{C}, ws::LDRWorkspace{T,E};
    K = div(length(MatProd.B), 2),
    isReverse::Bool = true, isSymmetric::Bool = false
) where {Fact, C, T, E}
    V = size(MatProd.B[1])
    i = eltype(ws.M) <: Real ? 1.0 : 1.0+0.0im
    F = ldrs(Matrix(i*I, V), 2)

    Bm = MatProd.B

    isReverse && begin 
        for n in K:-1:1
            copyto!(Fc[n], F[1])
            isSymmetric || copyto!(Fc[K + n], F[2])

            rmul!(F[1], Bm[n], ws)
            isSymmetric || rmul!(F[2], Bm[K + n], ws)
        end

        return F
    end

    for n in 1:K
        copyto!(Fc[n], F[1])
        isSymmetric || copyto!(Fc[K + n], F[2])

        lmul!(Bm[n], F[1], ws)
        isSymmetric || lmul!(Bm[K + n], F[2], ws)
    end
    
    return F
end
