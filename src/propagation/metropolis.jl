"""
    Metropolis Sampling (MCMC)
"""

function calc_trail_mcmc(
    system::System, walker::Walker, σ::Vector{Int64}
)
    """
    Calculate the statistical weight when the (i,j)th field component is proposed
    Shared with Metropolis

    # Arguments
    σ -> auxiliary field at the jth time slice

    # Returns
    Z -> statistical weights of the configuration
    F = (U, D, T) -> matrix decompositions as the last site is updated
    """
    # generate Bi
    B = singlestep_matrix(σ, system)
    # calculate the decomposition of B_i * B_{i-1} *...* B_1 * B_T *...* B_T
    F = [QRCP_lmul(B[1], walker.F[1]), QRCP_lmul(B[2], walker.F[2])]
    # diagonalize the decomposition
    expβϵ = [eigvals(F[1]), eigvals(F[2])]
    # calculate the statistical weight (partition function)
    Z = [
        pf_projection(system.V, system.N[1], expβϵ[1], system.expiφ, false),
        pf_projection(system.V, system.N[2], expβϵ[2], system.expiφ, false)
    ]
    
    system.isReal && return real(Z), F
    return Z, F
end

function propagate!_mcmc(
    system::System, walker::Walker, 
    field_index::Tuple{Int64, Int64}, tmp::Vector{UDT{T}}
    ) where {T<:FloatType}
    """
    Propagation at the jth site of the ith time slice

    # Arguments
    field_index[1] -> time slice index (i)
    field_index[2] -> site index (j)
    """
    # propose a flip at the jth site of the ith time slice
    flip!(walker.auxfield, field_index[2], field_index[1])
    # calculate the corresponding weight (partition function)
    Z, F = calc_trail_mcmc(
        system, walker, walker.auxfield[:, field_index[1]]
    )
    # weight ratio between the new and the old configurations
    ratio = (Z[1] / walker.weight[1]) * (Z[2] / walker.weight[2])
    # heat-bath ratio
    accept = abs(ratio) / (1 + abs(ratio))

    if rand() < accept
        # accept the proposal, update weights
        walker.weight .= Z
        # update the temporal matrix decompositions
        update_matrices!(tmp[1], F[1])
        update_matrices!(tmp[2], F[2])
    else
        # reject the proposal, rollback the change
        flip!(walker.auxfield, field_index[2], field_index[1])
    end
end

function sweep!(
    system::System, qmc::QMC, 
    walker::Walker, tmp::Vector{UDT{T}}) where {T<:FloatType}
    """
    Sweep the walker over the entire space-time lattice
    """
    update_count = 0
    
    for l = 1 : system.L

        update_matrices!(tmp[1], walker.F[1])
        update_matrices!(tmp[2], walker.F[2])
        # move the walker to the next time slice
        move!_mcmc(walker, system, l)
        
        for i = 1 : system.V
            # propagate through sites
            propagate!_mcmc(
                system, walker, (l, i), tmp
            )
            if i == system.V
                # update the matrices at the last spatial site
                update_matrices!(walker.F[1], tmp[1])
                update_matrices!(walker.F[2], tmp[2])
            end
        end

        # periodically calibrate the walker
        update_count += 1
        if update_count == qmc.update_interval
            calibrate!(system, qmc, walker, l)
            update_count = 0
        end
    end
end
