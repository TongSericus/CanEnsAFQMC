"""
    All operations in the propagation
"""

function full_propagation(
    σfield::Array{Int64,2}, system::System, qmc::QMC
)
    """
    Propagate the entire space-time auxiliary field
    using column-pivoted QR (QRCP) decomposition

    # Arguments
    σfield -> the entire Ns×L auxiliary field

    # Returns
    Q, D, T -> matrix decompositions with QDT = BL...B1
    """
    
    Ns = system.V

    Q = [Matrix(1.0I, Ns, Ns), Matrix(1.0I, Ns, Ns)]
    D = [Diagonal(ones(Ns)), Diagonal(ones(Ns))]
    T = [Matrix(1.0I, Ns, Ns), Matrix(1.0I, Ns, Ns)]

    for i = 1 : div(system.L, qmc.stab_interval)

        mat_product_up = Matrix(1.0I, Ns, Ns)
        mat_product_dn = Matrix(1.0I, Ns, Ns)

        for j = 1 : qmc.stab_interval
            σ = σfield[:, (i - 1) * qmc.stab_interval + j]
            B = singlestep_matrix(σ, system)
            mat_product_up = B[1] * mat_product_up
            mat_product_dn = B[2] * mat_product_dn
        end

        Q[1], D[1], T[1] = QRCP_update(Q[1], D[1], T[1], mat_product_up, 'L')
        Q[2], D[2], T[2] = QRCP_update(Q[2], D[2], T[2], mat_product_dn, 'L')

    end

    return Q, D, T

end

function full_propagation_lowrank(
    σfield::Array{Int64,2}, system::System, qmc::QMC
)
    """
    Propagate the entire space-time auxiliary field
    using column-pivoted QR (QRCP) decomposition with
    low-rank truncation

    # Arguments
    σfield -> the entire Ns*L auxiliary field

    # Returns
    Q, D, T -> matrix decompositions with QDT = BL...B1
    """
    
    Ns = system.V

    Q = [Matrix(1.0I, Ns, Ns), Matrix(1.0I, Ns, Ns)]
    D = [Diagonal(ones(Ns)), Diagonal(ones(Ns))]
    T = [Matrix(1.0I, Ns, Ns), Matrix(1.0I, Ns, Ns)]

    for i = 1 : div(system.L, qmc.stab_interval)

        mat_product_up = Matrix(1.0I, Ns, Ns)
        mat_product_dn = Matrix(1.0I, Ns, Ns)

        for j = 1 : qmc.stab_interval
            σ = σfield[:, (i - 1) * qmc.stab_interval + j]
            B = singlestep_matrix(σ, system)
            mat_product_up = B[1] * mat_product_up
            mat_product_dn = B[2] * mat_product_dn
        end

        Q[1], D[1], T[1] = QRCP_update_lowrank(
            Q[1], D[1], T[1], mat_product_up,
            'L', system.N[1], qmc.lrThreshold
        )
        Q[2], D[2], T[2] = QRCP_update_lowrank(
            Q[2], D[2], T[2], mat_product_dn, 
            'L', system.N[2], qmc.lrThreshold
        )

    end

    return Q, D, T
    
end

@inline function flip!(auxfield::Array{Int64,2}, i::Int64, j::Int64)
    auxfield[i, j] = -auxfield[i, j]
end

function move!_mcmc(walker::Walker, system::System, time_index::Int64)
    """
    Move the walker to the next time slice,
    i.e. calculate B = B_{l-1}...B_1 * B_L...B_l * B_l^-1
    """
    Bl = singlestep_matrix(walker.auxfield[:, time_index], system)
    walker.Q[1], walker.D[1], walker.T[1] = QRCP_update(
        walker.Q[1], walker.D[1], walker.T[1], inv(Bl[1]), 'R'
    )
    walker.Q[2], walker.D[2], walker.T[2] = QRCP_update(
        walker.Q[2], walker.D[2], walker.T[2], inv(Bl[2]), 'R'
    )
end

function move!_constrained(walker::Walker, system::System)
    """
    Move the walker to the next time slice,
    i.e. calculate B = Bl...B1 * BT...BT * (BT)^-1
    """
    walker.Q[1], walker.D[1], walker.T[1] = QRCP_update(
        walker.Q[1], walker.D[1], walker.T[1], system.BT_inv, 'R'
    )
    walker.Q[2], walker.D[2], walker.T[2] = QRCP_update(
        walker.Q[2], walker.D[2], walker.T[2], system.BT_inv, 'R'
    )
end

function update_matrices!(
    Q0::Vector{T1}, D0::Vector{T2}, T0::Vector{T1},
    Qt::Vector{T3}, Dt::Vector{T2}, Tt::Vector{T1}
) where {T1<:AbstractMatrix, T2<:AbstractMatrix, T3<:AbstractMatrix}
    
    Q0[1] = Qt[1]
    Q0[2] = Qt[2]
    D0[1] = Dt[1]
    D0[2] = Dt[2]
    T0[1] = Tt[1]
    T0[2] = Tt[2]
end

function copy_matrices!(walker::Walker, temp::MatDecomp, REV::Bool)

    if REV
        walker.Q[1] .= copy(temp.Q[1])
        walker.Q[2] .= copy(temp.Q[2])

        walker.D[1] .= copy(temp.D[1])
        walker.D[2] .= copy(temp.D[2])

        walker.T[1] .= copy(temp.T[1])
        walker.T[2] .= copy(temp.T[2])
    else
        temp.Q[1] .= copy(walker.Q[1])
        temp.Q[2] .= copy(walker.Q[2])

        temp.D[1] .= copy(walker.D[1])
        temp.D[2] .= copy(walker.D[2])

        temp.T[1] .= copy(walker.T[1])
        temp.T[2] .= copy(walker.T[2])
    end
end

function calibrate!(system::System, qmc::QMC, walker::Walker, time_index::Int64)
    """
    Q, D, T matrices need to be recalculated periodically

    # Arguments
    time_index -> time slice index
    """
    shifted_field = circshift(walker.auxfield, (0, -time_index))
    Q, D, T = full_propagation(shifted_field, system, qmc)
    update_matrices!(walker.Q, walker.D, walker.T, Q, D, T)
end
