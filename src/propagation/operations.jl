"""
    All operations in the propagation
"""

function full_propagation(
    auxfield::Matrix{Int64}, system::System, qmc::QMC
)
    """
    Propagate the entire space-time auxiliary field
    using column-pivoted QR (QRCP) decomposition

    # Arguments
    auxfield -> the entire Ns×L auxiliary field

    # Returns
    F = (Q, D, T) -> matrix decompositions with QDT = BL...B1
    """
    Ns = system.V
    F = [
        UDT(Matrix(1.0I, Ns, Ns), ones(Float64, Ns), Matrix(1.0I, Ns, Ns)),
        UDT(Matrix(1.0I, Ns, Ns), ones(Float64, Ns), Matrix(1.0I, Ns, Ns))
    ]

    for i = 1 : div(system.L, qmc.stab_interval)

        mat_product_up = Matrix(1.0I, Ns, Ns)
        mat_product_dn = Matrix(1.0I, Ns, Ns)

        for j = 1 : qmc.stab_interval
            σ = auxfield[:, (i - 1) * qmc.stab_interval + j]
            B = singlestep_matrix(σ, system)
            mat_product_up = B[1] * mat_product_up
            mat_product_dn = B[2] * mat_product_dn
        end

        F[1] = QRCP_lmul(mat_product_up, F[1])
        F[2] = QRCP_lmul(mat_product_dn, F[2])

    end

    return F

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
    F = QRCP_rmul(walker.F[1], inv(Bl[1]))
    update_matrices!(walker.F[1], F)
    F = QRCP_rmul(walker.F[2], inv(Bl[2]))
    update_matrices!(walker.F[2], F)
end

function move!_constrained(walker::Walker, system::System)
    """
    Move the walker to the next time slice,
    i.e. calculate B = Bl...B1 * BT...BT * (BT)^-1
    """
    F = QRCP_rmul(walker.F[1], system.BT_inv)
    update_matrices!(walker.F[1], F)
    F = QRCP_rmul(walker.F[2], system.BT_inv)
    update_matrices!(walker.F[2], F)
end

function update_matrices!(F0::UDT, Ft::UDT)
    F0.U .= Ft.U
    F0.D .= Ft.D
    F0.T .= Ft.T
end

function calibrate!(system::System, qmc::QMC, walker::Walker, time_index::Int64)
    """
    Q, D, T matrices need to be recalculated periodically

    # Arguments
    time_index -> time slice index
    """
    shifted_field = circshift(walker.auxfield, (0, -time_index))
    F = full_propagation(shifted_field, system, qmc)
    update_matrices!(walker.F[1], F[1])
    update_matrices!(walker.F[2], F[2])
end
