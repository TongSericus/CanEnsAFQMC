"""
    Matrix Operations
"""

function QRCP_update(
    Q::AbstractMatrix, D::AbstractMatrix, T::AbstractMatrix,
    B::AbstractMatrix, direction::Char
)
    """
    Calculate Q', D', T' with Q'D'T' = QDT * B or B * QDT

    # Arguments
    direction => which side of QDT would B be multiplied to
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr!(BQD, Val(true))
        D = Diagonal(QRCP_BQD.R)
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R
        # D^-1 * R * P^T
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T, smart permutation
        T = temp * T

        return QRCP_BQD.Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr!(DTB, Val(true))
        Q = Q * QRCP_DTB.Q
        D = Diagonal(QRCP_DTB.R)
        # D^-1 * R
        temp = inv(D) * QRCP_DTB.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_DTB.p] = temp[:, :]
        T = temp

        return Q, D, T

    else
        @error "direction can only be 'L' or 'R'"
    end

end

##### Low-rank Update #####
function QRCP_update_lowrank(
    Q::AbstractMatrix, D::AbstractMatrix, T::AbstractMatrix,
    B::AbstractMatrix, direction::Char,
    n::Int64, ξ::Float64
)
    """ 
    Calculate Q', D', T' with Q'D'T' = QDT * B or B * QDT

    In-place operations on Q, D and T

    # Arguments
    direction -> which side of QDT would B be multiplied to
    n -> filling (starting point for truncation from above)
    ξ -> truncation threshold
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr!(BQD, Val(true))
        d = diag(QRCP_BQD.R)
        nL = sum(abs.(d[n + 1 : end] / d[n]) .> ξ)
        Q = QRCP_BQD.Q[:, 1 : n + nL]
        D = Diagonal(d[1 : n + nL])
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R[1 : n + nL, :]
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T
        T = temp * T

        return Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr!(DTB, Val(true))
        d = diag(QRCP_DTB.R)
        nR = sum(abs.(d[n + 1 : end] / d[n]) .> ξ)
        Q = Q * QRCP_DTB.Q
        D .= Diagonal(QRCP_DTB.R)
        # D^-1 * R
        temp = inv(D) * QRCP_DTB.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_DTB.p] = temp[:, :]
        T .= temp

        return Q, D, T

    else
        ArgumentError("direction can only be 'L' or 'R'")
    end

end
