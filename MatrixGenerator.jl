"""
    All Matrix Operations Involved in the Simulation
"""

function kinetic_matrix_hubbard1D(Ns::Int64, t::Float64)

    kinetic_matrix = zeros(Ns, Ns)

    for i =  1 : Ns
        for j = 1 : Ns
            if abs(i - j) == 1 || abs(i - j) == Ns - 1
                kinetic_matrix[i, j] = -t
            end
        end
    end

    return kinetic_matrix

end

function kinetic_matrix_hubbard2D(NsX::Int64, NsY::Int64, t::Float64)

    kinetic_matrix = zeros(NsX * NsY, NsX * NsY)

    for i = 1 : NsX
        for j = 1 : NsY
            # indices of nearest neighbours (nn) of (i, j)
            nn_up = mod(j, NsY) + 1
            nn_dn = mod(j - 2, NsY) + 1 
            nn_lf = mod(i, NsX) + 1
            nn_rg = mod(i - 2, NsX) + 1 

            # parse indices into the kinetic matrix
            kinetic_matrix[(j - 1) * NsX + i, (nn_up - 1) * NsX + i] = -t
            kinetic_matrix[(j - 1) * NsX + i, (nn_dn - 1) * NsX + i] = -t
            kinetic_matrix[(j - 1) * NsX + i, (j - 1) * NsX + nn_lf] = -t
            kinetic_matrix[(j - 1) * NsX + i, (j - 1) * NsX + nn_rg] = -t
        end
    end

    return kinetic_matrix
    
end

function auxfield_matrix_hubbard(σ::Array{Int64,1}, system::System)

    nullfield = sum(iszero.(σ))
    auxfield_up = Vector{Float64}()
    auxfield_dn = Vector{Float64}()
    for σi in σ[1 : length(σ) - nullfield]
        push!(auxfield_up, system.auxfield[1][σi])
        push!(auxfield_dn, system.auxfield[2][σi])
    end
    auxfield_up = vcat(auxfield_up, ones(nullfield))
    auxfield_dn = vcat(auxfield_dn, ones(nullfield))

    return Diagonal(auxfield_up), Diagonal(auxfield_dn)

end

function singlestep_matrix(σ::Array{Int64,1}, system::System)
    """
    Calculate B = Bk/2 * Bv * Bk/2
    """

    if sum(iszero.(σ)) == length(σ)
        # Currently the trial propagator does not distinguish spins
        B = (system.BT, system.BT)
    else
        auxfield_matrix = auxfield_matrix_hubbard(σ, system)
        B = (system.Bk * auxfield_matrix[1] * system.Bk, system.Bk * auxfield_matrix[2] * system.Bk)
    end

end

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

        QRCP_update!(Q[1], D[1], T[1], mat_product_up, 'L')
        QRCP_update!(Q[2], D[2], T[2], mat_product_dn, 'L')

    end

    return Q, D, T

end

function QRCP_update!(
    Q::T1, D::T2, T::T1,
    B::T1, direction::Char
    ) where {T1<:MatrixType, T2<:MatrixType}
    """ 
    Calculate Q', D', T' with Q'D'T' = QDT * B or B * QDT

    In-place operations on Q, D and T

    # Arguments
    direction => which side of QDT would B be multiplied to
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr(BQD, Val(true))
        Q .= QRCP_BQD.Q * I
        D .= Diagonal(QRCP_BQD.R)
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T
        T .= temp * T

        return Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr(DTB, Val(true))
        Q .= Q * QRCP_DTB.Q
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

function QRCP_update(
    Q::T1, D::T2, T::T1,
    B::T1, direction::Char
    ) where {T1<:MatrixType, T2<:MatrixType}
    """
    QRCP update without overwriting
    """

    # Q'D'T' = B * QDT
    if direction == 'L'
        BQD = (B * Q) * D
        # column-pivoted QR (QRCP) decomposition
        QRCP_BQD = LinearAlgebra.qr(BQD, Val(true))
        Q = QRCP_BQD.Q * I
        D = Diagonal(QRCP_BQD.R)
        # D^-1 * R
        temp = inv(D) * QRCP_BQD.R
        # D^-1 * R * P^T
        temp[:, QRCP_BQD.p] = temp[:, :]
        # D^-1 * R * P^T * T, smart permutation
        T = temp * T

        return Q, D, T

    # Q'D'T' = QDT * B
    elseif direction == 'R'
        DTB = D * (T * B)
        # column-pivoted QR (QRCP) decomposition
        QRCP_DTB = LinearAlgebra.qr(DTB, Val(true))
        Q = Q * QRCP_DTB.Q
        D = Diagonal(QRCP_DTB.R)
        # D^-1 * R
        temp = inv(D) * QRCP_DTB.R
        # D^-1 * R * P^T, smart permutation
        temp[:, QRCP_DTB.p] = temp[:, :]
        T = temp

        return Q, D, T

    else
        ArgumentError("direction can only be 'L' or 'R'")
    end

end

##### Low-rank Update #####
function QRCP_update_lowrank(
    Q::T1, D::T2, T::T1,
    B::T1, direction::Char,
    n::Int64, ξ::Float64
    ) where {T1<:MatrixType, T2<:MatrixType}
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
        QRCP_BQD = LinearAlgebra.qr(BQD, Val(true))
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
        QRCP_DTB = LinearAlgebra.qr(DTB, Val(true))
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

function full_propagation_lowrank(
    σfield::Array{Int64,2}, system::System, qmc::QMC
    )
    """
    Propagate the entire space-time auxiliary field
    using column-pivoted QR (QRCP) decomposition with
    low-rank truncation

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

        Q[1], D[1], T[1] = QRCP_update_lowrank(
            Q[1], D[1], T[1], mat_product_up,
            'L', system.N[1], qmc.lowrank_threshold
        )
        Q[2], D[2], T[2] = QRCP_update_lowrank(
            Q[2], D[2], T[2], mat_product_dn, 
            'L', system.N[2], qmc.lowrank_threshold
        )

    end

    return Q, D, T

end