"""
    Generate matrices for constructing the propagator
"""

function kinetic_matrix_hubbard1D(Ns::Int64, t::Float64)

    T = zeros(Ns, Ns)

    for i =  1 : Ns
        # indices of nearest neighbours (nn) of i
        nn_lf = mod(i, Ns) + 1
        nn_rg = mod(i - 2, Ns) + 1

        T[i, nn_lf] = -t
        T[i, nn_rg] = -t
    end

    return T
end

decode_basis(i::Int64, NsX::Int64) = [(i - 1) % NsX + 1, div(i - 1, NsX) + 1]
encode_basis(i::Int64, j::Int64, NsX::Int64) = i + (j - 1) * NsX

function kinetic_matrix_hubbard2D(NsX::Int64, NsY::Int64, t::Float64)
    """
    Cartesian lattice coordinates:
    (1,3) (2,3) (3,3)       7 8 9
    (1,2) (2,2) (3,2)  ->   4 5 6
    (1,1) (2,1) (3,1)       1 2 3
    """

    Ns = NsX * NsY
    T = zeros(Ns, Ns)

    for i = 1 : Ns
        ix, iy = decode_basis(i, NsX)
        # indices of nearest neighbours (nn) of (i, j)
        nn_up = mod(iy, NsY) + 1
        nn_dn = mod(iy - 2, NsY) + 1
        nn_lf = mod(ix, NsX) + 1
        nn_rg = mod(ix - 2, NsX) + 1

        T[i, encode_basis(ix, nn_up, NsX)] = -t
        T[i, encode_basis(ix, nn_dn, NsX)] = -t
        T[i, encode_basis(nn_lf, iy, NsX)] = -t
        T[i, encode_basis(nn_rg, iy, NsX)] = -t
    end

    return T
end

function auxfield_matrix_hubbard(σ::AbstractArray{Int64}, auxfield::Matrix{Float64})
    plusfield = isone.(σ)
    minusfield = isone.(-σ)
    
    afmat_up = plusfield * auxfield[1,1] .+ 
        minusfield * auxfield[2,1]
    afmat_dn = plusfield * auxfield[1,2] .+ 
        minusfield * auxfield[2,2]

    return afmat_up, afmat_dn
end

function singlestep_matrix(σ::AbstractArray{Int64}, system::System)
    """
    Compute B = Bk/2 * Bv * Bk/2
    """
    afmat = auxfield_matrix_hubbard(σ, system.auxfield)
    B = [system.Bk * Diagonal(afmat[1]) * system.Bk, system.Bk * Diagonal(afmat[2]) * system.Bk]
end

function singlestep_matrix!(
    B1::AbstractMatrix{T}, B2::AbstractMatrix{T}, σ::AbstractArray{Int64}, system::System
) where {T<:FloatType}
    """
    Compute B = Bk/2 * Bv * Bk/2
    """
    afmat = auxfield_matrix_hubbard(σ, system.auxfield)
    mul!(B1, system.Bk, Diagonal(afmat[1]) * system.Bk)
    mul!(B2, system.Bk, Diagonal(afmat[2]) * system.Bk)
end
