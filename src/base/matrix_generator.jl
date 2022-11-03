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
    """
    Hubbard HS field matrix generator
    """
    pfield = isone.(σ)
    mfield = isone.(-σ)
    
    afmat_up = pfield * auxfield[1,1] .+ mfield * auxfield[2,1]
    afmat_dn = pfield * auxfield[1,2] .+ mfield * auxfield[2,2]

    return afmat_up, afmat_dn
end

function singlestep_matrix(
    σ::AbstractArray{Int64}, system::Hubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter
)
    """
    Compute B = exp(-ΔτK/2) * exp(-ΔτV(σ)) * exp(-ΔτK/2)
    """
    afmat_up, afmat_dn = auxfield_matrix_hubbard(σ, system.auxfield)

    if useFirstOrderTrotter
        B = [system.Bkf * Diagonal(afmat_up), system.Bkf * Diagonal(afmat_dn)]
    else
        B = [system.Bk * Diagonal(afmat_up) * system.Bk, system.Bk * Diagonal(afmat_dn) * system.Bk]
    end
    
    return B
end

function singlestep_matrix!(
    Bup::AbstractMatrix{T}, Bdn::AbstractMatrix{T}, σ::AbstractArray{Int64}, system::Hubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(Bup)
) where {T<:Number}
    """
    Compute B = exp(-ΔτK/2) * exp(-ΔτV(σ)) * exp(-ΔτK/2) in-place
    """
    afmat_up, afmat_dn = auxfield_matrix_hubbard(σ, system.auxfield)

    if useFirstOrderTrotter
        mul!(Bup, system.Bkf, Diagonal(afmat_up))
        mul!(Bdn, system.Bkf, Diagonal(afmat_dn))
    else
        mul!(tmpmat, Diagonal(afmat_up), system.Bk)
        mul!(Bup, system.Bk, tmpmat)
        
        mul!(tmpmat, Diagonal(afmat_dn), system.Bk)
        mul!(Bdn, system.Bk, tmpmat)
    end

    return nothing
end
