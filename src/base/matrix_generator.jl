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

function auxfield_matrix_hubbard(σ::Vector{Int64}, auxfield::Vector{Vector{Float64}})
    nullfield = iszero.(σ)
    plusfield = isone.(σ)
    minusfield = isone.(-σ)
    
    afmat_up = plusfield * auxfield[1][1] .+ 
        minusfield * auxfield[2][1] .+
        nullfield
    afmat_dn = plusfield * auxfield[1][2] .+ 
        minusfield * auxfield[2][2] .+
        nullfield

    return Diagonal(afmat_up), Diagonal(afmat_dn)
end

function singlestep_matrix(σ::Vector{Int64}, system::System)
    """
    Compute B = Bk/2 * Bv * Bk/2
    """
    # Currently the trial propagator does not distinguish spins
    sum(iszero.(σ)) == length(σ) && (B = [system.BT, system.BT])

    afmat = auxfield_matrix_hubbard(σ, system.auxfield)
    B = [system.Bk * afmat[1] * system.Bk, system.Bk * afmat[2] * system.Bk]
end

function generate_rmat(system::System)
    """
    Compute all r1 - r2 that would be used in the momentum distribution calculations
    """
    # 1D system
    if system.Ns[2] == 1
        rmat = zeros(Int64, system.V, system.V)
        for r1 = 1 : system.Ns[1]
            for r2 = r1 : system.Ns[1]
                rmat[r1, r2] = r1 - r2
                rmat[r2, r1] = -rmat[r1, r2]
            end
        end
        return rmat
    end

    # 2D system
    rmat = reshape([[0, 0] for _ = 1 : system.V ^ 2], (system.V, system.V))
    for r1 = 1 : system.Ns[1]^2
        for r2 = r1 : system.Ns[1]^2
            r1x, r1y = decode_basis(r1, Ns[1])
            r2x, r2y = decode_basis(r2, Ns[1])
            rmat[r1, r2] = [r1x - r2x, r1y - r2y]
            rmat[r2, r1] = -rmat[r1, r2]
        end
    end
    return rmat
end

function generate_DFTmat(
    kpath::Vector{Vector{Float64}}, rmat::Matrix{Vector{T}}
) where {T <: Real}
    """
    Generate the discrete Fourier transform matrices
    """
    DFTmats = Vector{Matrix{ComplexF64}}()
    for k in kpath
        DFTmat = similar(rmat, ComplexF64)
        fill!(DFTmat, 0.0)
        for (i, r) in enumerate(rmat)
            DFTmat[i] = exp(im * dot(k, r))
        end
        push!(DFTmats, DFTmat)
    end

    return DFTmats
end