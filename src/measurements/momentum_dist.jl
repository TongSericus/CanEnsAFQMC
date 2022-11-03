""" 
    Measure momentum distribution
"""
function generate_rmat(system::System)
    V = system.V
    # 1D 
    if system.Ns[2] == 1
        rmat = zeros(Int64, V, V)
        for r1 = 1 : V
            for r2 = r1 : V
                rmat[r1, r2] = r1 - r2
                rmat[r2, r1] = r2 - r1
            end
        end

        return rmat
    end
end

function generate_DFTmats(system::System; rmat = generate_rmat(system))
    V = system.V
    DFTmats = Matrix{ComplexF64}[]
    # 1D 
    if system.Ns[2] == 1
        kpoints = div(V, 2)
        kpath = [i*π/kpoints for i in -kpoints:kpoints-1]

        for k in kpath
            DFTmat = similar(rmat, ComplexF64)
            for (i, r) in enumerate(rmat)
                DFTmat[i] = exp(im * dot(k, r))
            end
            push!(DFTmats, DFTmat)
        end
    end

    return DFTmats
end

function measure_nk(
    DFTmats::Vector{Matrix{ComplexF64}},
    G_up::AbstractMatrix{T}, G_dn::AbstractMatrix{T}
) where {T<:Number}
    V = size(G_up)[1]

    nk_up = zeros(ComplexF64, length(DFTmats))
    nk_dn = zeros(ComplexF64, length(DFTmats))
    for (i, DFTmat) in enumerate(DFTmats)
        nk_up[i] = sum(DFTmat .* G_up) / V
        nk_dn[i] = sum(DFTmat .* G_dn) / V
    end
    
    return nk_up, nk_dn
end
