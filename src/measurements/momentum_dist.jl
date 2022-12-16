""" 
    Measure momentum distribution
"""
function measure_MomentumDist(
    DFTmats::Vector{Matrix{ComplexF64}},
    Dup::AbstractMatrix{T}, Ddn::AbstractMatrix{T};
    isReal::Bool = true
) where {T<:Number}
    V = size(Dup, 1)

    nk_up = zeros(ComplexF64, length(DFTmats))
    nk_dn = zeros(ComplexF64, length(DFTmats))
    for (i, DFTmat) in enumerate(DFTmats)
        nk_up[i] = sum(DFTmat .* Dup) / V
        nk_dn[i] = sum(DFTmat .* Ddn) / V
    end

    nk = (nk_up + nk_dn) / 2
    
    isReal && return real(nk)
    return nk
end
