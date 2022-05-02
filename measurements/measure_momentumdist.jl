""" 
    Measure momentum distribution
"""

function measure_momentum_dist(
    system::System, measure::GeneralMeasure, G::Vector{T}
    ) where {T<:AbstractMatrix}
    """
    Measure the momentum distribution along a given symmetry path
    kpath -> a symmetry path in the reciprocal lattice
    """
    nk = (
        zeros(ComplexF64, length(measure.DFTmats)),
        zeros(ComplexF64, length(measure.DFTmats))
        )
    for (i, DFTmat) in enumerate(measure.DFTmats)
        nk[1][i] = sum(DFTmat .* G[1]) / system.V
        nk[2][i] = sum(DFTmat .* G[2]) / system.V
    end

    return nk

end
