""" 
    Measure momentum distribution
"""

function measure_momentum_dist(
    system::System, measure::GeneralMeasure, walker_list::Vector{WalkerProfile}
)
    """
    Measure the momentum distribution along a given symmetry path
    kpath -> a symmetry path in the reciprocal lattice
    """
    nk = (
        zeros(ComplexF64, length(measure.DFTmats)),
        zeros(ComplexF64, length(measure.DFTmats))
        )
    for (i, DFTmat) in enumerate(measure.DFTmats)
        nk[1][i] = sum(DFTmat .* walker_list[1].G) / system.V
        nk[2][i] = sum(DFTmat .* walker_list[2].G) / system.V
    end

    return nk

end
