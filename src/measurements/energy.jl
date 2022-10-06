"""
    Measure energy
"""
function measure_energy(
    system::Hubbard, 
    G_up::AbstractMatrix{T}, G_dn::AbstractMatrix{T}
) where {T<:FloatType}
    """
    Measure the kinetic (one-body), the potential (two-body) energy and total energy
    """
    Ek, Ep = 0, 0

    for i in eachindex(@view system.T[1 : end, 1 : end])
        if system.T[i] != 0
            Ek += -system.t * (G_up[i[1], i[2]] + G_dn[i[1], i[2]])
        end
    end

    for i = 1 : system.V
        Ep += system.U * (G_up[i, i] * G_dn[i, i])
    end

    return real(Ek), real(Ep), real(Ek + Ep)

end
