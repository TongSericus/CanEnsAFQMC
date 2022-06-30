"""
    Measure energy
"""
function measure_energy(
    system::Hubbard, walker_list::Vector{WalkerProfile{T1, T2, T3}}
) where {T1<:FloatType, T2<:FloatType, T3<:FloatType}
    """
    Measure the kinetic (one-body), the potential (two-body) energy and total energy
    """
    Ek, Ep = 0, 0

    for i in eachindex(@view system.T[1 : end, 1 : end])
        if system.T[i] != 0
            Ek += -system.t * (walker_list[1].G[i[1], i[2]] + walker_list[2].G[i[1], i[2]])
        end
    end

    for i = 1 : system.V
        Ep += system.U * (walker_list[1].G[i, i] * walker_list[2].G[i, i])
    end

    return Ek, Ep, Ek + Ep

end
