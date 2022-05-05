"""
    Measure energy
"""
function measure_energy_hubbard(
    system::System, walker_list::Vector{WalkerProfile}
)
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
