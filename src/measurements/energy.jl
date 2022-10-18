"""
    Measure energy
"""
function measure_Energy(
    system::Hubbard, 
    Dup::AbstractMatrix{T}, Ddn::AbstractMatrix{T};
    E::AbstractVector{T} = zeros(T, 3)
) where {T<:Number}
    """
    Measure the kinetic (one-body), the potential (two-body) energy and total energy
    """

    for i in eachindex(@view system.T[1 : end, 1 : end])
        if system.T[i] != 0
            E[1] += -system.t * (Dup[i[1], i[2]] + Ddn[i[1], i[2]])
        end
    end

    for i = 1 : system.V
        E[2] += system.U * (Dup[i, i] * Ddn[i, i])
    end

    E[3] = E[1] + E[2]

    return E
end
