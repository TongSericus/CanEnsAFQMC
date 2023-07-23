"""
    Energy measurements
"""

"""
    measure_Energy(system::Hubbard, ρ₊::DensityMatrix, ρ₋::DensityMatrix)

    Energy measurement for Hubbard type model
"""
function measure_Energy(
    system::Hubbard, ρ₊::DensityMatrix, ρ₋::DensityMatrix;
    E::AbstractVector = zeros(eltype(system.auxfield), 3)
)
    ρ₁₊ = ρ₊.ρ₁
    ρ₁₋ = ρ₋.ρ₁
    T = system.T

    # kinetic
    for i in eachindex(T)
        if T[i] != 0
            E[1] += T[i] * (ρ₁₊[i] + ρ₁₋[i])
        end
    end

    # potential
    for i = 1 : system.V
        E[2] += system.U * (ρ₁₊[i, i] * ρ₁₋[i, i])
    end

    # total
    E[3] = E[1] + E[2]

    return E
end
