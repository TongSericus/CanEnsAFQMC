"""
    Charge Correlation Function and the corresponding Fourier Transform (Structural Factor)
"""

function enumerate_r()
end

function add_r()
end

struct CorrFuncSampler
end

function CorrFuncSampler(system::System)
end

"""
    Compute ⟨n_{i}n_{j}⟩ = ⟨(n_{i↑} + n_{i↓})(n_{j↑} + n_{j↓})⟩
"""
function measure_ChargeCorr(
    system::System, DMup::DensityMatrices{T, Tn, Tp}, DMdn::DensityMatrices{T, Tn, Tp};
    Ns = system.V, 
    nᵢ₊ᵣnᵢ::AbstractMatrix{T} = zeros(T, Ns, Ns)
) where {T, Tn, Tp}
    Do = [DMup.Do, DMdn.Do]
    Dt = [DMup.Dt, DMdn.Dt]

    @inbounds for j in 1 : Ns
        for i in 1 : Ns
            nᵢ₊ᵣnᵢ[i, j] = Dt[1][i, j] + Dt[2][i, j] + Do[1][i, i] * Do[2][j ,j] + Do[1][j, j] * Do[2][i, i]
        end
    end

    return nᵢ₊ᵣnᵢ
end

"""
    measure_SpinCorr(system::System)

    Compute second-order spin-order in z-direction:
    ⟨Sᵢ₊ᵣSᵢ⟩ = ⟨(nᵢ₊ᵣ↑-nᵢ₊ᵣ↓)(nᵢ↑-nᵢ↓))⟩
"""
function measure_SpinCorr(system::System, ρ₊::DensityMatrix, ρ₋::DensityMatrix;
    Ns = system.V, 
    Sᵢ₊ᵣSᵢ::AbstractMatrix{T} = zeros(T, Ns, Ns)
) where {T, Tn, Tp}
    Do = [DMup.Do, DMdn.Do]
    Dt = [DMup.Dt, DMdn.Dt]

    @inbounds for j in 1 : Ns
        for i in 1 : Ns
            Sᵢ₊ᵣSᵢ[i, j] = Dt[1][i, j] + Dt[2][i, j] - Do[1][i, i] * Do[2][j ,j] - Do[1][j, j] * Do[2][i, i]
        end
    end

    return Sᵢ₊ᵣSᵢ
end
