"""
    Charge Correlation Function and the corresponding Fourier Transform (Structural Factor)
"""
function measure_ChargeCorr(
    system::System, DMup::DensityMatrices{T, Tn, Tp}, DMdn::DensityMatrices{T, Tn, Tp};
    Ns = system.V, 
    ninj::AbstractMatrix{T} = zeros(T, Ns, Ns)
) where {T, Tn, Tp}
    """
    Compute ⟨n_{i}n_{j}⟩ = ⟨(n_{i↑} + n_{i↓})(n_{j↑} + n_{j↓})⟩
    """
    Do = [DMup.Do, DMdn.Do]
    Dt = [DMup.Dt, DMdn.Dt]

    @inbounds for j in 1 : Ns
        for i in 1 : Ns
            ninj[i, j] = Dt[1][i, j] + Dt[2][i, j] + Do[1][i, i] * Do[2][j ,j] + Do[1][j, j] * Do[2][i, i]
        end
    end

    return ninj
end

function measure_SpinCorr(system::System, DMup::DensityMatrices{T, Tn, Tp}, DMdn::DensityMatrices{T, Tn, Tp};
    Ns = system.V, 
    sisj::AbstractMatrix{T} = zeros(T, Ns, Ns)
) where {T, Tn, Tp}
    """
    Compute spin-order in z-direction:
        ⟨s_{i}s_{j}⟩ = ⟨(n_{i↑} - n_{i↓})(n_{j↑} - n_{j↓})⟩
    """
    Do = [DMup.Do, DMdn.Do]
    Dt = [DMup.Dt, DMdn.Dt]

    @inbounds for j in 1 : Ns
        for i in 1 : Ns
            sisj[i, j] = Dt[1][i, j] + Dt[2][i, j] - Do[1][i, i] * Do[2][j ,j] - Do[1][j, j] * Do[2][i, i]
        end
    end

    return sisj
end
