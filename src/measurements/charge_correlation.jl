function measure_ChargeCorr(
    system::Hubbard, DMup::DensityMatrices{T, Tn, Tp}, DMdn::DensityMatrices{T, Tn, Tp};
    Ns = system.V, 
    ninj::AbstractMatrix{T} = zeros(T, Ns, Ns)
) where {T, Tn, Tp}
    Do = [DMup.Do, DMdn.Do]
    Dt = [DMup.Dt, DMdn.Dt]

    @inbounds for j in 1 : Ns
        for i in 1 : Ns
            ninj[i, j] = Dt[1][i, j] + Dt[2][i, j] + Do[1][i, i] * Do[2][j ,j] + Do[1][j, j] * Do[2][i, i]
        end
    end

    return ninj
end

function compute_StructFactor(ninj::AbstractMatrix{T}, q::Float64) where T
    L = size(ninj, 1)
    Sq = 0
    for j in 1 : L
        for i in 1 : L
            Sq += exp(im * q * (i - j)) * ninj[i, j]
        end
    end

    return Sq / L
end