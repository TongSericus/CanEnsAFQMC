"""
    A correlated sampling scheme to compute energy difference
    using the method introduced in doi.org/10.1103/PhysRevLett.87.022501
"""
function measure_HeatCapacity_denom(
    system_pβ::System, system_mβ::System, qmc::QMC, walker::Walker;
    Z_pmβ::AbstractVector = zeros(Float64, 2),
    sign_pmβ::AbstractVector{ComplexF64} = zeros(ComplexF64, 2),
    returnF::Bool=false
)
    N = system_pβ.N

    F_pβ, dummy1, dummy2 = run_full_propagation(walker.auxfield, system_pβ, qmc)
    weight_pβ = compute_PF(F_pβ[1], N[1]) + compute_PF(F_pβ[2], N[2])

    F_mβ, dummy1, dummy2 = run_full_propagation(walker.auxfield, system_mβ, qmc)
    weight_mβ = compute_PF(F_mβ[1], N[1]) + compute_PF(F_mβ[2], N[2])

    # compute the denominator Z(β ± δβ)
    Z_pβ = weight_pβ - sum(walker.weight)
    sign_pmβ[1] = sign(exp(imag(Z_pβ)im))
    Z_pmβ[1] = exp(real(Z_pβ))

    Z_mβ = weight_mβ - sum(walker.weight)
    sign_pmβ[2] = sign(exp(imag(Z_mβ)im))
    Z_pmβ[2] = exp(real(Z_mβ))

    returnF && return F_pβ, F_mβ
    return Z_pmβ, sign_pmβ
end

function measure_HeatCapacity_num(
    system_pβ::System, system_mβ::System, qmc::QMC, walker::Walker;
    Z_pmβ::AbstractVector = zeros(Float64, 2),
    DMup_pmβ::DensityMatrices=DensityMatrices(system_pβ),
    DMdn_pmβ::DensityMatrices=DensityMatrices(system_pβ),
    H_pmβ::AbstractVector = zeros(eltype(DMup_pmβ.Do), 2),
    sign_pmβ::AbstractVector{ComplexF64} = zeros(ComplexF64, 2)
)
    N = system_pβ.N

    F_pβ, F_mβ = measure_HeatCapacity_denom(
        system_pβ, system_mβ, qmc, walker, 
        Z_pmβ=Z_pmβ, sign_pmβ=sign_pmβ,
        returnF=true
    )

    # compute the numerator H(β ± δβ)
    fill_DM!(DMup_pmβ, F_pβ[1], N[1], computeTwoBody=false)
    fill_DM!(DMdn_pmβ, F_pβ[2], N[2], computeTwoBody=false)
    Etot = measure_Energy(system_pβ, DMup_pmβ.Do, DMdn_pmβ.Do)[3]
    H_pmβ[1] = Etot * Z_pmβ[1]

    fill_DM!(DMup_pmβ, F_mβ[1], N[1], computeTwoBody=false)
    fill_DM!(DMdn_pmβ, F_mβ[2], N[2], computeTwoBody=false)
    Etot = measure_Energy(system_mβ, DMup_pmβ.Do, DMdn_pmβ.Do)[3]
    H_pmβ[2] = Etot * Z_pmβ[2]

    return H_pmβ, sign_pmβ
end
