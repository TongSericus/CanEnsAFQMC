"""
    One- and Two-body Density Matrices for Observable Measurements
"""

struct DensityMatrix{T,E}
    # One-body RDM: <c_i^{+} cj>
    ρ₁::Matrix{T}

    ws::LDRWorkspace{ComplexF64,E}

    # eigendecomposition
    λ::Vector{ComplexF64}
    P::Matrix{ComplexF64}
    P⁻¹::Matrix{ComplexF64}
    t::Base.RefValue{UnitRange{Int}}

    # Fourier transform
    Nft::Int
    iφ::Vector{ComplexF64}
    expiφ::Vector{ComplexF64}
    expiφμ::Vector{ComplexF64}
    Z̃ₘ::Vector{ComplexF64}   # Fourier weights
    ρₘ::Array{ComplexF64, 3} # frequency-dependent 1-RDM
    Gₘ::Array{ComplexF64, 3} # ρₘ with permuted dimension
end

function DensityMatrix(system::System; Nft::Int = system.V+1)
    T = eltype(system.auxfield)
    Ns = system.V

    ρ₁ = zeros(T, Ns, Ns)

    # eigendecomposition
    λ = zeros(ComplexF64, Ns)
    P = zeros(ComplexF64, Ns, Ns)
    P⁻¹ = zeros(ComplexF64, Ns, Ns)
    t = Ref(1:Ns)

    ws = ldr_workspace(P)

    # Fourier transform
    iφ = im * [2*π*m / Nft for m = 1 : Nft]
    expiφ = exp.(iφ)
    expiφμ = zeros(ComplexF64, Nft)
    Z̃ₘ = zeros(ComplexF64, Nft)
    ρₘ = zeros(ComplexF64, Ns, Ns, Nft)
    Gₘ = zeros(ComplexF64, Nft, Ns, Ns)

    return DensityMatrix{T, eltype(ws.v)}(
        ρ₁, ws,
        λ, P, P⁻¹, t,
        Nft, iφ, expiφ, expiφμ, Z̃ₘ, ρₘ, Gₘ
    )
end

"""
    compute_Fourier_weights(...)

    Compute the Fourier coefficients. Results are in the logrithmic form
    to avoid numerical overflows
"""
function compute_Fourier_weights(
    system::System, ρ::DensityMatrix, spin::Int
)
    N = system.N[spin]
    Ns = ρ.Nft
    expiφ = ρ.expiφ
    λ = ρ.λ[ρ.t[]]
    Z̃ₘ = ρ.Z̃ₘ

    expβμ = fermilevel(λ, N)
    βμN = N * log(expβμ)
    for i in 1:Ns
        ρ.expiφμ[i] = expiφ[i] / expβμ
    end

    λ = λ / expβμ
    for m in 1:Ns
        Z̃ₘ[m] = sum(log.(1 .+ expiφ[m]*λ)) + βμN - N*ρ.iφ[m]
    end

    return Z̃ₘ
end

function compute_RDM(ρ::DensityMatrix, ws::LDRWorkspace)

    Ns = ρ.Nft
    ρₘ = ρ.ρₘ

    t = ρ.t[]
    λ = @view ρ.λ[t]
    P = @view ρ.P[:, t]
    P⁻¹ = @view ρ.P⁻¹[t, :]
    expiφμ = ρ.expiφμ

    for m in 1 : Ns
        nₖφₘ = @view ws.M[t, 1]
        @inbounds for i in eachindex(λ)
            nₖφₘ[i] = expiφμ[m]*λ[i] / (1 + expiφμ[m]*λ[i])
        end

        nₖP⁻¹ = @view ws.M′[t, :]
        mul!(nₖP⁻¹, Diagonal(nₖφₘ), P⁻¹)
        Gₐ = @views ρₘ[:, :, m]
        mul!(Gₐ, P, nₖP⁻¹)
    end

    # permute the dimensions
    Gₘ = ρ.Gₘ
    permutedims!(Gₘ, ρₘ, [3, 1, 2])

    return ρₘ
end

"""
    update!(system::System, walker::Walker, ρ::DensityMatrix, spin::Int)

    Perform the eigendecomposition and write it into ρ
"""
function update!(
    system::System, walker::Walker{T, LDR{T,E}}, ρ::DensityMatrix, spin::Int
) where {T,E}
    # compute eigendecomposition
    λ, P, P⁻¹ = eigen(walker.F[spin], ρ.ws)
    copyto!(ρ.λ, λ)
    copyto!(ρ.P, P)
    copyto!(ρ.P⁻¹, P⁻¹)

    # compute Fourier coefficients
    compute_Fourier_weights(system, ρ, spin)
    Z̃ₘ = ρ.Z̃ₘ
    logZ = walker.weight[spin] + log(walker.sign[spin])
    for m in eachindex(Z̃ₘ)
        Z̃ₘ[m] = exp(Z̃ₘ[m] - logZ)
    end

    # compute frequency-dependent 1-RDMs
    compute_RDM(ρ, ρ.ws)

    # compute 1-RDM using inverse Fourier transform
    ρ₁ = ρ.ρ₁
    Gₘ = ρ.Gₘ
    for i in eachindex(IndexCartesian(), ρ₁)
        tmp = sum(Gₘ[:, i[1], i[2]] .* Z̃ₘ) / ρ.Nft
        ρ₁[i] = T <: Real ? real(tmp) : tmp
    end

    return nothing
end

function update!(
    system::System, walker::Walker{T, LDRLowRank{T,E}}, ρ::DensityMatrix, spin::Int
) where {T,E}
    # compute eigendecomposition
    λ, P, P⁻¹ = eigen(walker.F[spin], ρ.ws)
    ρ.t[] = walker.F[spin].t[]
    t = ρ.t[]
    @views copyto!(ρ.λ[t], λ)
    @views copyto!(ρ.P[:, t], P)
    @views copyto!(ρ.P⁻¹[t, :], P⁻¹)

    # compute frequency-dependent 1-RDMs
    compute_Fourier_weights(system, ρ, spin)
    Z̃ₘ = ρ.Z̃ₘ
    logZ = walker.weight[spin] + log(walker.sign[spin])
    for m in eachindex(Z̃ₘ)
        Z̃ₘ[m] = exp(Z̃ₘ[m] - logZ)
    end

    # compute frequency-dependent 1-RDMs
    compute_RDM(ρ, ρ.ws)

    # compute 1-RDM using inverse Fourier transform
    ρ₁ = ρ.ρ₁
    Gₘ = ρ.Gₘ
    for i in eachindex(IndexCartesian(), ρ₁)
        tmp = sum(Gₘ[:, i[1], i[2]] .* Z̃ₘ) / ρ.Nft
        ρ₁[i] = T <: Real ? real(tmp) : tmp
    end

    return nothing
end

function update!(system::System, walker::Walker, ρ₋::DensityMatrix, ρ₊::DensityMatrix)
    isConj = !system.useChargeHST

    ## update the eigendecomposition ##
    copyto!(ρ₋.λ, ρ₊.λ)
    isConj && conj!(ρ₋.λ)
    # update eigenvectors
    copyto!(ρ₋.P, ρ₊.P)
    isConj && conj!(ρ₋.P)
    copyto!(ρ₋.P⁻¹, ρ₊.P⁻¹)
    isConj && conj!(ρ₋.P⁻¹)
    # and the truncation
    ρ₋.t[] = ρ₊.t[]

    # compute the Fourier weights and frequency-dependent 1-RDMs 
    # (these are not simply complex conjugates)
    compute_Fourier_weights(system, ρ₋, 2)
    Z̃ₘ = ρ₋.Z̃ₘ
    logZ = walker.weight[2] + log(walker.sign[2])
    for m in eachindex(Z̃ₘ)
        Z̃ₘ[m] = exp(Z̃ₘ[m] - logZ)
    end
    compute_RDM(ρ₋, ρ₋.ws)

    # update the 1-RDM
    copyto!(ρ₋.ρ₁, ρ₊.ρ₁)
    isConj && conj!(ρ₋.ρ₁)

    return nothing
end

"""
    Compute the two-body estimator <cᵢ⁺ cⱼ cₖ⁺ cₗ> as
    <cᵢ⁺ cⱼ cₖ⁺ cₗ> = <cᵢ⁺ cⱼ> <cₖ⁺ cₗ> + <cᵢ⁺ cl> (δₖⱼ - <cₖ⁺ cⱼ>)
"""
function ρ₂(ρ::DensityMatrix, i::Int, j::Int, k::Int, l::Int)
    Gₘ = ρ.Gₘ
    s = 0
    for m in 1 : ρ.Nft
        tmp = Gₘ[m, i, j] * Gₘ[m, k, l] + Gₘ[m, i, l] * ((k==j) - Gₘ[m, k, j])
        s += tmp * ρ.Z̃ₘ[m]
    end

    return s / ρ.Nft
end
