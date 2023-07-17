"""
    Generate matrices for constructing the propagator
"""

function hopping_matrix_Hubbard_1d(
    L::Int, t::Float64;
    isOBC::Bool = true
)
    T = zeros(L, L)

    isOBC ? begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-2]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-2]
        hop_amp = [-t for _ in 0 : L-2]
    end :
    # periodic boundary condition
    begin
        hop_ind_left = [CartesianIndex(i+1, (i+1)%L+1) for i in 0 : L-1]
        hop_ind_right = [CartesianIndex((i+1)%L+1, i+1) for i in 0 : L-1]
        hop_amp = [-t for _ in 0 : L-1]
    end

    @views T[hop_ind_left] = hop_amp
    @views T[hop_ind_right] = hop_amp
    
    return T
end

function hopping_matrix_Hubbard_2d(Lx::Int, Ly::Int64, t::Float64)
    L = Lx * Ly
    T = zeros(L, L)

    x = collect(0:L-1) .% Lx       # x positions for sites
    y = div.(collect(0:L-1), Lx)   # y positions for sites
    T_x = (x .+ 1) .% Lx .+ Lx * y      # translation along x-direction
    T_y = x .+ Lx * ((y .+ 1) .% Ly)    # translation along y-direction

    hop_ind_left = [CartesianIndex(i+1, T_x[i+1] + 1) for i in 0 : L-1]
    hop_ind_right = [CartesianIndex(T_x[i+1] + 1, i+1) for i in 0 : L-1]
    hop_ind_down = [CartesianIndex(i+1, T_y[i+1] + 1) for i in 0 : L-1]
    hop_ind_up = [CartesianIndex(T_y[i+1] + 1, i+1) for i in 0 : L-1]

    @views T[hop_ind_left] .= -t
    @views T[hop_ind_right] .= -t
    @views T[hop_ind_down] .= -t
    @views T[hop_ind_up] .= -t
    
    return T
end

### Auxiliary-field Matrix ###
"""
    auxfield_matrix_hubbard(σ, auxfield)
    
    Hubbard HS field matrix generator
"""
function auxfield_matrix_hubbard(
    σ::AbstractArray{Int}, auxfield::Vector{T};
    V₊ = zeros(T, length(σ)),
    V₋ = zeros(T, length(σ)),
    isChargeHST::Bool = false
) where T
    isChargeHST && begin
        for i in eachindex(σ)
            isone(σ[i]) ? (idx₊ = idx₋ = 1) : (idx₊ = idx₋ = 2)
            V₊[i] = auxfield[idx₊]
            V₋[i] = auxfield[idx₋]
        end
        
        return V₊, V₋
    end
    
    for i in eachindex(σ)
        isone(σ[i]) ? (idx₊ = 1; idx₋ = 2) : (idx₊ = 2; idx₋ = 1)
        V₊[i] = auxfield[idx₊]
        V₋[i] = auxfield[idx₋]
    end
    
    return V₊, V₋
end

function singlestep_matrix(
    σ::AbstractArray{Int64}, system::Hubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter
)
    """
    Compute B = exp(-ΔτK/2) * exp(-ΔτV(σ)) * exp(-ΔτK/2)
    """
    afmat_up, afmat_dn = auxfield_matrix_hubbard(σ, system.auxfield)

    if useFirstOrderTrotter
        B = [system.Bkf * Diagonal(afmat_up), system.Bkf * Diagonal(afmat_dn)]
    else
        B = [system.Bk * Diagonal(afmat_up) * system.Bk, system.Bk * Diagonal(afmat_dn) * system.Bk]
    end
    
    return B
end

function singlestep_matrix!(
    Bup::AbstractMatrix{T}, Bdn::AbstractMatrix{T}, σ::AbstractArray{Int64}, system::Hubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(Bup)
) where {T<:Number}
    """
    Compute B = exp(-ΔτK/2) * exp(-ΔτV(σ)) * exp(-ΔτK/2) in-place
    """
    afmat_up, afmat_dn = auxfield_matrix_hubbard(σ, system.auxfield)

    if useFirstOrderTrotter
        mul!(Bup, system.Bkf, Diagonal(afmat_up))
        mul!(Bdn, system.Bkf, Diagonal(afmat_dn))
    else
        mul!(tmpmat, Diagonal(afmat_up), system.Bk)
        mul!(Bup, system.Bk, tmpmat)
        
        mul!(tmpmat, Diagonal(afmat_dn), system.Bk)
        mul!(Bdn, system.Bk, tmpmat)
    end

    return nothing
end

"""
    Compute the propagator matrix for generic Hubbard Model 
    (spin decomposition is used, up and down parts are different)
"""
function imagtime_propagator!(
    B₊::AbstractMatrix{T}, B₋::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::GenericHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isChargeHST = system.useChargeHST
    )
    V₊, V₋ = system.V₊, system.V₋

    if useFirstOrderTrotter
        mul!(B₊, Bₖ, Diagonal(V₊))
        mul!(B₋, Bₖ, Diagonal(V₋))
    else
        mul!(tmpmat, Diagonal(V₊), Bₖ)
        mul!(B₊, Bₖ, tmpmat)
        
        mul!(tmpmat, Diagonal(V₋), Bₖ)
        mul!(B₋, Bₖ, tmpmat)
    end

    return nothing
end

"""
    Compute the propagator matrix for generic Hubbard Model
    (charge decomposition is used, up and down parts are the same)
"""
function imagtime_propagator!(
    B::AbstractMatrix{T},
    σ::AbstractArray{Int}, system::GenericHubbard;
    useFirstOrderTrotter::Bool = system.useFirstOrderTrotter,
    tmpmat = similar(B₊)
) where {T<:Number}
    Bₖ = system.Bk

    auxfield_matrix_hubbard(
        σ, system.auxfield, 
        V₊=system.V₊, V₋=system.V₋,
        isChargeHST = system.useChargeHST
    )
    V₊ = system.V₊

    if useFirstOrderTrotter
        mul!(B, Bₖ, Diagonal(V₊))
    else
        mul!(tmpmat, Diagonal(V₊), Bₖ)
        mul!(B, Bₖ, tmpmat)
    end

    return nothing
end
