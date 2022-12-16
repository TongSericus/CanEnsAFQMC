"""
    QR with rank-1 update
    Stolen and modified from https://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRUpdate.html
"""
function GivensRotation(a::Float64, b::Float64)
    # Calculate the Given's rotation that rotates [a;b] to [r;0]:
    c = 0.; s = 0.; r = 0.
    if (b == 0.)
        c = sign(a)
        s = 0.
        r = abs(a)
    elseif (a == 0.)
        c = 0.
        s = -sign(b)
        r = abs(b)
    elseif (abs(a) .> abs(b))
        t = b/a
        u = sign(a)*abs(sqrt(1+t^2))
        c = 1/u
        s = -c*t
        r = a*u
    else
        t = a/b
        u = sign(b)*abs(sqrt(1+t^2))
        s = -1/u
        c = -s*t
        r = b*u
    end
    return (c, s, r)
end

function HessenbergQR(R::Matrix{T}) where {T<:Number}
    # Compute the QR factorization of an upper-Hessenberg matrix: 
    
    n = size(R, 1)
    Q = Matrix(1.0I, n, n)
    # Convert R from upper-hessenberg form to upper-triangular form using n-1 Givens rotations:
    for k = 1:n-1
        (c, s, r) = GivensRotation(R[k,k], R[k+1,k])
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] - s*Q[j,k+1]
            Q[j,k+1] = s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end
    return (Q, R)
end

function QR_update(F::UDR, u::Vector{T}, v::Vector{T}) where {T<:Number}
    # Compute the QR factorization of Q*R + u*v': 

    Q = copy(F.U)
    R = Diagonal(F.D) * F.R
    
    # Note that Q*R + u*v' = Q*(R + w*v') with w = Q'*u:
    w = Q' * u
    n = size(Q, 1)
    
    # Convert R+w*v' into upper-hessenberg form using n-1 Givens rotations:
    for k = n-1:-1:1
        (c, s, r) = GivensRotation(w[k], w[k+1])
        w[k+1] = 0.; w[k] = r
        # Compute G*R[k:k+1,:] and Q[:,k:k+1]*G', where G = [c -s ; s c]
        for j = 1:n
            newrow = c*R[k,j] - s*R[k+1,j]
            R[k+1,j] = s*R[k,j] + c*R[k+1,j]
            R[k,j] = newrow
            newcol = c*Q[j,k] - s*Q[j,k+1]
            Q[j,k+1] = s*Q[j,k] + c*Q[j,k+1]
            Q[j,k] = newcol
        end
    end
    # R <- R + w*v' is now upper-hessenberg:
    R[1,:] += w[1] * v 
    
    (Q1, R1) = HessenbergQR(R)
    D = Vector{T}(undef, n)
    @inbounds for i in 1 : n
        D[i] = abs(R[i, i])
    end
    lmul!(Diagonal(1 ./ D), R1)
    
    # Return updated QR factorization: 
    return UDR(Q * Q1, D, R1)
end

### Two-body Observables Calculation ###
function compute_γ!(
    γ::AbstractArray{Ty, 3},
    λ::Vector{T}, N::Int, P::AbstractMatrix{Tp}, invP::AbstractMatrix{Tp};
    Ns = length(λ),
    Nft = length(λ),
    iφ = im * [2 * π * m / Nft for m = 1 : Nft],
    expiφ = exp.(iφ)
) where {Ty, T, Tp}
    """
    Compute the Fourier-frequency-dependent one-body debsity matrices
    """
    expβμ, expiφmβμ, η, logZ = pf_projection_stable(λ, N, returnFull=true, Nft=Nft, iφ=iφ, expiφ=expiφ)
    
    ñ = zeros(ComplexF64, Ns)
    mat = similar(invP)
    for m in 1 : Nft

        for i in 1 :Ns
            ñ[i] = expiφmβμ[m] * λ[i] / (1 + expiφmβμ[m] * λ[i])
        end

        mul!(mat, Diagonal(ñ), invP)
        @views mul!(γ[:, :, m], P, mat)
    end

    return log(expβμ), η, logZ
end

function ninj_projection_stable(
    λ::Vector{T}, N::Int, P::AbstractMatrix{Tp}, invP::AbstractMatrix{Tp};
    Ns = length(λ),
    Nft = length(λ),
    iφ = im * [2 * π * m / Nft for m = 1 : Nft],
    expiφ = exp.(iφ),
    γ = zeros(ComplexF64, Ns, Ns, Nft),
    ninj = zeros(ComplexF64, Ns, Ns)
) where {T, Tp}
    βμ, η, logZ = compute_γ!(
        γ, λ, N, P, invP, 
        Ns=Ns, Nft = Nft, iφ=iφ, expiφ=expiφ
    )

    @inbounds for j in 1 : Ns
        for i in i : Ns
            δij = (i == j)
            for m in 1 : Ns
                ñiñj = γ[i, i, m] * γ[j, j, m] - γ[j, i, m] * γ[i, j, m] + δij * γ[j, i, m]
                logñ = -iφ[m] * N + log(ñiñj) + η[m]
                logninj = logñ + βμ * N - logZ
                ninj[i, j] += exp(logninj)
            end
            ninj[i, j] /= Nft
            ninj[j, i] = ninj[i, j]
        end
    end

    return ninj
end
