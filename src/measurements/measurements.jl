struct Measurements{T}
    # Energy -> Ek, Ep, Etot
    E::Vector{T}

    # Heat Capacity
    Z_pmβ::Vector{T}
    H_pmβ::Vector{T}
    sign_pmβ::Vector{T}

    # Momentum Distribution
    nk::T

    # Charge Structural Factor
    Sq::T
end

### Matrices for Fourier Transform ###
function generate_rmat(system::System)
    V = system.V
    # 1D 
    if system.Ns[2] == 1
        rmat = zeros(Int64, V, V)

        for r1 = 1 : V
            for r2 = r1 : V
                rmat[r1, r2] = r1 - r2
                rmat[r2, r1] = r2 - r1
            end
        end

        return rmat
    # 2D
    else
        NsX = system.Ns[1]

        rxmat = zeros(Int64, V, V)
        rymat = zeros(Int64, V, V)

        for r1 = 1 : V
            for r2 = 1 : V
                r1x, r1y = decode_basis(r1, NsX)
                r2x, r2y = decode_basis(r2, NsX)

                rxmat[r1, r2] = r1x - r2x
                rymat[r1, r2] = r1y - r2y
            end
        end

        return rxmat, rymat
    end
end

function generate_DFTmat(system::System, q::Q; rmat = generate_rmat(system)) where {Q<:Number}
    V = system.V

    DFTmat = Matrix{ComplexF64}(I, V, V)
    for j in 1 : V
        for i in 1 : V
            qr = dot(q, rmat[i, j])
            DFTmat[i, j] = exp(im * qr)
        end
    end

    return DFTmat
end


function generate_DFTmat(
    system::System, qx::Q, qy::Q;
    rmat = generate_rmat(system)
) where {Q<:Number}
    V = system.V
    rxmat, rymat = rmat

    DFTmat = Matrix{ComplexF64}(I, V, V)
    for j in 1 : V
        for i in 1 : V
            qr = dot([qx, qy], [rxmat[i, j], rymat[i, j]])
            DFTmat[i, j] = exp(im * qr)
        end
    end

    return DFTmat
end

function generate_DFTmats(system::System)
    rmat = generate_rmat(system)

    # 1D
    if system.Ns[2] == 1
        V = system.V
        L = div(system.Ns[1], 2)
        q = [n*π / L for n in -L + 1 : L]

        return [generate_DFTmat(system, q[i], rmat = rmat) for i in 1:V]
    # 2D
    else
        NsX, NsY = system.Ns
        Lx, Ly = div.(system.Ns, 2)
        qx = [n*π / Lx for n in -Lx + 1 : Lx]
        qy = [n*π / Ly for n in -Ly + 1 : Ly]

        return [generate_DFTmat(system, qx[i], qy[j], rmat = rmat) for i in 1:NsX for j in 1:NsY]
    end
end
