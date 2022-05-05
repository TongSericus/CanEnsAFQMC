"""
    Autocorrelation function estimate and sample reblocking
    Steal from https://dfm.io/posts/autocorr/ and rewrite into Julia form
"""

import FFTW: fft, ifft
import Statistics: mean, std

function next_pow_two(n::Int64)
    """
    Returns the next power of two greater than or equal to n
    """
    i = 1
    while i < n
        i = i << 1
    end
    return i
end

function autocorr_func_1d(x::Vector{T}) where {T <: FloatType}
    """
    Estimate the normalized autocorrelation function of a 1-D series
    """
    n = next_pow_two(length(x))
    x_expand = vcat(x .- mean(x), zeros(T, 2 * n - length(x)))
    # Compute the FFT and then (from that) the auto-correlation function
    f = fft(x_expand)
    acf = real(ifft(f .* conj(f)))[1 : length(x)]
    acf /= acf[1]
    return acf
end

# Automated windowing procedure following Sokal (1989)
auto_window(τf, c) = argmin([i for i = 0 : length(τf) - 1] .< c * τf)

function autocorr_gw2010(x::Vector{T}, c::Float64 = 5.0) where {T <: FloatType}
    """
    Following the suggestion from Goodman & Weare (2010)
    """
    acf = autocorr_func_1d(x)
    τf = 2 * cumsum(acf) .- 1
    return τf[auto_window(τf, c)]
end

function reblock(x::Vector{T}) where {T <: FloatType}
    """
    Compute the reblocked sample averages and error bars using ACF
    """
    Nmax = convert(Int64, floor(log2(length(x))))
    τ = Vector{T}()
    for i = 0 : Nmax - 1
        n = convert(Int64, floor(length(x) / 2 ^ i))
        push!(τ, autocorr_gw2010(x[1 : n]))
    end

    block_size = convert(Int64, round(maximum(τ)))
    nblocks = div(length(x), block_size)
    xblocked = Vector{T}()
    for i = 1 : nblocks
        offset = (i - 1) * block_size + 1
        push!(xblocked, mean(x[offset : offset + block_size - 1]))
    end

    return mean(xblocked), std(xblocked) / sqrt(nblocks)

end
