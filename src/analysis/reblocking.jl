"""
    Steal from https://dfm.io/posts/autocorr/ and rewrite into Julia form
"""
using FFTW: fft, ifft
function next_pow_two(n::Int64)
    i = 1
    while i < n
        i = i << 1
    end
    return i
end

function autocorr_func_1d(x::Vector{T}) where {T <: FloatType}
    n = next_pow_two(x)
    x = vcat(x .- mean(x), zeros(T, 2 * n - length(x)))
    # Compute the FFT and then (from that) the auto-correlation function
    f = fft(x)
    acf = real(ifft(f .* conj(f)))[1 : length(x)]
    acf /= acf[1]
    return acf
end

function autocorr_func_test(x::Vector{T}) where {T <: FloatType}
    N = length(x)
    mu = mean(x)
    Ctau = zeros(T, N)
    for tau = 0 : N - 1
        for n = 1 : N - tau
            Ctau[tau + 1] += (x[n] - mu) * (x[n + tau] - mu)
        end
    end
    return Ctau
end