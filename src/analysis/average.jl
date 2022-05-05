using JLD

function average_samples(dirs::Vector{String}, isReal::Bool)
    # names of non-empty fields
    f = Symbol[]
    for (index, dir) in enumerate(dirs)
        data = load(dir, "sample_list")
        if index == 1

        end
    end
end

@generated function isEmptyField(sample::MCSample)
    assignments = [
        :(isempty(sample.$name)) for name in fieldnames(MCSample)
    ]
end
