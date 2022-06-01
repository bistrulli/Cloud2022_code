# type aliases
T_simSetting = Dict{String, Dict{String, Any}}
T_H = Dict{Tuple{String, String}, DataFrame}
T_qn = Union{Tuple{String,String}, String}

tryparse(::Type{Int64}, a::Int64) = a

import StatsBase.var
var(ph::EMpht.PhaseType) = 2*ph.π'*((ph.T^2) \ ones(ph.p)) - mean(ph)^2

# Requires that first dimension is where the MCsim are iterated over
function concatMC(y::Array{T, 3}) where T <: Any
    y_c = Array{T, 2}(undef, size(y,2), size(y, 3))
    for i = 1:size(y, 2)
        for j = 1:size(y, 3)
            y_c[i,j] = vcat(y[:, i, j]...)
        end
    end
    return y_c
end

# Requires that first dimension is where the MCsim are iterated over
function concatMC(y::Array{T, 2}) where T <: Any
    y_c = Array{T, 1}(undef, size(y,2))
    for i = 1:size(y, 2)
        y_c[i] = vcat(y[:, i]...)
    end
    return y_c
end

 # Split the input vector x between the values in t. Returns t-1 values and
 # their indices in vectors
function splitByT(x::Vector{T}, t::Vector{T}) where T <: Number
    @assert all(diff(t) .>= 0) "vector t must be a vector of ascending intervals"
    x_split = Vector{Vector{T}}(undef, length(t)-1)
    x_idx = Vector{Vector{Bool}}(undef, length(t)-1)
    for k = 1:length(t)-1
        x_idx[k] = t[k] .<= x .<= t[k+1]
        x_split[k] = x[x_idx[k]]
    end
    return x_split, x_idx
end

# Get all the indices as bools of occurances of t in the given steps
function stepIndices(t::Vector{T}, steps::Vector{T}; lag=0) where T <: Number
    if lag == 0
        lag = unique(diff(steps))[1]
    end
    
    ind = Vector{Vector{Bool}}(undef, length(steps))
    
    ind[1] = zeros(Bool, length(t))
    k = 2
    for k = 2:length(steps)
        ind[k] = max(steps[k]-lag, 0) .< t .< steps[k]
    end
    return ind
end

# Get all the indices as bools of occurances of t in the given steps. Assumes 
# that t is sorted
function stepIndicesSorted(t::Vector{T}, steps::Vector{T}; lag=0) where T <: Number
    if lag == 0
        lag = steps[2] - steps[1]
    end
    
    ind = Vector{UnitRange}(undef, length(steps))
    
    ind[1] = 1:0
    idx_upper = 1
    idx_lower = 1

    moved = false
    for k = 2:length(steps)

        # move forward the upper bound
        if idx_upper < length(t)
            while t[idx_upper+1] < steps[k]
                moved = true
                idx_upper += 1
                if idx_upper >= length(t)
                    break
                end
            end
        end

        # move forward the lower bound
        if idx_lower < length(t)
            while t[idx_lower] < max(steps[k]-lag, 0)
                moved = true
                idx_lower += 1
                if idx_lower >= length(t)
                    break
                end
            end
        end

        if moved
            ind[k] = idx_lower:idx_upper 
            moved = false
        else
            ind[k] = 1:0
        end
    end
    return ind
end

function test_steps()
    s1 = stepIndicesSorted(t, steps)
    s2 = stepIndices(t, steps)

    # Test overlapping
    s1_nozero = s1[length.(s1) .> 0]
    @assert all((maximum.(s1_nozero)[1:end-1] .- minimum.(s1_nozero)[2:end]) .== -1)    

    s1_vecs = [zeros(Bool, length(t)) for k = 1:length(steps)]
    for (i, interval) in enumerate(s1) 
        s1_vecs[i][interval] .= true
    end

end

# Filter outliers in the input vectors
function filtOutliers(x::Array{Float64, 1}; ϵ=0.00025, α=0.999, β=100)
    if isempty(x)
        return x
    end

    x .+= ϵ
    idx = collect(1:length(x))
    if maximum(x) > β*median(x)
        a = quantile(x, α)
        filter!(i -> x[i] < a, idx)
    end
    return x[idx]
end

import Distributions.scale
# Scales the inputted PH distribution
function scale(ph::EMpht.PhaseType, a::T) where T <: Number
    return EMpht.PhaseType(ph.π, a*ph.T)
end

# Retrieves the cdf over a data vector as a function closure.
function cdfData(y::Vector{Float64}) 
    ys = sort(y)
    pvec = (1:length(y)) ./ length(y)
    function _cdfData(x)
        if x < 0; return 0.0; end
        idx = findfirst(z -> z > x, ys)
        return isnothing(idx) ? 1.0 : pvec[idx]
    end
    return x -> _cdfData(x)
end

# Finds the quantile of the given input function
function findQuantile(f, α; x0=1.0, order=0)
    return fzero((t -> f(t) - α), x0, order=order)
end

# Finds the quantile of the given input data
function findQuantileData(x, α)
    xs = sort(x)
    pvec = (1:length(xs)) ./ length(xs)
    k = 1
    while pvec[k] < α
        if k == length(xs)
            break
        end
        k += 1
    end
    return xs[k]
end

# Finds the fully connected components in a connection graph given the starting seeds
function findConnComp(G::BitMatrix, starts::Vector{Int64})
    connComps = Vector{Vector{Int64}}(undef, length(starts))
    for (i, n0) in enumerate(starts)
        to_check = findall(G[n0, :])
        visited = [n0]
        while !isempty(to_check)
            node = pop!(to_check)
            if !(node ∈ visited)
                append!(visited, node)
                push!(to_check, findall(G[node, :])...)
            end
        end
        connComps[i] = sort(visited)
    end
    return connComps
end

function createErlangPH(m, p)
    l = p/m
    T = diagm(-l*ones(p)) + diagm(1 => l*ones(p-1)) 
    a = [1; zeros(p-1)]
    return EMpht.PhaseType(a, T)
end