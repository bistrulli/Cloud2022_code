include("src/QueueModelTracking.jl")

## Parameters
if !@isdefined datafolder
    datafolder = "data/"
end
loadgenerators = ["ir", "st"]

# Number of phase states per PH distribution 
phases = Dict("ec"=>1, "m"=>5, "d"=>5)

# Number of processors per service
processors_per_service = 4

# When binning the data, use the following time steps and lag
dt = 0.5
lag = 20.0

# The ratio between y and x panes in subplots
w_ratio = 2/3

# Plotting opts
xpanes = round(Int64, 21*w_ratio)
ypanes = ceil(Int64, 21 / xpanes)

# Use the following simulation for extraction
sim_idx = 1

## Load the CSV data
simfolders = [joinpath(datafolder, p) 
    for p in filter(x -> occursin("sim", x), readdir(datafolder))]
byfunc(x) = parse(Int, split(x, "sim")[end])
sort!(simfolders, by=byfunc)

tracefolders = [joinpath(p, "traces") for p in simfolders]
loadgenfiles = [[joinpath(p, "load_generator_$f.csv") for f in loadgenerators] for p in simfolders]
loadgensettingsfiles = [[joinpath(p, "settings_$f.yaml") for f in loadgenerators] for p in simfolders]

MCsims = length(simfolders)

# Each index in the arrays contains the results from one experiment
printstyled("Extracting data files\n",bold=true, color=:green);
simSettings = Vector{T_simSetting}(undef, MCsims)
for k = 1:MCsims
    simSettings[k] = T_simSetting()
    for (j, lg) in enumerate(loadgenerators)
        simSettings[k][lg] = YAML.load_file(loadgensettingsfiles[k][j]; dicttype=Dict{String, Any}) 
        simSettings[k][lg]["experimentTime"] = unix2datetime.(simSettings[k][lg]["experimentTime"])
    end
end

# The sim settings are mostly identical, only experimentTime, simIntEnd, requests
# are allowed to be different over MCsims in load generators
ss_keys = unique([unique(keys.(values(simSetting))) for simSetting in simSettings])
@assert length(ss_keys[1]) == 1
ss_keys = ss_keys[1][1]
for key in ss_keys
    if !(key ∈ ["experimentTime", "simIntEnd", "requests"])
        for lg in loadgenerators
            @assert length(unique(get.(get.(simSettings, lg, 0), key, rand(1)))) == 1
        end
    end
end

# The number of simulation intervals of different settings are the same. 
simSecDur = unique(get.(values(simSettings[1]), "simSecDur", 0))
@assert length(simSecDur) == 1
simSecDur = Float64.(simSecDur[1])
si = length(simSecDur)

simSecEnd = cumsum(simSecDur)
simSecInt = [ (i-1 == 0 ? 0 : simSecEnd[i-1], simSecEnd[i]) for i = 1:si]

# Create the binned step vector and the buffer to include complete lag
steps = collect(0:dt:maximum(get.(values(simSettings[1]), "simTime", 0)))
buffer = findfirst(steps .>= lag)

experiment_start, _ = getSimExpTimeIntervals(simSettings[1])
_, experiment_end = getSimExpTimeIntervals(simSettings[end])

# Extract the experiment data exported from the load generators and Envoy
dfs_load = readLoadGenData(loadgenfiles, simSettings, simSecDur, plotting=true)
dfs_pods, dfs_err = readTraceData(tracefolders, simSettings, plotting=true)

# Reformulate the enovy tracing data to the Hr data objects
printstyled("Transforming data files\n",bold=true, color=:green);
data_H = traceData2Hr.(dfs_pods)

# Remove simulations with too many missing values
data_sizes = [size.(values(h), 1) for h in data_H]
mean_size = mean(data_sizes)
rel_err = [norm(float(ds) - mean_size, 1) for ds in data_sizes] ./ sum(mean_size)

rm_true = rel_err .> 0.05
if sum(rm_true) > 0
    printstyled("Too many missing values found, removing\n",bold=true, color=:red)
    for i in findall(rm_true)
        println("\tidx: $i rel_err: $(rel_err[i])")
    end
    data_H = data_H[.!rm_true]
    simSettings = simSettings[.!rm_true]
    MCsims = length(data_H)
end

## Extract the queueing network topology
printstyled("Extracting queue network topology\n",bold=true, color=:green);

# Extract classes and queue
classes, ext_arrs,  queue_type = createClasses(data_H[1], simSettings[1])
queues = unique(getindex.(classes, 1))

# Boolean vectors for class containment in network, application 
classInApp = ones(Bool, length(classes))
for (k, class) in enumerate(classes)
    if queue_type[class[1]] == "ec"
        classInApp[k] = 0
    end
end

queueInApp = ones(Bool, length(queues))
queueIsService = zeros(Bool, length(queues))
for (k, queue) in enumerate(queues)
    if queue_type[queue] == "ec"
        queueInApp[k] = 0
    elseif queue_type[queue] == "m"
        queueIsService[k] = 1
    end
end

# Retrieve important queuing network parameters
Cq = [sum([qc == q for (qc, _, _) in classes]) for q in queues]
S = zeros(Int64, length(classes))
queue_disc = Dict{T_qn, String}()
queue_servers = Dict{T_qn, Int}()
for (i, (q, _, _)) = enumerate(classes)
    S[i] = phases[queue_type[q]]
end
for q in queues
    queue_disc[q] = (queue_type[q] == "m" ? "PS" : "INF")
    queue_servers[q] = (queue_type[q] == "m" ? processors_per_service :  typemax(Int64))
end
M, Mc, N = getPhasePos(S, length(queues), Cq)

x_inApp = (n -> n ∈ findall(classInApp)).(N)

# Find chains. Assumed to be the same across all MC simulations
chains_closed = Vector{Vector{Vector{Int64}}}(undef, si)
chains_open = Vector{Vector{Vector{Int64}}}(undef, si)
for i = 1:si
    t0, _ = datetime2unix.(getSimExpTimeIntervals(simSettings[1]))
    H_si = getHInTspan(data_H[1], simSecInt[i] .+ t0)
    connGraph = getClassRoutes(H_si, classes, queue_type) .> 0
   
    closed_starts = findall(.!classInApp)
    open_start = findall(getExternalArrivals(H_si, ext_arrs, classes) .> 0)

    chains_closed[i] = findConnComp(connGraph, closed_starts)
    chains_open[i] = findConnComp(connGraph, open_start)
end

visitedClassesNbr = [sort(unique(vcat([chains_open[i]; chains_closed[i]]...))) 
    for i = 1:si]
visitedClasses = [(c -> c ∈ visitedClassesNbr[i]).(1:length(classes)) for i = 1:si]

## Extract queueing network parameters and variable

printstyled("Creating class arrival and departures\n",bold=true, color=:green);

# Calculate arrival/departure times for each class in each sim. 
# On the form [SIM][CLASS]
ta_class = Matrix{Vector{Float64}}(undef, MCsims, length(classes))
td_class = Matrix{Vector{Float64}}(undef, MCsims, length(classes))
ri_class = Matrix{Vector{Int64}}(undef, MCsims, length(classes))
for i = 1:MCsims
    ta_class[i, :], td_class[i, :], ri_class[i, :] = 
        getArrivalDeparture(data_H[i], simSettings[i], classes, queue_type)
end
@assert all(vcat(td_class...) .>= vcat(ta_class...))
tw_class = td_class - ta_class

function padEnds(t_vec, t0, tf)
    t_vec = (t_vec[1] > t0 ? [t0; t_vec] : t_vec)
    t_vec = (t_vec[end] < tf ? [t_vec; tf] : t_vec)
    return t_vec
end

acceptable_early_err = 0.2 * minimum(diff(simSecEnd))

# Important that we dont have too long streaks of missing values in the beginning
# and the end. Comment out if we want to divert all requests from some service.
#= @assert maximum(minimum.(ta_class)) < acceptable_early_err && 
    maximum(minimum.(td_class)) < acceptable_early_err
@assert abs(minimum(maximum.(ta_class)) - simSecEnd[end]) < acceptable_early_err && 
    abs(minimum(maximum.(td_class)) - simSecEnd[end]) < acceptable_early_err =#

mdiff_ta = maximum(maximum.(diff.(ta_class)))
mdiff_td = maximum(maximum.(diff.(td_class)))
if mdiff_ta > acceptable_early_err || mdiff_td > acceptable_early_err
    printstyled("Large differences\n",bold=true, color=:red)
    println("\tmax diff class arrivals: $mdiff_ta")
    println("\tmax diff class departures: $mdiff_td")
end

printstyled("Extracting requests class paths\n",bold=true, color=:green);
# Extract path of requests over classes
paths_class = Vector{Dict{Int64, DataFrame}}(undef, MCsims)
paths_err = Vector{Dict{Int64, DataFrame}}(undef, MCsims)
for i = 1:MCsims
    paths_class[i], paths_err[i] = getAllPaths(
        ri_class[i, :], ta_class[i, :], td_class[i, :], 
        classes, ext_arrs, queue_type)
end
nbr_p_err = sum(length.(values.(paths_err)))
nbr_p = sum(length.(values.(paths_class)))
if sum(length.(values.(paths_err))) > 0
    printstyled("Path extraction error\n",bold=true, color=:red)
    println("\terr: $(nbr_p_err / (nbr_p_err + nbr_p) * 100) %")
end

printstyled("Creating queue arrival and departures\n",bold=true, color=:green);
# Calculate the arrival/departures times for each queue in each sim.
# On the form [SIM][CLASS]
ta_queue = Matrix{Vector{Float64}}(undef, MCsims, length(queues))
td_queue = Matrix{Vector{Float64}}(undef, MCsims, length(queues))
for i = 1:MCsims
    ta_queue[i, :], td_queue[i, :], _ = 
        joinDataOverQueues(ta_class[i,:], td_class[i,:], classes)
    for k = 1:length(queues)
        @assert sum(length.(ta_class[i, Mc .== k])) == length(ta_queue[i, k])
        @assert sort(vcat(ta_class[i, Mc .== k]...)) ==  sort(ta_queue[i, k])
    end
end

printstyled("Split class/queue arrival/departures over simulation intervals\n",bold=true, color=:green);
# Split the arrivals over the simulation experiment intervals
ta_class_split = Array{Vector{Float64}, 3}(undef, MCsims, length(classes), si)
td_class_split = Array{Vector{Float64}, 3}(undef, MCsims, length(classes), si)
ta_queue_split = Array{Vector{Float64}, 3}(undef, MCsims, length(queues), si)
td_queue_split = Array{Vector{Float64}, 3}(undef, MCsims, length(queues), si)
for i = 1:MCsims
    for j = 1:length(classes)
        ta_class_split[i, j, :], itv = splitByT(ta_class[i, j], [0; simSecEnd])
        for k = 1:si
            td_class_split[i, j, k] = td_class[i, j][itv[k]]
        end
    end
    for j = 1:length(queues)
        ta_queue_split[i, j, :], itv = splitByT(ta_queue[i, j], [0; simSecEnd])
        for k = 1:si
            td_queue_split[i, j, k] = td_queue[i, j][itv[k]]
        end
    end
end

printstyled("Calculating class and queue populations\n",bold=true, color=:green);
# Extract populations for queues and classes
pop_class = getQueueLengths.(ta_class, td_class)
pop_queue = getQueueLengths.(ta_queue, td_queue)

# Check correctness of two different ways of retrieving queue lengths
pop_queue_other = Matrix{Matrix{Float64}}(undef, size(pop_queue))
for sim = 1:MCsims
    for i = 1:length(queues)
        pop_queue_other[sim, i] = addQueueLengths(pop_class[sim, Mc .== i])
    end
end
@assert all(pop_queue .== pop_queue_other)

# Extract populations for queue and classes in each simulation interval
pop_class_split = Array{Matrix{Float64}, 3}(undef, MCsims, length(classes), si)
pop_queue_split = Array{Matrix{Float64}, 3}(undef, MCsims, length(queues), si)
for i = 1:si
    start_time = (i-1 == 0 ? 0 : simSecEnd[i-1])

    if i < si
        end_time = simSecEnd[i]
    else
        end_time = 0
    end
 
    pop_class_split[:,:,i] = getQueueLengths.(ta_class, td_class, start_time=start_time, end_time=end_time)
    pop_queue_split[:,:,i] = getQueueLengths.(ta_queue, td_queue, start_time=start_time, end_time=end_time)
end

# Check correctness of two different ways of retrieving the split queue lengths
pop_queue_split_other = Array{Matrix{Float64}, 3}(undef, size(pop_queue_split))
for sim = 1:MCsims
    for i = 1:length(queues)
        for j = 1:si
            pop_queue_split_other[sim, i, j] = addQueueLengths(
                pop_class_split[sim, Mc .== i, j])
        end
    end
end
@assert all(pop_queue_split .== pop_queue_split_other)

# Calcualte the mean class/queue populations, and assert that the sum of 
# mean class populations is the same as the corresponding mean queue population.
#    First over each simluation interval
pop_class_mean = reshape(mean(getQueueLengthAvg.(pop_class_split), dims=1), length(classes), si)
pop_queue_mean = reshape(mean(getQueueLengthAvg.(pop_queue_split), dims=1), length(queues), si)
@assert norm(vcat([sum(pop_class_mean[Mc .== i, :], dims=1) 
    for i = 1:length(queues)]...) - pop_queue_mean, Inf) < 0.01

printstyled("Calculating utilization and optimal smoothing parameter\n",bold=true, color=:green);
util = zeros(length(queues), si)
p_opt = zeros(length(queues), si)
for (i, q) in enumerate(queues)
    for j = 1:si
        util[i, j] = mean(getUtil.(pop_queue_split[:, i, j], queue_servers[q]))
        p_opt[i, j] = getOptimPNorm(pop_queue_mean[i,j], util[i, j], queue_servers[q])
    end
end

#   Then over each step bin for all simulations
printstyled("Calculating class/queue pop and util over each step\n",bold=true, color=:green);
pop_class_steps_mc, _ = getQueueLengthAvgSteps(pop_class, steps)
pop_queue_steps_mc, util_steps = getQueueLengthAvgSteps(pop_queue, steps, 
    K=[queue_servers[q] for q in queues], D=[queue_disc[q] for q in queues])
for i = 1:length(queues)
    @assert all(isapprox.(sum(pop_class_steps_mc[Mc .== i]), pop_queue_steps_mc[i]))
    @assert all(0 .<= util_steps[i] .<= 1) 
end

## Extract the service times given either an INF or PS(4) model
printstyled("Calculating service times\n",bold=true, color=:green);
ts_class = Array{Array{Float64, 1}, 2}(undef, size(ta_class)...)
for i = 1:size(ta_class, 1)
    ts_class[i, :] = getServiceTimes(ta_class[i,:], td_class[i,:], 
        queue_servers, classes) 
end

ts_class_split = Array{Array{Float64, 1}, 3}(undef, MCsims, length(classes), si)
for i = 1:MCsims
    for j = 1:length(classes)
        _, itv = splitByT(ta_class[i, j], [0; simSecEnd])
        for k = 1:si
            ts_class_split[i, j, k] = ts_class[i, j][itv[k]]
        end
    end
end

ts_class_split_mc = concatMC(ts_class_split)

# Fit PH distributions to classes
printstyled("Fitting PH distributions\n",bold=true, color=:green);
ph_vec = Matrix{EMpht.PhaseType}(undef, length(classes), si)
for (i, (q, _, _)) = enumerate(classes)
    for j = 1:si
        println("Fitting PH dist: $i, $j")
        ph_vec[i, j] = fitPhaseDist(filtOutliers(ts_class_split_mc[i,j], ϵ=0, α=0.99, β=10), 
            phases[queue_type[q]], max_iter=200, verbose=false)
    end
end
