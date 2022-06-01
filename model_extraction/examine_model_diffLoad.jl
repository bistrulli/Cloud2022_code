# Creates the nomial fluid model with the smoothed fluid model 
# optimally fit on all data

include("src/QueueModelTracking.jl")

# Point towards folder gathered from diffload experiment
datafolder = ""

include("examine_preload.jl")

## Extract arrivals/routing

w_a = zeros(length(classes), MCsims, si)
λ = zeros(length(classes), si)
for j = 1:si

    Tw = zeros(length(classes), MCsims)
    for i = 1:MCsims
        t0, _ = datetime2unix.(getSimExpTimeIntervals(simSettings[i]))

        w_a[:, i, j] = getExternalArrivals(getHInTspan(data_H[i], simSecInt[j] .+ t0), 
            ext_arrs, classes)

        Tw[:, i] =  (ta -> isempty(ta) ? 0 : ta[end] - ta[1]).(ta_class_split[i, :, j])
    end

    
    λ[:, j] = mean(map(x -> x > 0 ? x : 0, w_a[:, :, j]) ./ Tw, dims=2)
end
λ[isnan.(λ)] .= 0.0

w = zeros(length(classes), length(classes), si)  
P = zeros(length(classes), length(classes), si)
class_idx_dict = classes2map(classes)
for j = 1:si
    for i = 1:MCsims
        t0, _ = datetime2unix.(getSimExpTimeIntervals(simSettings[i]))
        H_si = getHInTspan(data_H[i], simSecInt[j] .+ t0)
        W_raw = getClassRoutes(H_si, classes, queue_type)
        w[:, :, j] += W_raw
    end

    for (i, (q, n, u)) in enumerate(classes)
        w_tmp = w_a[i] < 0 ? w_a[class_idx_dict[(q, n, 1)]] : 0
        if (sum(w[i, :, j]) + w_tmp) > 0
            P[i, :, j] = w[i, :, j] ./ (sum(w[i, :, j]) + w_tmp)
        end
    end
end

si_idx(t) = (x -> isnothing(x) ? si : x)(findfirst(t .< simSecEnd))

## Run the fluid model

K = [queue_servers[q] for q in queues]


function dx_fluid_min!(dx, x, p, t)
    Ψ, A, B = getStackedPHMatrices(ph_vec[:,si_idx(t)])
    W = Ψ + B*P[:,:,si_idx(t)]*A'
    dx .= W' * (g.(QNorm1(x, M, length(queues)), K[M]) .* x) + 
        sum(A*λ[:,si_idx(t)], dims=2)[:]
end

function dx_fluid_opt!(dx, x, p, t)
    Ψ, A, B = getStackedPHMatrices(ph_vec[:,si_idx(t)])
    W = Ψ + B*P[:,:,si_idx(t)]*A'
    dx .= W' * (g_smooth.(QNorm1(x, M, length(queues)), K[M], 
        p_opt[:, si_idx(t)][M]) .* x) + sum(A*λ[:,si_idx(t)], dims=2)[:]
end

function getPredCoefficents(idx, t)
    ph_pred = ph_vec[:, idx]
    p_pred = p_opt[:, idx]

    ph_pred[.!classInApp] .= ph_vec[.!classInApp, si_idx(t)]

    return ph_pred, p_pred
end

function dx_fluid_pred!(dx, x, p, t)

    ph_pred, p_pred = getPredCoefficents(p[1], t)

    Ψ, A, B = getStackedPHMatrices(ph_pred)

    W = Ψ + B*P[:,:,si_idx(t)]*A'
    ps = g_smooth.(QNorm1(x, M, length(queues)), K[M], p_pred[M])
    dx .= W' * (ps .* x) + sum(A*λ[:,si_idx(t)], dims=2)[:]
end

pred_fit_idxs = [11, 12]

examine_pred = true

alg = lsoda()

x0_c = zeros(length(classes))
x0_c[1] = simSettings[1]["st"]["connections"]
_, A_st, _ = getStackedPHMatrices(ph_vec[:, 1])
x0 = A_st*x0_c

simStart, simEnd = datetime2unix.(getSimExpTimeIntervals(simSettings[1]))
tspan = (0.0, simEnd - simStart)

prob_min = ODEProblem(dx_fluid_min!, x0, tspan)
prob_opt = ODEProblem(dx_fluid_opt!, x0, tspan)
prob_pred = [ODEProblem(dx_fluid_pred!, x0, tspan, [idx]) for idx in pred_fit_idxs]

sol_min = solve(prob_min, alg, saveat=steps) #, reltol=1e-8, abstol=1e-8)
sol_opt = solve(prob_opt, alg, saveat=steps)
sol_pred = [solve(p, alg, saveat=steps) for p in prob_pred]

function extract_queue_vals(sol)
    pop_class = [sum(hcat(sol.u...)[N .== i, :], dims=1)[:] for i in 1:length(classes)]
    pop_queue = [sum(hcat(sol.u...)[M .== i, :], dims=1)[:] for i in 1:length(queues)]

    pop_class_mean = Array{Float64, 2}(undef, length(classes), si)
    pop_queue_mean = Array{Float64, 2}(undef, length(queues), si)

    _, itv = splitByT(sol.t, [0; simSecEnd])

    for j = 1:si
        for i = 1:length(classes)
            pop_class_mean[i, j] = mean([sum(x[N .== i]) for x in sol.u[itv[j]]])
        end
        for i = 1:length(queues)
            pop_queue_mean[i, j] = mean([sum(x[M .== i]) for x in sol.u[itv[j]]])
        end
    end
    return pop_class, pop_queue, pop_class_mean, pop_queue_mean
end

pop_class_fmin, pop_queue_fmin, pop_class_mean_fmin, pop_queue_mean_fmin = 
    extract_queue_vals(sol_min)
pop_class_fopt, pop_queue_fopt, pop_class_mean_fopt, pop_queue_mean_fopt = 
    extract_queue_vals(sol_opt)

pop_class_fpred = []
pop_queue_fpred = []
pop_class_mean_fpred = []
pop_queue_mean_fpred = []
for sol in sol_pred 
    p1, p2, p3, p4 = extract_queue_vals(sol)
    push!(pop_class_fpred, p1)
    push!(pop_queue_fpred, p2)
    push!(pop_class_mean_fpred, p3)
    push!(pop_queue_mean_fpred, p4)
end

# Plot class pop over time
t = collect(tspan[1]:0.1:tspan[2])
ymax = maximum(maximum.(pop_class_steps_mc[classInApp]))
fig = figure(10)
clf()
for (i, (q, _, _)) in enumerate(classes)
    subplot(ypanes, xpanes, i)
    plot(steps, pop_class_steps_mc[i], "C0", label="q mean")
    plot(sol_min.t, pop_class_fmin[i], "C1", label="fluid_min")
    plot(sol_opt.t, pop_class_fopt[i], "C2", label="fluid_opt")
    
    for (idx, pred_idx) in enumerate(pred_fit_idxs)
        plot(sol_pred[idx].t, pop_class_fpred[idx][i], "C$(2+idx)", label="fluid_pred $pred_idx")
    end

    if classInApp[i]
        ylim([0, 1.25*ymax])
    end

    (i == 1) ? legend() : 0
    titlestr = (i==1) ? "class pop. avg\n" : ""

    title(titlestr*queue_type[q]*"$(Mc[i]), c$(i)")
end

# plot queue length over time
t = collect(tspan[1]:0.1:tspan[2])
ymax = maximum(maximum.(pop_queue_steps_mc[queueInApp]))
fig = figure(11)
clf()
for (i, q) in enumerate(queues)
    subplot(1, length(queues), i)

    plot(steps, pop_queue_steps_mc[i], "C0", label="q mean")
    plot(sol_min.t, pop_queue_fmin[i], "C1", label="fluid_min")
    plot(sol_opt.t, pop_queue_fopt[i], "C2", label="fluid_opt")

    for (idx, pred_idx) in enumerate(pred_fit_idxs)
        plot(sol_pred[idx].t, pop_queue_fpred[idx][i], "C$(2+idx)", label="fluid_pred $pred_idx")
    end

    if queueInApp[i]
        ylim([0, 1.25*ymax])
    end

    (i == 1) ? legend() : 0
    titlestr = (i==1) ? "queue pop. avg\n" : ""

    title(titlestr*queue_type[q]*"q$i")
end

# Plot total queue length
t = collect(tspan[1]:0.1:tspan[2])
fig = figure(12)
clf()
plot(steps, sum(pop_class_steps_mc[classInApp]), "C0", label="q mean")
plot(sol_min.t, sum(pop_class_fmin[classInApp]), "C1", label="fluid_min")
plot(sol_opt.t, sum(pop_class_fopt[classInApp]), "C2", label="fluid_opt")

for (idx, pred_idx) in enumerate(pred_fit_idxs)
    plot(sol_pred[idx].t, sum(pop_class_fpred[idx][classInApp]), "C$(2+idx)", label="fluid_pred $pred_idx")
end

legend() 
title("Total avg requests in system")


## Calculate and plot RT mean/quantile over all classes

tw_class_split_mc = filtOutliers.(concatMC(td_class_split - ta_class_split), ϵ=0)

cf_q_data = zeros(length(classes), si, 3+length(pred_fit_idxs), 2) # Service-Class/ simsection / data-sim / mean-quant

do_plot = false

if do_plot
    figure(13)
    clf()
end
for (i, (q, _, _)) in enumerate(classes)
    for j = 1:si
        do_plot ? subplot(length(classes), si, si*(i-1) + j) : 0

        k = Mc[i]

        tmax =  3*quantile(tw_class_split_mc[i, j], 0.99)
        t = collect(range(0, stop=tmax, length=1000))
        
        v_data = cdf_plot_data(t, tw_class_split_mc[i, j], do_plot=do_plot)
        idx_data_m = findfirst(x-> !isnan(x), v_data[:, 2])
        idx_data_a = findfirst(x-> !isnan(x), v_data[:, 3])
        !isnothing(idx_data_m) ? cf_q_data[i, j, 1, 1] = t[idx_data_m] : 0
        !isnothing(idx_data_a) ? cf_q_data[i, j, 1, 2] = t[idx_data_a] : 0
      
        Ψ = ph_vec[i, j].T
        ζ = ph_vec[i, j].π
        ps_fmin = g(pop_queue_mean_fmin[k, j], queue_servers[q])
        cdf_min(t) = tr_cdf(t, Ψ, ps_fmin, ζ)
        v_fmin = cdf_plot(t, cdf_min, color="C1", label="min", do_plot=do_plot)
        idx_min_m = findfirst(x-> !isnan(x), v_fmin[:, 2])
        idx_min_a = findfirst(x-> !isnan(x), v_fmin[:, 3])
        !isnothing(idx_min_m) ? cf_q_data[i, j, 2, 1] = t[idx_min_m] : 0
        !isnothing(idx_min_a) ? cf_q_data[i, j, 2, 2] = t[idx_min_a] : 0

        ps_fopt = g_smooth(pop_queue_mean_fopt[k, j], queue_servers[q], p_opt[k, j])
        cdf_opt(t) = tr_cdf(t, Ψ, ps_fopt, ζ)
        v_fopt = cdf_plot(t, cdf_opt, color="C2", label="opt", do_plot=do_plot)
        idx_opt_m = findfirst(x-> !isnan(x), v_fopt[:, 2])
        idx_opt_a = findfirst(x-> !isnan(x), v_fopt[:, 3])
        !isnothing(idx_opt_m) ? cf_q_data[i, j, 3, 1] = t[idx_opt_m] : 0
        !isnothing(idx_opt_a) ? cf_q_data[i, j, 3, 2] = t[idx_opt_a] : 0

        for (idx, pred_idx) in enumerate(pred_fit_idxs)
            ph_pred, p_pred = getPredCoefficents(pred_idx, (simSecEnd - simSecDur)[j])
            ps_fpred = g_smooth(pop_queue_mean_fpred[idx][k, j], queue_servers[q], 
                p_pred[k])
            cdf_pred(t) = tr_cdf(t, ph_pred[i].T, ps_fpred, ph_pred[i].π)
            v_fpred = cdf_plot(t, cdf_pred, color="C$(2+idx)", label="pred $pred_idx", 
                do_plot=do_plot)
            idx_pred_m = findfirst(x-> !isnan(x), v_fpred[:, 2])
            idx_pred_a = findfirst(x-> !isnan(x), v_fpred[:, 3])
            !isnothing(idx_pred_m) ? cf_q_data[i, j, 3+idx, 1] = t[idx_pred_m] : 0
            !isnothing(idx_pred_a) ? cf_q_data[i, j, 3+idx, 2] = t[idx_pred_a] : 0
        end
        
        xlim([0, tmax])
        ylim([0, 1.25])

        if j == 1
            title("Class RT CDF, $(queue_type[q])$(Mc[i]) c$i")
            if i == 1
                legend()
            end
        end
    end
end

## Calculate and plot RT mean/quantile over Cr

Cri = Array{Array{Bool}}(undef, length([chains_closed[2]]) + length(chains_closed[2]))
Cr = []
for (i, chain) in enumerate([chains_closed[2]; chains_open[2]])
    Cri[i] = zeros(Bool, length(classes))
    Cri[i][filter(c -> classInApp[c], chain)] .= 1
    push!(Cr, classes[Cri[i]])
end


ta_cr = Matrix{Vector{Float64}}(undef, length(Cr), MCsims)
td_cr = Matrix{Vector{Float64}}(undef, length(Cr), MCsims)
ind = Matrix{Any}(undef, length(Cr), MCsims)
for k = 1:length(Cr)
    for i = 1:MCsims
        ta_cr[k, i], td_cr[k, i], _ = getArrivalDeparture(paths_class[i], Cr[k])
        _, ind[k, i] = splitByT(ta_cr[k, i],[0; simSecEnd])
    end
end
tw_cr = td_cr - ta_cr

cf_q_Cr_data = zeros(si, length(Cr), 3+length(pred_fit_idxs), 2)
cdf_data = zeros(1000, 10, si, length(Cr))

do_plot = false

if do_plot
    figure(14)
    clf()
end

# Loop over set in Cri as well.

for j = 1:si
    for k = 1:length(Cr)
        do_plot ? subplot(si, length(Cr), (j-1)*length(Cr) + k) : 0

        println("$k, $j")
        tw_mc = vcat([tw_cr[k, i][ind[k, i][j]] for i = 1:MCsims]...)

        t = collect(range(0, stop=3*quantile(tw_mc, 0.99), length=size(cdf_data, 1)))
        cdf_data[:, 1, j, k] = t

        v_data = cdf_plot_data(t, tw_mc, do_plot=do_plot)
        idx_data_m = findfirst(x-> !isnan(x), v_data[:, 2])
        idx_data_a = findfirst(x-> !isnan(x), v_data[:, 3])
        !isnothing(idx_data_m) ? cf_q_Cr_data[j, k, 1, 1] = t[idx_data_m] : 0
        !isnothing(idx_data_a) ? cf_q_Cr_data[j, k, 1, 2] = t[idx_data_a] : 0
        cdf_data[:, 2:4, j, k] = v_data

        Pb = zeros(length(classes), length(classes))
        Pb[Cri[k], Cri[k]] = P[Cri[k], Cri[k], j]
        Eb = P[.!Cri[k], Cri[k], j]

        Ψ, A, B = getStackedPHMatrices(ph_vec[:, j])
        Wb = (Ψ + B*Pb*A')

        # Generate CDF for standard mean-field fluid model
        ps_fmin = g.(pop_queue_mean_fmin[M, j], K[M])
        x_fmin = sol_min.u[findfirst(sol_min.t .> cumsum(simSecDur)[j]) - 1]
        outflow_fmin = B' * (ps_fmin .* x_fmin)
        inflow_Cr_fmin = Eb' * outflow_fmin[.!Cri[k]] + λ[Cri[k], j]
        β_fmin = zeros(length(classes))
        β_fmin[Cri[k]] = normalize(inflow_Cr_fmin, 1)
        cf_min_Cr(t) = tr_cdf(t, Wb, ps_fmin, A, β_fmin)

        v_min = cdf_plot(t, cf_min_Cr, color="C1", label="min", do_plot=do_plot)
        idx_min_m = findfirst(x-> !isnan(x), v_min[:, 2])
        idx_min_a = findfirst(x-> !isnan(x), v_min[:, 3])
        !isnothing(idx_min_m) ? cf_q_Cr_data[j, k, 2, 1] = t[idx_min_m] : 0
        !isnothing(idx_min_a) ? cf_q_Cr_data[j, k, 2, 2] = t[idx_min_a] : 0

        # Generate CDF for optimal smoothed mean-field fluid model
        ps_fopt = g_smooth.(pop_queue_mean_fopt[M, j], K[M], p_opt[M, j])
        x_fopt = sol_opt.u[findfirst(sol_opt.t .> cumsum(simSecDur)[j]) - 1]
        outflow_fopt = B' * (ps_fopt .* x_fopt)
        inflow_Cr_fopt = Eb' * outflow_fopt[.!Cri[k]] + λ[Cri[k], j]
        β_fopt = zeros(length(classes))
        β_fopt[Cri[k]] = normalize(inflow_Cr_fopt, 1)
        cf_opt_Cr(t) = tr_cdf(t, Wb, ps_fopt, A, β_fopt)

        v_opt = cdf_plot(t, cf_opt_Cr, color="C2", label="opt", do_plot=do_plot)
        cdf_data[:, 5:7, j, k] = v_opt
        idx_opt_m = findfirst(x-> !isnan(x), v_opt[:, 2])
        idx_opt_a = findfirst(x-> !isnan(x), v_opt[:, 3])
        !isnothing(idx_opt_m) ? cf_q_Cr_data[j, k, 3, 1] = t[idx_opt_m] : 0
        !isnothing(idx_opt_a) ? cf_q_Cr_data[j, k, 3, 2] = t[idx_opt_a] : 0

        # Generate CDF for the predictive smoothed mean-field models
        for (idx, pred_idx) in enumerate(pred_fit_idxs)
            ph_pred, p_pred = getPredCoefficents(pred_idx, (simSecEnd - simSecDur)[j])
            Ψ_p, A_p, B_p = getStackedPHMatrices(ph_pred)
            Wb_p = (Ψ_p + B_p*Pb*A_p')
            ps_fpred = g_smooth.(pop_queue_mean_fpred[idx][M, j], K[M], p_pred[M])
            x_fpred = sol_pred[idx].u[findfirst(sol_pred[idx].t .> cumsum(simSecDur)[j]) - 1]
            outflow_fpred = B' * (ps_fpred .* x_fpred)
            inflow_Cr_fpred = Eb' * outflow_fpred[.!Cri[k]] + λ[Cri[k], j]
            β_fpred = zeros(length(classes))
            β_fpred[Cri[k]] = normalize(inflow_Cr_fpred, 1)
            cf_pred_Cr(t) = tr_cdf(t, Wb_p, ps_fpred, A_p, β_fpred)
            
            v_pred = cdf_plot(t, cf_pred_Cr, color="C$(2+idx)", label="pred $pred_idx", 
                do_plot=do_plot)
            cdf_data[:, 8:10, j, k] = v_pred
            idx_pred_m = findfirst(x-> !isnan(x), v_pred[:, 2])
            idx_pred_a = findfirst(x-> !isnan(x), v_pred[:, 3])
            !isnothing(idx_pred_m) ? cf_q_Cr_data[j, k, 3+idx, 1] = t[idx_pred_m] : 0
            !isnothing(idx_pred_a) ? cf_q_Cr_data[j, k, 3+idx, 2] = t[idx_pred_a] : 0
        end

        if j == 1
            ylabel("RT stats over Cr")
            legend()
        end
    end
end


## LL
# effective arrival rate
arr_queue = reshape(sum(length.(ta_queue_split), dims=1), 
    length(queues), si) ./ repeat(simSecDur' * MCsims, length(queues), 1)
arr_class = reshape(sum(length.(ta_class_split), dims=1), 
    length(classes), si) ./ repeat(simSecDur' * MCsims, length(classes), 1)

# response time
tw_queue_mean = reshape(mean(mean.(td_queue_split - ta_queue_split), 
    dims=1), length(queues), si)
tw_class_mean = reshape(mean(mean.(td_class_split - ta_class_split), 
    dims=1), length(classes), si)

#
pop_queue_mean_LL = arr_queue .* tw_queue_mean
pop_class_mean_LL = arr_class .* tw_class_mean

## Present model metrics

isService = [queue_type[q] == "m" for q in queues]

errfnc(x) = abs(x) 

# population errors
pop_class_REmat_opt = errfnc.(pop_class_mean_fopt - pop_class_mean) ./ pop_class_mean
pop_class_AEmat_opt = errfnc.(pop_class_mean_fopt - pop_class_mean)

pop_class_REmat_pred = [errfnc.(p - pop_class_mean) ./ pop_class_mean 
    for p in pop_class_mean_fpred]
pop_class_AEmat_pred = [errfnc.(p - pop_class_mean) for p in pop_class_mean_fpred]

# RT mean
RTmean_REmat_opt = errfnc.(cf_q_data[:, :, 3, 1] - cf_q_data[:, :, 1, 1])./cf_q_data[:, :, 1, 1]
RTmean_REmat_pred = [errfnc.(cf_q_data[:, :, 3+i, 1] - cf_q_data[:, :, 1, 1]) ./ 
    cf_q_data[:, :, 1, 1] for i in 1:length(pred_fit_idxs)]

# RT p95
RTp95_REmat_opt = errfnc.(cf_q_data[:, :, 3, 2] - cf_q_data[:, :, 1, 2])./cf_q_data[:, :, 1, 2]
RTp95_AEmat_opt = errfnc.(cf_q_data[:, :, 3, 2] - cf_q_data[:, :, 1, 2])

RTp95_REmat_pred = [errfnc.(cf_q_data[:, :, 3+i, 2] - cf_q_data[:, :, 1, 2]) ./ 
    cf_q_data[:, :, 1, 2] for i in 1:length(pred_fit_idxs)]
RTp95_AEmat_pred = [errfnc.(cf_q_data[:, :, 3+i, 2] - cf_q_data[:, :, 1, 2]) 
    for i in 1:length(pred_fit_idxs)]

figure(19)
clf()
plot(1:si, mean(util[isService, :], dims=1)', "k--")
plot(1:si, util[isService, :]')

figure(20)
clf()
subplot(3, 2, 1)
maxv = maximum(sum(pop_class_mean[classInApp, :], dims=1)[:])
plot(1:si, sum(pop_class_mean[classInApp, :], dims=1)[:], "C0*-")
plot(1:si, sum(pop_class_mean_fopt[classInApp, :], dims=1)[:], "C2*-")
for i = 1:length(pred_fit_idxs)
    plot(1:si, sum(pop_class_mean_fpred[i][classInApp, :], dims=1)[:], "C$(2+i)*-")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])
subplot(3, 2, 2)
maxv = maximum(maximum.(pop_class_AEmat_pred))
plot(1:si, mean(pop_class_AEmat_opt, dims=1)[:], "C2*-")
plot(1:si, maximum(pop_class_AEmat_opt, dims=1)[:], "C2--")
plot(1:si, minimum(pop_class_AEmat_opt, dims=1)[:], "C2--")
for i = 1:length(pred_fit_idxs)
    plot(1:si, mean(pop_class_AEmat_pred[i], dims=1)[:], "C$(2+i)*-")
    plot(1:si, maximum(pop_class_AEmat_pred[i], dims=1)[:], "C$(2+i)--")
    plot(1:si, minimum(pop_class_AEmat_pred[i], dims=1)[:], "C$(2+i)--")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])
subplot(3, 2, 3)
maxv = maximum(cf_q_Cr_data[:, 1, 1, 2])
plot(1:si, cf_q_Cr_data[:, 1, 1, 2], "C0*-")
plot(1:si, cf_q_Cr_data[:, 1, 3, 2], "C2*-")
for i = 1:length(pred_fit_idxs)
    plot(1:si, cf_q_Cr_data[:, 1, 3+i, 2], "C$(2+i)*-")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])
subplot(3, 2, 4)
maxv = maximum(maximum.((x -> x[Cri[1], :]).(RTp95_AEmat_pred)))
plot(1:si, mean(RTp95_AEmat_opt[Cri[1], :], dims=1)[:], "C2*-")
plot(1:si, maximum(RTp95_AEmat_opt[Cri[1], :], dims=1)[:], "C2--")
plot(1:si, minimum(RTp95_AEmat_opt[Cri[1], :], dims=1)[:], "C2--")
for i = 1:length(pred_fit_idxs)
    plot(1:si, mean(RTp95_AEmat_pred[i][Cri[1], :], dims=1)[:], "C$(2+i)*-")
    plot(1:si, maximum(RTp95_AEmat_pred[i][Cri[1], :], dims=1)[:], "C$(2+i)--")
    plot(1:si, minimum(RTp95_AEmat_pred[i][Cri[1], :], dims=1)[:], "C$(2+i)--")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])
subplot(3, 2, 5)
maxv = maximum(cf_q_Cr_data[:, 2, 1, 2])
plot(1:si, cf_q_Cr_data[:, 2, 1, 2], "C0*-")
plot(1:si, cf_q_Cr_data[:, 2, 3, 2], "C2*-")
for i = 1:length(pred_fit_idxs)
    plot(1:si, cf_q_Cr_data[:, 2, 3+i, 2], "C$(2+i)*-")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])
subplot(3, 2, 6)
maxv = maximum(maximum.((x -> x[Cri[2], :]).(RTp95_AEmat_pred)))
plot(1:si, mean(RTp95_AEmat_opt[Cri[2], :], dims=1)[:], "C2*-")
plot(1:si, maximum(RTp95_AEmat_opt[Cri[2], :], dims=1)[:], "C2--")
plot(1:si, minimum(RTp95_AEmat_opt[Cri[2], :], dims=1)[:], "C2--")
for i = 1:length(pred_fit_idxs)
    plot(1:si, mean(RTp95_AEmat_pred[i][Cri[2], :], dims=1)[:], "C$(2+i)*-")
    plot(1:si, maximum(RTp95_AEmat_pred[i][Cri[2], :], dims=1)[:], "C$(2+i)--")
    plot(1:si, minimum(RTp95_AEmat_pred[i][Cri[2], :], dims=1)[:], "C$(2+i)--")
end
#plot([pred_fit_idx, pred_fit_idx], [0, maxv/2], "k--")
ylim([0, 1.1*maxv])

## print utils and rates

digs=3
u = map(x -> length(x) == digs+2 ? x[2:end] : x[2:end] * "0", 
    string.(round.(util[isService, :]', digits=digs)))

v = u[:, 3]
u[:, 3] = u[:, 4]
u[:, 4] = v

for i = 1:16
    for j = 1:4
        print("$(u[i, j])")
    end
    println("")
end

λ_m = zeros(4)
ec_m = zeros(4)
for k = 1:4
    λ_m[k] = mean([λ[8, (i-1)*4 + k] for i = 1:4])
    ec_m[k] = mean(mean.(ph_vec[1, (4*(k-1) + 1):(4*(k-1) + 4)]))
end




## Save to csv data

using DelimitedFiles

headers_1 = permutedims(["X", "X-", "X+", "utilb1", "utilb2", "utilf", "utils",
    "totalQlData", "totalQlOpt", "totalQlPred1", "totalQlPred2",
    "closedP95Data", "closedP95Opt", "closedP95Pred1", "closedP95Pred2",
    "openP95Data", "openP95Opt", "openP95Pred1", "openP95Pred2",
    "AEQlMeanOpt", "AEQlMaxOpt", "AEQlMinOpt", "AEQlMeanPred1", "AEQlMaxPred1", "AEQlMinPred1",
    "AEQlMeanPred2", "AEQlMaxPred2", "AEQlMinPred2",
    "AEP95ClosedMeanOpt", "AEP95ClosedMaxOpt", "AEP95ClosedMinOpt", "AEP95ClosedMeanPred1", "AEP95ClosedMaxPred1", "AEP95ClosedMinPred1",
    "AEP95ClosedMeanPred2", "AEP95ClosedMaxPred2", "AEP95ClosedMinPred2",
    "AEP95OpenMeanOpt", "AEP95OpenMaxOpt", "AEP95OpenMinOpt", "AEP95OpenMeanPred1", "AEP95OpenMaxPred1", "AEP95OpenMinPred1",
    "AEP95OpenMeanPred2", "AEP95OpenMaxPred2", "AEP95OpenMinPred2"])

open("paper_data/extracting_experiment_diffload.csv", "w") do f
    writedlm(f, headers_1, ",")
    writedlm(f, hcat(1:16, (1:16) .- 0.2, (1:16) .+ 0.2,
        util[isService, :]',
        sum(pop_class_mean[classInApp, :], dims=1)[:], 
        sum(pop_class_mean_fopt[classInApp, :], dims=1)[:],
        sum(pop_class_mean_fpred[1][classInApp, :], dims=1)[:],
        sum(pop_class_mean_fpred[2][classInApp, :], dims=1)[:],
        cf_q_Cr_data[:, 1, 1, 2],
        cf_q_Cr_data[:, 1, 3, 2],
        cf_q_Cr_data[:, 1, 4, 2],
        cf_q_Cr_data[:, 1, 5, 2],
        cf_q_Cr_data[:, 2, 1, 2],
        cf_q_Cr_data[:, 2, 3, 2],
        cf_q_Cr_data[:, 2, 4, 2],
        cf_q_Cr_data[:, 2, 5, 2],
        mean(pop_class_AEmat_opt, dims=1)[:],
        maximum(pop_class_AEmat_opt, dims=1)[:] - mean(pop_class_AEmat_opt, dims=1)[:],
        -minimum(pop_class_AEmat_opt, dims=1)[:] + mean(pop_class_AEmat_opt, dims=1)[:],
        mean(pop_class_AEmat_pred[1], dims=1)[:],
        maximum(pop_class_AEmat_pred[1], dims=1)[:] - mean(pop_class_AEmat_pred[1], dims=1)[:],
        -minimum(pop_class_AEmat_pred[1], dims=1)[:] + mean(pop_class_AEmat_pred[1], dims=1)[:],
        mean(pop_class_AEmat_pred[2], dims=1)[:],
        maximum(pop_class_AEmat_pred[2], dims=1)[:] - mean(pop_class_AEmat_pred[2], dims=1)[:],
        -minimum(pop_class_AEmat_pred[2], dims=1)[:] + mean(pop_class_AEmat_pred[2], dims=1)[:],
        mean(RTp95_AEmat_opt[Cri[1], :], dims=1)[:],
        maximum(RTp95_AEmat_opt[Cri[1], :], dims=1)[:] - mean(RTp95_AEmat_opt[Cri[1], :], dims=1)[:],
        -minimum(RTp95_AEmat_opt[Cri[1], :], dims=1)[:] + mean(RTp95_AEmat_opt[Cri[1], :], dims=1)[:], 
        mean(RTp95_AEmat_pred[1][Cri[1], :], dims=1)[:],
        maximum(RTp95_AEmat_pred[1][Cri[1], :], dims=1)[:] - mean(RTp95_AEmat_pred[1][Cri[1], :], dims=1)[:],
        -minimum(RTp95_AEmat_pred[1][Cri[1], :], dims=1)[:] + mean(RTp95_AEmat_pred[1][Cri[1], :], dims=1)[:],
        mean(RTp95_AEmat_pred[2][Cri[1], :], dims=1)[:],
        maximum(RTp95_AEmat_pred[2][Cri[1], :], dims=1)[:] - mean(RTp95_AEmat_pred[2][Cri[1], :], dims=1)[:],
        -minimum(RTp95_AEmat_pred[2][Cri[1], :], dims=1)[:] + mean(RTp95_AEmat_pred[2][Cri[1], :], dims=1)[:],
        mean(RTp95_AEmat_opt[Cri[2], :], dims=1)[:],
        maximum(RTp95_AEmat_opt[Cri[2], :], dims=1)[:] - mean(RTp95_AEmat_opt[Cri[2], :], dims=1)[:],
        -minimum(RTp95_AEmat_opt[Cri[2], :], dims=1)[:] + mean(RTp95_AEmat_opt[Cri[2], :], dims=1)[:], 
        mean(RTp95_AEmat_pred[1][Cri[2], :], dims=1)[:],
        maximum(RTp95_AEmat_pred[1][Cri[2], :], dims=1)[:] - mean(RTp95_AEmat_pred[1][Cri[2], :], dims=1)[:],
        -minimum(RTp95_AEmat_pred[1][Cri[2], :], dims=1)[:] + mean(RTp95_AEmat_pred[1][Cri[2], :], dims=1)[:],
        mean(RTp95_AEmat_pred[2][Cri[2], :], dims=1)[:],
        maximum(RTp95_AEmat_pred[2][Cri[2], :], dims=1)[:] - mean(RTp95_AEmat_pred[2][Cri[2], :], dims=1)[:],
        -minimum(RTp95_AEmat_pred[2][Cri[2], :], dims=1)[:] + mean(RTp95_AEmat_pred[2][Cri[2], :], dims=1)[:]), ",")
end