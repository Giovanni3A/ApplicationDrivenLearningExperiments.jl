# EDA over hybrid_storage results: per-architecture training health,
# residual / shift biases, and representative-sample drill-downs.

using Pkg
cur_dir = @__DIR__
Pkg.activate(dirname(dirname(cur_dir)))

using CSV
using DataFrames
using Statistics
using JuMP
import Gurobi
using Flux
import NNlib
import JLD2
import Plots
using StatsPlots: groupedbar

import ApplicationDrivenLearning
import ParametricOptInterface as POI
import MathOptInterface as MOI

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/model.jl")

ADL = ApplicationDrivenLearning

# ---------- Setup ----------

const ARCHS = (:uni, :type, :many)
const FIG_DIR = joinpath(result_path, "figures")
isdir(FIG_DIR) || mkpath(FIG_DIR)

arch_csv(arch, kind) =
    joinpath(result_path, "$(kind)_$(String(arch)).csv")

state_path(arch, stage) = joinpath(
    result_path,
    (stage == :ls ? "pretrain_" : "gradient_") * "$(String(arch)).jld2",
)

println("Result path: $result_path")
println("Figure dir:  $FIG_DIR")

# ===================================================================
# Section 1 — Did training work?
# ===================================================================

println("\n========== Section 1: training health summary ==========\n")

summary_dfs = Dict{Symbol,DataFrame}()
for arch in ARCHS
    summary_dfs[arch] = CSV.read(arch_csv(arch, "results_summary"), DataFrame)
end

summary_wide = DataFrame(
    Architecture = String[],
    LS_Train_Cost = Float64[],
    GD_Train_Cost = Float64[],
    DCost_Train = Float64[],
    LS_Test_Cost = Float64[],
    GD_Test_Cost = Float64[],
    DCost_Test = Float64[],
    LS_Test_G_RMSE = Float64[],
    GD_Test_G_RMSE = Float64[],
    DG_RMSE_Test = Float64[],
    LS_Test_Pi_RMSE = Float64[],
    GD_Test_Pi_RMSE = Float64[],
    DPi_RMSE_Test = Float64[],
)

for arch in ARCHS
    df = summary_dfs[arch]
    ls = df[df.Model.=="Pretrained", :][1, :]
    gd = df[df.Model.=="ADL Gradient", :][1, :]
    push!(
        summary_wide,
        (
            String(arch),
            ls.Train_Cost,
            gd.Train_Cost,
            gd.Train_Cost - ls.Train_Cost,
            ls.Test_Cost,
            gd.Test_Cost,
            gd.Test_Cost - ls.Test_Cost,
            ls.Test_G_RMSE,
            gd.Test_G_RMSE,
            gd.Test_G_RMSE - ls.Test_G_RMSE,
            ls.Test_Pi_RMSE,
            gd.Test_Pi_RMSE,
            gd.Test_Pi_RMSE - ls.Test_Pi_RMSE,
        ),
    )
end

println("Costs are minimised, so ΔCost < 0 means ADL improved over LS:")
show(summary_wide; allcols = true, allrows = true)
println()
println()

CSV.write(joinpath(result_path, "eda_summary_costs.csv"), summary_wide)

println("\nPer-architecture verdict:")
for row in eachrow(summary_wide)
    train_dir = row.DCost_Train < 0 ? "improved" : "worsened"
    test_dir = row.DCost_Test < 0 ? "improved" : "worsened"
    println(
        "  ",
        row.Architecture,
        ": ADL ", train_dir, " train cost by \$",
        round(abs(row.DCost_Train); digits = 1),
        ", ", test_dir, " test cost by \$",
        round(abs(row.DCost_Test); digits = 1),
        "  (test ΔG RMSE = ", round(row.DG_RMSE_Test; digits = 2),
        ", ΔPi RMSE = ", round(row.DPi_RMSE_Test; digits = 2),
        ")",
    )
end

# Bar plot: test cost LS vs GD per architecture
let
    archs_lbl = [String(a) for a in ARCHS]
    ls_test = [summary_wide[summary_wide.Architecture.==a, :].LS_Test_Cost[1] for a in archs_lbl]
    gd_test = [summary_wide[summary_wide.Architecture.==a, :].GD_Test_Cost[1] for a in archs_lbl]
    fig = groupedbar(
        archs_lbl,
        hcat(ls_test, gd_test);
        labels = ["LS" "GD"],
        xlabel = "Architecture",
        ylabel = "Test LP cost (\$)",
        title = "Test cost: pretrained (LS) vs ADL (GD)",
        legend = :topright,
        bar_width = 0.7,
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "costs_bar.png"))
end

# ===================================================================
# Section 2 — Residuals & architecture bias
# ===================================================================

println("\n========== Section 2: residual bias (test set) ==========\n")

forecasts_test = Dict{Symbol,DataFrame}()
for arch in ARCHS
    local d = CSV.read(arch_csv(arch, "forecasts_test"), DataFrame)
    d.r_G_LS = d.G_LS .- d.G_actual
    d.r_G_GD = d.G_GD .- d.G_actual
    d.r_Pi_LS = d.Pi_LS .- d.Pi_actual
    d.r_Pi_GD = d.Pi_GD .- d.Pi_actual
    forecasts_test[arch] = d
end

# Aggregate residual stats per (project, hour) per arch per stage
residual_long = DataFrame(
    Architecture = String[],
    Quantity = String[],
    Stage = String[],
    Project = Int[],
    Hour = Int[],
    MeanBias = Float64[],
    StdResid = Float64[],
    MeanAbsResid = Float64[],
)
for arch in ARCHS
    local d = forecasts_test[arch]
    agg = combine(
        groupby(d, [:project, :hour]),
        :r_G_LS => mean => :mean_G_LS,
        :r_G_LS => std => :std_G_LS,
        :r_G_LS => (x -> mean(abs.(x))) => :mae_G_LS,
        :r_G_GD => mean => :mean_G_GD,
        :r_G_GD => std => :std_G_GD,
        :r_G_GD => (x -> mean(abs.(x))) => :mae_G_GD,
        :r_Pi_LS => mean => :mean_Pi_LS,
        :r_Pi_LS => std => :std_Pi_LS,
        :r_Pi_LS => (x -> mean(abs.(x))) => :mae_Pi_LS,
        :r_Pi_GD => mean => :mean_Pi_GD,
        :r_Pi_GD => std => :std_Pi_GD,
        :r_Pi_GD => (x -> mean(abs.(x))) => :mae_Pi_GD,
    )
    for r in eachrow(agg)
        for (qty, stg, mcol, scol, acol) in (
            ("G", "LS", :mean_G_LS, :std_G_LS, :mae_G_LS),
            ("G", "GD", :mean_G_GD, :std_G_GD, :mae_G_GD),
            ("Pi", "LS", :mean_Pi_LS, :std_Pi_LS, :mae_Pi_LS),
            ("Pi", "GD", :mean_Pi_GD, :std_Pi_GD, :mae_Pi_GD),
        )
            push!(
                residual_long,
                (
                    String(arch),
                    qty,
                    stg,
                    r.project,
                    r.hour,
                    r[mcol],
                    r[scol],
                    r[acol],
                ),
            )
        end
    end
end
CSV.write(
    joinpath(result_path, "eda_residuals_by_project_hour.csv"),
    residual_long,
)

# Plot mean bias by hour: 2 rows (G, π) × N_PROJ cols
let
    project_names = String.(PROJECT_IDS)
    panels = []
    arch_color = Dict(:uni => :steelblue, :type => :darkorange, :many => :seagreen)
    for (qty_idx, (qty_label, qty_str)) in
        enumerate((("G [MW]", "G"), ("π [\$/MWh]", "Pi")))
        for p = 1:N_PROJ
            sub =
                filter(r -> r.Quantity == qty_str && r.Project == p, residual_long)
            plt = Plots.plot(
                xlabel = "Hour",
                ylabel = "Mean residual ($qty_label)",
                title = "Project $p ($(project_names[p]))",
                legend = (qty_idx == 1 && p == N_PROJ) ? :outerright : nothing,
            )
            for arch in ARCHS, stage in ("LS", "GD")
                d = filter(
                    r -> r.Architecture == String(arch) && r.Stage == stage,
                    sub,
                )
                sort!(d, :Hour)
                Plots.plot!(
                    plt,
                    d.Hour,
                    d.MeanBias;
                    label = "$(String(arch))-$stage",
                    color = arch_color[arch],
                    linestyle = stage == "LS" ? :dash : :solid,
                    linewidth = 1.5,
                )
            end
            Plots.hline!(plt, [0]; color = :black, linestyle = :dot, label = "")
            push!(panels, plt)
        end
    end
    fig = Plots.plot(
        panels...;
        layout = (2, N_PROJ),
        size = (1700, 800),
        plot_title = "Mean residual (forecast − actual) by hour — test",
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "bias_by_hour_test.png"))
end

# Calibration: forecast vs actual binned-mean per project, LS stage primary
let
    panels = []
    project_names = String.(PROJECT_IDS)
    for (qty_label, qty_actual, qty_LS, qty_GD) in (
        ("G [MW]", :G_actual, :G_LS, :G_GD),
        ("π [\$/MWh]", :Pi_actual, :Pi_LS, :Pi_GD),
    )
        for p = 1:N_PROJ
            plt = Plots.plot(
                xlabel = "Actual $qty_label",
                ylabel = "Forecast $qty_label",
                title = "Project $p ($(project_names[p]))",
                legend = (qty_label == "G [MW]" && p == N_PROJ) ? :bottomright : nothing,
            )
            # combine archs, take LS first
            all_act = Float64[]
            arch_color = Dict(:uni => :steelblue, :type => :darkorange, :many => :seagreen)
            for arch in ARCHS
                df_p = filter(r -> r.project == p, forecasts_test[arch])
                act = df_p[!, qty_actual]
                append!(all_act, act)
                ls = df_p[!, qty_LS]
                gd = df_p[!, qty_GD]
                bins = range(minimum(act), maximum(act); length = 16)
                centers = (bins[1:end-1] .+ bins[2:end]) ./ 2
                ls_mean = Float64[]
                gd_mean = Float64[]
                for i = 1:length(bins) - 1
                    mask = (act .>= bins[i]) .& (act .< bins[i+1])
                    if any(mask)
                        push!(ls_mean, mean(ls[mask]))
                        push!(gd_mean, mean(gd[mask]))
                    else
                        push!(ls_mean, NaN)
                        push!(gd_mean, NaN)
                    end
                end
                Plots.plot!(
                    plt,
                    centers,
                    ls_mean;
                    label = "$(String(arch))-LS",
                    color = arch_color[arch],
                    linestyle = :dash,
                )
                Plots.plot!(
                    plt,
                    centers,
                    gd_mean;
                    label = "$(String(arch))-GD",
                    color = arch_color[arch],
                    linestyle = :solid,
                )
            end
            xs = sort(unique(all_act))
            Plots.plot!(
                plt,
                xs,
                xs;
                color = :black,
                linestyle = :dot,
                label = "y = x",
            )
            push!(panels, plt)
        end
    end
    fig = Plots.plot(
        panels...;
        layout = (2, N_PROJ),
        size = (1700, 800),
        plot_title = "Calibration (binned forecast mean vs actual) — test",
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "calibration_test.png"))
end

# Residual distribution: histogram (density=true) per project, faceted by arch, LS and GD overlaid
let
    panels = []
    project_names = String.(PROJECT_IDS)
    for (qty_label, qty_LS_resid, qty_GD_resid) in (
        ("G residual [MW]", :r_G_LS, :r_G_GD),
        ("π residual [\$/MWh]", :r_Pi_LS, :r_Pi_GD),
    )
        for arch in ARCHS
            for p = 1:N_PROJ
                df_p = filter(r -> r.project == p, forecasts_test[arch])
                ls = df_p[!, qty_LS_resid]
                gd = df_p[!, qty_GD_resid]
                # symmetric x-range from data
                xmax = quantile(vcat(abs.(ls), abs.(gd)), 0.99)
                plt = Plots.histogram(
                    ls;
                    bins = 60,
                    label = "LS",
                    alpha = 0.45,
                    color = :steelblue,
                    normalize = :pdf,
                    xlims = (-xmax, xmax),
                    xlabel = "$qty_label ($(String(arch)), proj $p)",
                    ylabel = "density",
                    legend = (qty_label == "G residual [MW]" && arch == :uni && p == 1) ? :topright : nothing,
                )
                Plots.histogram!(
                    plt,
                    gd;
                    bins = 60,
                    label = "GD",
                    alpha = 0.45,
                    color = :darkorange,
                    normalize = :pdf,
                )
                Plots.vline!(plt, [0]; color = :black, linestyle = :dot, label = "")
                push!(panels, plt)
            end
        end
    end
    # 6 panels per quantity (3 archs × 3 projects), 2 quantities → 18 panels total
    fig = Plots.plot(
        panels...;
        layout = (2, 9),
        size = (2400, 600),
        plot_title = "Residual distributions — test",
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "residual_distribution_test.png"))
end

println("Section 2: residual stats CSV + 3 figures written.")

# ===================================================================
# Section 3 — ADL-induced shifts (GD vs LS)
# ===================================================================

println("\n========== Section 3: ADL shifts (GD − LS) ==========\n")

shift_long = DataFrame(
    Architecture = String[],
    Project = Int[],
    Hour = Int[],
    Mean_dG = Float64[],
    Std_dG = Float64[],
    Mean_dPi = Float64[],
    Std_dPi = Float64[],
)
for arch in ARCHS
    local d = forecasts_test[arch]
    d.d_G = d.G_GD .- d.G_LS
    d.d_Pi = d.Pi_GD .- d.Pi_LS
    agg = combine(
        groupby(d, [:project, :hour]),
        :d_G => mean => :mean_dG,
        :d_G => std => :std_dG,
        :d_Pi => mean => :mean_dPi,
        :d_Pi => std => :std_dPi,
    )
    for r in eachrow(agg)
        push!(
            shift_long,
            (
                String(arch),
                r.project,
                r.hour,
                r.mean_dG,
                r.std_dG,
                r.mean_dPi,
                r.std_dPi,
            ),
        )
    end
end
CSV.write(
    joinpath(result_path, "eda_adl_shift_by_project_hour.csv"),
    shift_long,
)

# Heatmap: rows = (G, π) × (project), cols = hour, cells = mean Δ
let
    panels = []
    project_names = String.(PROJECT_IDS)
    for (qty_label, mean_col) in
        (("Δ G [MW]", :Mean_dG), ("Δ π [\$/MWh]", :Mean_dPi))
        for arch in ARCHS
            mat = zeros(N_PROJ, H)
            for p = 1:N_PROJ, h = 1:H
                d = filter(
                    r ->
                        r.Architecture == String(arch) &&
                            r.Project == p &&
                            r.Hour == h,
                    shift_long,
                )
                mat[p, h] = d[1, mean_col]
            end
            absmax = max(maximum(abs.(mat)), 1e-6)
            plt = Plots.heatmap(
                1:H,
                project_names,
                mat;
                xlabel = "Hour",
                ylabel = "Project",
                title = "$(String(arch)): $qty_label",
                clims = (-absmax, absmax),
                color = Plots.cgrad([:steelblue, :white, :darkorange]),
            )
            push!(panels, plt)
        end
    end
    fig = Plots.plot(
        panels...;
        layout = (2, 3),
        size = (1800, 700),
        plot_title = "Mean ADL shift (GD − LS) per project × hour — test",
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "adl_shift_heatmap.png"))
end

# Line plot: mean Δ vs hour, lines per (arch, project)
let
    panels = []
    project_names = String.(PROJECT_IDS)
    arch_color = Dict(:uni => :steelblue, :type => :darkorange, :many => :seagreen)
    for (qty_label, mean_col) in
        (("Δ G [MW]", :Mean_dG), ("Δ π [\$/MWh]", :Mean_dPi))
        for p = 1:N_PROJ
            plt = Plots.plot(
                xlabel = "Hour",
                ylabel = qty_label,
                title = "Project $p ($(project_names[p]))",
                legend = (qty_label == "Δ G [MW]" && p == N_PROJ) ? :outerright : nothing,
            )
            for arch in ARCHS
                d = filter(
                    r -> r.Architecture == String(arch) && r.Project == p,
                    shift_long,
                )
                sort!(d, :Hour)
                Plots.plot!(
                    plt,
                    d.Hour,
                    d[!, mean_col];
                    label = String(arch),
                    color = arch_color[arch],
                    linewidth = 2,
                )
            end
            Plots.hline!(plt, [0]; color = :black, linestyle = :dot, label = "")
            push!(panels, plt)
        end
    end
    fig = Plots.plot(
        panels...;
        layout = (2, N_PROJ),
        size = (1700, 800),
        plot_title = "Mean ADL shift vs hour — test",
    )
    Plots.savefig(fig, joinpath(FIG_DIR, "adl_shift_by_hour.png"))
end

println("Section 3: shift CSV + 2 figures written.")

# ===================================================================
# Section 4A — Forecast-shift shortlist (cheap, CSV-only)
# ===================================================================

println("\n========== Section 4A: shift-magnitude shortlist ==========\n")

# Per-project std normalisation (for the composite shift score)
proj_std_G = Float64[std(filter(r -> r.project == p, forecasts_test[:uni]).G_actual) for p = 1:N_PROJ]
proj_std_Pi = Float64[std(filter(r -> r.project == p, forecasts_test[:uni]).Pi_actual) for p = 1:N_PROJ]

# Build composite shift score S(t,p) per architecture
shortlist_per_arch = Dict{Symbol,DataFrame}()
for arch in ARCHS
    local d = forecasts_test[arch]
    d.score = abs.(d.d_G) ./ proj_std_G[d.project] .+ abs.(d.d_Pi) ./ proj_std_Pi[d.project]
    agg = combine(
        groupby(d, [:t, :project]),
        :score => sum => :total_shift,
    )
    sort!(agg, :total_shift; rev = true)
    shortlist_per_arch[arch] = first(agg, 30)
end

# Union of (arch-agnostic) t values to re-solve
shortlist_ts = sort(unique(vcat([df.t for df in values(shortlist_per_arch)]...)))
println("Total distinct t to re-solve: $(length(shortlist_ts)) (3 archs × 2 stages each)")

# ===================================================================
# Section 4B — Targeted LP re-solve on the shortlist
# ===================================================================

println("\n========== Section 4B: LP re-solve on shortlist ==========\n")

# Build per-arch PredictiveModel from saved states. We mirror the network
# structures in pretrain_{uni,type,many}.jl exactly so loadmodel! finds a match.

function _build_uni_chain()
    if N_HIDDEN_LAYERS == 0
        return Flux.Chain(
            Flux.Dense(length(feature_names) => length(target_names)),
            concat_z_activation,
        )
    end
    layers = Any[
        Flux.Dense(length(feature_names) => HIDDEN_SIZE),
        Flux.relu,
    ]
    for _ = 1:(N_HIDDEN_LAYERS - 1)
        push!(layers, Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE))
        push!(layers, Flux.relu)
    end
    push!(layers, Flux.Dense(HIDDEN_SIZE => length(target_names)))
    push!(layers, concat_z_activation)
    return Flux.Chain(layers...)
end

function _build_subnet_generic(in_sz, out_sz, act)
    if N_HIDDEN_LAYERS == 0
        return Flux.Chain(Flux.Dense(in_sz => out_sz), act)
    end
    layers = Any[Flux.Dense(in_sz => HIDDEN_SIZE), Flux.relu]
    for _ = 1:(N_HIDDEN_LAYERS - 1)
        push!(layers, Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE))
        push!(layers, Flux.relu)
    end
    push!(layers, Flux.Dense(HIDDEN_SIZE => out_sz))
    push!(layers, act)
    return Flux.Chain(layers...)
end

function _build_type_nns()
    return [
        _build_subnet_generic(
            length(g_feat_idx),
            length(_G_TARGET_IDX),
            g_output_activation,
        ),
        _build_subnet_generic(
            length(pi_feat_idx),
            length(_PI_TARGET_IDX),
            pi_output_activation,
        ),
    ]
end

function _build_many_nns()
    return [
        _build_subnet_generic(
            length(g_feat_idx_per_proj[1]),
            H,
            g_output_activation_many,
        ),
        _build_subnet_generic(
            length(pi_feat_idx_per_proj[1]),
            H,
            pi_output_activation_many,
        ),
    ]
end

function _build_iom(arch)
    iom = Vector{Dict{Vector{Int64},Vector{ADL.Forecast}}}()
    if arch == :type
        push!(
            iom,
            Dict(g_feat_idx => vec([G_fc[h, p] for h = 1:H, p = 1:N_PROJ])),
        )
        push!(
            iom,
            Dict(pi_feat_idx => vec([pi_fc[h, p] for h = 1:H, p = 1:N_PROJ])),
        )
    elseif arch == :many
        g_dict = Dict{Vector{Int64},Vector{ADL.Forecast}}()
        pi_dict = Dict{Vector{Int64},Vector{ADL.Forecast}}()
        for p = 1:N_PROJ
            g_dict[g_feat_idx_per_proj[p]] = vec(G_fc[:, p])
            pi_dict[pi_feat_idx_per_proj[p]] = vec(pi_fc[:, p])
        end
        push!(iom, g_dict)
        push!(iom, pi_dict)
    end
    return iom
end

function _load_pred_model(arch, stage)
    st = JLD2.load(state_path(arch, stage), "state")
    if arch == :uni
        nn = _build_uni_chain()
        Flux.loadmodel!(nn, st isa AbstractVector ? st[1] : st)
        return ADL.PredictiveModel(nn)
    else
        nns = arch == :type ? _build_type_nns() : _build_many_nns()
        Flux.loadmodel!(nns, st)
        return ADL.PredictiveModel(nns, _build_iom(arch))
    end
end

# Per-sample LP cost: solve Plan with forecast, then Assess with actuals.
# Returns (assess_cost, plan_charge_vals, plan_discharge_vals, assess_sale_vals, assess_soc_vals).
function _solve_single_sample(t::Int)
    yhat = model.forecast(X_test[t, :])
    MOI.set.(
        model.plan,
        POI.ParameterValue(),
        model.plan_forecast_params,
        yhat,
    )
    JuMP.optimize!(ADL.Plan(model))
    pc = JuMP.value.(plan_charge)
    pd = JuMP.value.(plan_discharge)

    y_actual = ADL.dict_to_var_indexed_matrix(Y_dict_test, model.forecast_vars)[t, :]
    fix.(ADL.assess_forecast_vars(model), y_actual; force = true)
    set_normalized_rhs.(
        model.assess[:assess_policy_fix],
        JuMP.value.(ADL.plan_policy_vars(model)),
    )
    JuMP.optimize!(ADL.Assess(model))
    cost = JuMP.objective_value(ADL.Assess(model))
    asale = JuMP.value.(assess_sale)
    asoc = JuMP.value.(assess_soc)
    return cost, pc, pd, asale, asoc
end

# Storage of (arch, stage, t) → (cost, charge, discharge, sale, soc)
SolveKey = Tuple{Symbol,Symbol,Int}
solved = Dict{SolveKey,NamedTuple}()

for arch in ARCHS
    for stage in (:ls, :gd)
        println("Re-solving $(arch)/$(stage) for $(length(shortlist_ts)) samples...")
        pm = _load_pred_model(arch, stage)
        ADL.set_forecast_model(model, pm)
        for t in shortlist_ts
            cost, pc, pd, asale, asoc = _solve_single_sample(t)
            solved[(arch, stage, t)] = (
                cost = cost,
                charge = pc,
                discharge = pd,
                sale = asale,
                soc = asoc,
            )
        end
    end
end

# Per-(arch, t) compute ADL improvement = LS cost − GD cost (positive = ADL won)
example_df = DataFrame(
    Architecture = String[],
    t = Int[],
    LS_Cost = Float64[],
    GD_Cost = Float64[],
    ADL_Improvement = Float64[],
)
for arch in ARCHS
    for t in shortlist_ts
        ls_cost = solved[(arch, :ls, t)].cost
        gd_cost = solved[(arch, :gd, t)].cost
        push!(
            example_df,
            (String(arch), t, ls_cost, gd_cost, ls_cost - gd_cost),
        )
    end
end
sort!(example_df, :ADL_Improvement; rev = true)
CSV.write(joinpath(result_path, "eda_examples_summary.csv"), example_df)

# Pick best / median / worst per architecture (and a representative project)
function _project_with_largest_shift(arch, t)
    df = forecasts_test[arch]
    sub = filter(r -> r.t == t, df)
    proj_scores =
        combine(groupby(sub, :project), :score => sum => :s)
    return proj_scores.project[argmax(proj_scores.s)]
end

function _plot_example(arch, t, p, label)
    df_arch = forecasts_test[arch]
    row_mask = (df_arch.t .== t) .& (df_arch.project .== p)
    sub = df_arch[row_mask, :]
    sort!(sub, :hour)
    hours = sub.hour

    ls = solved[(arch, :ls, t)]
    gd = solved[(arch, :gd, t)]

    panel_G = Plots.plot(
        hours, sub.G_actual;
        label = "actual", color = :black, lw = 2,
        xlabel = "Hour", ylabel = "G [MW]",
        title = "$(String(arch)) / t=$t / proj=$p / $label",
    )
    Plots.plot!(panel_G, hours, sub.G_LS; label = "LS", color = :steelblue, linestyle = :dash)
    Plots.plot!(panel_G, hours, sub.G_GD; label = "GD", color = :darkorange)

    panel_Pi = Plots.plot(
        hours, sub.Pi_actual;
        label = "actual", color = :black, lw = 2,
        xlabel = "Hour", ylabel = "π [\$/MWh]",
    )
    Plots.plot!(panel_Pi, hours, sub.Pi_LS; label = "LS", color = :steelblue, linestyle = :dash)
    Plots.plot!(panel_Pi, hours, sub.Pi_GD; label = "GD", color = :darkorange)

    panel_decisions = Plots.plot(
        hours, ls.charge[:, p];
        label = "charge LS", color = :steelblue, linestyle = :dash,
        xlabel = "Hour", ylabel = "MW",
    )
    Plots.plot!(panel_decisions, hours, gd.charge[:, p]; label = "charge GD", color = :darkorange)
    Plots.plot!(panel_decisions, hours, -ls.discharge[:, p]; label = "−discharge LS", color = :steelblue, linestyle = :dot)
    Plots.plot!(panel_decisions, hours, -gd.discharge[:, p]; label = "−discharge GD", color = :darkorange, linestyle = :dot)

    panel_state = Plots.plot(
        hours, ls.sale[:, p];
        label = "sale LS", color = :steelblue, linestyle = :dash,
        xlabel = "Hour", ylabel = "MW or MWh",
    )
    Plots.plot!(panel_state, hours, gd.sale[:, p]; label = "sale GD", color = :darkorange)
    Plots.plot!(panel_state, hours, ls.soc[:, p]; label = "SoC LS", color = :seagreen, linestyle = :dash)
    Plots.plot!(panel_state, hours, gd.soc[:, p]; label = "SoC GD", color = :seagreen)

    fig = Plots.plot(panel_G, panel_Pi, panel_decisions, panel_state;
                     layout = (2, 2), size = (1400, 900),
                     plot_title = "$(String(arch)) — $label — t=$t, proj=$p — ADL improvement \$$(round(solved[(arch,:ls,t)].cost - solved[(arch,:gd,t)].cost, digits=1))")
    fname = "examples_$(String(arch))_$(label)_t$(t)_p$(p).png"
    Plots.savefig(fig, joinpath(FIG_DIR, fname))
    return fname
end

println("\nGenerating representative example figures...")
example_index = DataFrame(
    Architecture = String[],
    Kind = String[],
    t = Int[],
    Project = Int[],
    ADL_Improvement = Float64[],
    Figure = String[],
)
for arch in ARCHS
    arch_str = String(arch)
    sub = filter(r -> r.Architecture == arch_str, example_df)
    sort!(sub, :ADL_Improvement; rev = true)
    best_row = sub[1, :]
    worst_row = sub[end, :]
    median_idx = ceil(Int, size(sub, 1) / 2)
    median_row = sub[median_idx, :]
    for (label, row) in (
        ("best", best_row),
        ("median", median_row),
        ("worst", worst_row),
    )
        t = row.t
        p = _project_with_largest_shift(arch, t)
        fname = _plot_example(arch, t, p, label)
        push!(
            example_index,
            (arch_str, label, t, p, row.ADL_Improvement, fname),
        )
    end
end
CSV.write(joinpath(result_path, "eda_examples_index.csv"), example_index)

println("Section 4: $(nrow(example_index)) example figures + summary CSV written.")

# ===================================================================
# Section 5 — Cross-model narrative
# ===================================================================

println("\n========== Section 5: narrative ==========\n")

for arch in ARCHS
    arch_str = String(arch)
    row = summary_wide[summary_wide.Architecture .== arch_str, :][1, :]
    sub = filter(r -> r.Architecture == arch_str, example_df)
    sort!(sub, :ADL_Improvement; rev = true)
    pos = count(>(0), sub.ADL_Improvement)
    neg = count(<(0), sub.ADL_Improvement)
    shift_rows_arch = filter(r -> r.Architecture == arch_str, shift_long)
    mean_dG_overall = mean(shift_rows_arch.Mean_dG)
    mean_dPi_overall = mean(shift_rows_arch.Mean_dPi)
    residual_g_overall = mean(
        filter(r -> r.Architecture == arch_str && r.Stage == "LS" && r.Quantity == "G",
               residual_long).MeanBias,
    )
    residual_pi_overall = mean(
        filter(r -> r.Architecture == arch_str && r.Stage == "LS" && r.Quantity == "Pi",
               residual_long).MeanBias,
    )
    best_t = sub.t[1]
    println("[$arch_str]")
    println("  Test ΔCost (GD - LS) = \$$(round(row.DCost_Test, digits=1))  ⇒ ADL ",
            row.DCost_Test < 0 ? "improved" : "worsened", " test performance.")
    println("  LS architectural bias on test: mean(G_LS − G) = $(round(residual_g_overall, digits=2)),  mean(π_LS − π) = $(round(residual_pi_overall, digits=2))")
    println("  ADL shift on test:             mean(G_GD − G_LS) = $(round(mean_dG_overall, digits=2)),  mean(π_GD − π_LS) = $(round(mean_dPi_overall, digits=2))")
    println("  Shortlist re-solves: ADL won on $pos / lost on $neg  (best t = $best_t)")
    println()
end

println("EDA finished. Outputs:")
println("  Tables  → ", joinpath(result_path, "eda_*.csv"))
println("  Figures → ", FIG_DIR, "/")
