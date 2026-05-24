# Solve Plan + Assess for one sample and dump key variable values.
# Usage: julia --project=../.. inspect_one.jl

using Pkg
cur_dir = @__DIR__
Pkg.activate(dirname(dirname(cur_dir)))

using JuMP
using Statistics
using Random
import Gurobi
using Flux
import NNlib
import JLD2
import JobQueueMPI
import Printf

using ApplicationDrivenLearning

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/model.jl")

pretrain = false
include("utils/pretrain.jl")

if model_type == :uni
    pred_model = ApplicationDrivenLearning.PredictiveModel(nn)
elseif model_type in (:type, :many)
    pred_model = ApplicationDrivenLearning.PredictiveModel(nns, input_output_map)
end
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

# --- Pick one sample ---
const SPLIT = :test
const T_IDX = 1

X = SPLIT == :train ? X_train : X_test
Y_dict = SPLIT == :train ? Y_dict_train : Y_dict_test
println("\n=== Inspecting $(SPLIT) sample t=$(T_IDX) (model_type=$(model_type)) ===\n")

Y_matrix = ADL.dict_to_var_indexed_matrix(Y_dict, model.forecast_vars)
y_actual = Y_matrix[T_IDX, :]
yhat = model.forecast(X[T_IDX, :])

# Solve Plan + Assess for this sample (this is the same routine ADL uses internally).
cost = ADL.compute_single_step_cost(model, y_actual, yhat)
println("Per-sample assess Cost = $(round(cost; digits=2))\n")

# --- Reshape forecast and actuals into (H, 2 channels, N_PROJ) ---
yhat_arr = deflatten(yhat)
y_arr = deflatten(y_actual)

# Helper: read JuMP value at a stage (.plan / .assess) for the policy vars.
get_plan(v) = JuMP.value.(v.plan)
get_assess(v) = JuMP.value.(v.assess)

# --- Extract solutions ---
charge_plan = JuMP.value.([charge[h, p].plan for h in 1:H, p in 1:N_PROJ])
discharge_plan = JuMP.value.([discharge[h, p].plan for h in 1:H, p in 1:N_PROJ])
charge_assess = JuMP.value.([charge[h, p].assess for h in 1:H, p in 1:N_PROJ])
discharge_assess = JuMP.value.([discharge[h, p].assess for h in 1:H, p in 1:N_PROJ])

plan_sale_val = JuMP.value.(ADL.Plan(model)[:plan_sale])
plan_shortfall_val = JuMP.value.(ADL.Plan(model)[:plan_shortfall])
plan_soc_val = JuMP.value.(ADL.Plan(model)[:plan_soc])

assess_sale_val = JuMP.value.(ADL.Assess(model)[:assess_sale])
assess_buy_val = JuMP.value.(ADL.Assess(model)[:assess_buy])

# --- Print per-project summary ---
for p in 1:N_PROJ
    pid = PROJECT_IDS[p]
    println("======================================")
    println("Project $(p) = $(pid)   (e_max=$(BESS[pid].e_max), p_max=$(BESS[pid].p_max), soc_0=$(BESS[pid].soc_0))")
    println("======================================")
    Printf.@printf("%3s | %8s %8s | %8s %8s | %8s %8s | %8s %8s %8s | %8s | %8s\n",
        "h", "G_fc", "G_act", "π_fc", "π_act",
        "charge", "dchrg",
        "p_sale", "p_short", "p_soc",
        "a_sale", "a_buy")
    for h in 1:H
        Printf.@printf("%3d | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.2f %8.2f | %8.2f | %8.2f\n",
            h,
            yhat_arr[h, 1, p], y_arr[h, 1, p],
            yhat_arr[h, 2, p], y_arr[h, 2, p],
            charge_plan[h, p], discharge_plan[h, p],
            plan_sale_val[h, p], plan_shortfall_val[h, p], plan_soc_val[h, p],
            assess_sale_val[h, p], assess_buy_val[h, p])
    end
    Printf.@printf("\nΣ over 24h: charge=%.2f  discharge=%.2f  plan_sale=%.2f  shortfall=%.2f  assess_sale=%.2f  buy=%.2f\n",
        sum(charge_plan[:, p]), sum(discharge_plan[:, p]),
        sum(plan_sale_val[:, p]), sum(plan_shortfall_val[:, p]),
        sum(assess_sale_val[:, p]), sum(assess_buy_val[:, p]))
    # Revenue/cost breakdown for this project under actuals
    rev = sum(y_arr[h, 2, p] * assess_sale_val[h, p] for h in 1:H)
    cost_buy = sum(y_arr[h, 2, p] * assess_buy_val[h, p] for h in 1:H)
    Printf.@printf("Assess: revenue (π_act·sale) = %.2f, buy cost (π_act·buy) = %.2f, net = %.2f\n\n",
        rev, cost_buy, rev - cost_buy)
end

# --- Quick feasibility sanity checks ---
println("======================================")
println("Sanity checks")
println("======================================")
max_pol_diff = max(
    maximum(abs.(charge_plan .- charge_assess)),
    maximum(abs.(discharge_plan .- discharge_assess)),
)
println("max |policy_plan − policy_assess| = $(max_pol_diff) (should be ~0)")

# Plan: charge + sale = G_fc + shortfall + discharge (no curtail anymore)
# Plan SoC bounds
println("plan_soc range: [$(round(minimum(plan_soc_val); digits=2)), $(round(maximum(plan_soc_val); digits=2))]")
# Cycle limit per project
for p in 1:N_PROJ
    pid = PROJECT_IDS[p]
    Printf.@printf("Project %d: Σdischarge=%.2f  cycle_limit=%.2f  Σcharge=%.2f  cyclic_check=|soc[24]-soc_0|=%.4f\n",
        p, sum(discharge_plan[:, p]), n_cycles * BESS[pid].e_max, sum(charge_plan[:, p]),
        abs(plan_soc_val[H, p] - BESS[pid].soc_0))
end
