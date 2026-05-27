# Analyze results.

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

using ApplicationDrivenLearning

include("config.jl")
include("utils/constants.jl")
include("utils/data.jl")
include("utils/model.jl")

pretrain = false
include("utils/pretrain.jl")

function build_pred_model_from(path)
    state = JLD2.load(path, "state")
    if model_type == :uni
        Flux.loadmodel!(nn, state isa AbstractVector ? state[1] : state)
        return ApplicationDrivenLearning.PredictiveModel(nn)
    elseif model_type in (:type, :many)
        Flux.loadmodel!(nns, state)
        return ApplicationDrivenLearning.PredictiveModel(nns, input_output_map)
    else
        error("model_type $model_type not implemented")
    end
end

function compute_assess_metrics(model, X, Y_dict)
    Y_matrix = ADL.dict_to_var_indexed_matrix(Y_dict, model.forecast_vars)
    # Profit: π_actual · assess_sale (signed; combines sale revenue when assess_sale > 0
    # with forced-buy cost when assess_sale < 0 due to locked charge exceeding G_actual).
    # Equivalent to −Cost / N_PROJ for the per-(t, p) breakdown.
    profit = Matrix{Float64}(undef, size(X, 1), N_PROJ)
    # Battery arbitrage: Σ_h π_actual · (discharge_h − charge_h). Captures the
    # price-arbitrage P&L of the storage policy, isolating the charge/discharge
    # decisions from the underlying generation that would be sold anyway.
    arbitrage = Matrix{Float64}(undef, size(X, 1), N_PROJ)
    for t in 1:size(X, 1)
        if t % 1000 == 0
            println("Computing assess metrics for sample $t / $(size(X, 1))")
        end
        y = Y_matrix[t, :]
        ADL.compute_single_step_cost(
            model,
            y,
            model.forecast(X[t, :])
        )
        pi_actual = deflatten(y)[:, 2, :]
        sale_vals = JuMP.value.(ADL.Assess(model)[:assess_sale])
        discharge_vals = JuMP.value.([discharge[h, p].assess for h in 1:H, p in 1:N_PROJ])
        charge_vals = JuMP.value.([charge[h, p].assess for h in 1:H, p in 1:N_PROJ])
        profit[t, :] = sum(pi_actual .* sale_vals, dims = 1)
        arbitrage[t, :] = sum(
            pi_actual .* (discharge_vals .- charge_vals),
            dims = 1,
        )
    end
    return profit, arbitrage
end

# Oracle: perfect-foresight benchmark. Solves the LP with forecast = actual, so the
# planner picks the best (charge, discharge) schedule knowing the true G and π. The
# resulting profit is the ceiling any forecasting model could reach. Compare:
#   - Oracle vs LS: maximum forecast-quality improvement available.
#   - Oracle vs ADL: how much ADL is leaving on the table.
function compute_oracle_metrics(model, X, Y_dict)
    Y_matrix = ADL.dict_to_var_indexed_matrix(Y_dict, model.forecast_vars)
    profit = Matrix{Float64}(undef, size(X, 1), N_PROJ)
    arbitrage = Matrix{Float64}(undef, size(X, 1), N_PROJ)
    for t in 1:size(X, 1)
        if t % 1000 == 0
            println("Computing oracle metrics for sample $t / $(size(X, 1))")
        end
        y = Y_matrix[t, :]
        # Pass actuals as both true Y and forecast yhat → planner has perfect info.
        ADL.compute_single_step_cost(model, y, y)
        pi_actual = deflatten(y)[:, 2, :]
        sale_vals = JuMP.value.(ADL.Assess(model)[:assess_sale])
        discharge_vals = JuMP.value.([discharge[h, p].assess for h in 1:H, p in 1:N_PROJ])
        charge_vals = JuMP.value.([charge[h, p].assess for h in 1:H, p in 1:N_PROJ])
        profit[t, :] = sum(pi_actual .* sale_vals, dims = 1)
        arbitrage[t, :] = sum(
            pi_actual .* (discharge_vals .- charge_vals),
            dims = 1,
        )
    end
    return profit, arbitrage
end

# LS
pred_model = build_pred_model_from(pretrained_model_state)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

ls_pred_train = deflatten_model_forecast(model.forecast(X_train')', model.forecast_vars)
ls_G_err_train = ls_pred_train[:, :, 1, :] - deflatten(Y_train_clean)[:, :, 1, :]
ls_G_rmse_train = mean(ls_G_err_train .^ 2) .^ 0.5
ls_Pi_err_train = ls_pred_train[:, :, 2, :] - deflatten(Y_train_clean)[:, :, 2, :]
ls_Pi_rmse_train = mean(ls_Pi_err_train .^ 2) .^ 0.5
ls_cost_train = ADL.compute_cost(model, X_train, Y_dict_train)
ls_profit_train, ls_arb_train = compute_assess_metrics(model, X_train, Y_dict_train)

ls_pred_test = deflatten_model_forecast(model.forecast(X_test')', model.forecast_vars)
ls_G_err_test = ls_pred_test[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
ls_G_rmse_test = mean(ls_G_err_test .^ 2) .^ 0.5
ls_Pi_err_test = ls_pred_test[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
ls_Pi_rmse_test = mean(ls_Pi_err_test .^ 2) .^ 0.5
ls_cost_test = ADL.compute_cost(model, X_test, Y_dict_test)
ls_profit_test, ls_arb_test = compute_assess_metrics(model, X_test, Y_dict_test)

println("Pretrained predictor train RMSE: G = $ls_G_rmse_train, Pi = $ls_Pi_rmse_train")
println("Pretrained predictor train Cost: $ls_cost_train")
println("Pretrained predictor train Profit: $(mean(ls_profit_train))")
println("Pretrained predictor train Arbitrage: $(mean(ls_arb_train))")
println("Pretrained predictor test RMSE: G = $ls_G_rmse_test, Pi = $ls_Pi_rmse_test")
println("Pretrained predictor test Cost: $ls_cost_test")
println("Pretrained predictor test Profit: $(mean(ls_profit_test))")
println("Pretrained predictor test Arbitrage: $(mean(ls_arb_test))")

# ADL
pred_model = build_pred_model_from(final_model_state)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

gd_pred_train = deflatten_model_forecast(model.forecast(X_train')', model.forecast_vars)
gd_G_err_train = gd_pred_train[:, :, 1, :] - deflatten(Y_train_clean)[:, :, 1, :]
gd_G_rmse_train = mean(gd_G_err_train .^ 2) .^ 0.5
gd_Pi_err_train = gd_pred_train[:, :, 2, :] - deflatten(Y_train_clean)[:, :, 2, :]
gd_Pi_rmse_train = mean(gd_Pi_err_train .^ 2) .^ 0.5
gd_cost_train = ADL.compute_cost(model, X_train, Y_dict_train)
gd_profit_train, gd_arb_train = compute_assess_metrics(model, X_train, Y_dict_train)

gd_pred_test = deflatten_model_forecast(model.forecast(X_test')', model.forecast_vars)
gd_G_err_test = gd_pred_test[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
gd_G_rmse_test = mean(gd_G_err_test .^ 2) .^ 0.5
gd_Pi_err_test = gd_pred_test[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
gd_Pi_rmse_test = mean(gd_Pi_err_test .^ 2) .^ 0.5
gd_cost_test = ADL.compute_cost(model, X_test, Y_dict_test)
gd_profit_test, gd_arb_test = compute_assess_metrics(model, X_test, Y_dict_test)

println("Gradient-mode predictor train RMSE: G = $gd_G_rmse_train, Pi = $gd_Pi_rmse_train")
println("Gradient-mode predictor train Cost: $gd_cost_train")
println("Gradient-mode predictor train Profit: $(mean(gd_profit_train))")
println("Gradient-mode predictor train Arbitrage: $(mean(gd_arb_train))")
println("Gradient-mode predictor test RMSE: G = $gd_G_rmse_test, Pi = $gd_Pi_rmse_test")
println("Gradient-mode predictor test Cost: $gd_cost_test")
println("Gradient-mode predictor test Profit: $(mean(gd_profit_test))")
println("Gradient-mode predictor test Arbitrage: $(mean(gd_arb_test))")

# Oracle (perfect foresight): solves the LP with forecast = actual per sample.
# This gives the ceiling profit any predictor could achieve.
oracle_profit_train, oracle_arb_train = compute_oracle_metrics(model, X_train, Y_dict_train)
oracle_profit_test, oracle_arb_test = compute_oracle_metrics(model, X_test, Y_dict_test)
oracle_cost_train = -N_PROJ * mean(oracle_profit_train)
oracle_cost_test = -N_PROJ * mean(oracle_profit_test)

println("Oracle (perfect foresight) train Cost: $oracle_cost_train")
println("Oracle (perfect foresight) train Profit: $(mean(oracle_profit_train))")
println("Oracle (perfect foresight) train Arbitrage: $(mean(oracle_arb_train))")
println("Oracle (perfect foresight) test Cost: $oracle_cost_test")
println("Oracle (perfect foresight) test Profit: $(mean(oracle_profit_test))")
println("Oracle (perfect foresight) test Arbitrage: $(mean(oracle_arb_test))")

println("Headroom (Oracle − LS) train Cost: $(oracle_cost_train - ls_cost_train)")
println("Headroom (Oracle − LS) test Cost: $(oracle_cost_test - ls_cost_test)")
println("ADL captured (LS − ADL)/(LS − Oracle) train: $((ls_cost_train - gd_cost_train) / (ls_cost_train - oracle_cost_train))")
println("ADL captured (LS − ADL)/(LS − Oracle) test:  $((ls_cost_test - gd_cost_test) / (ls_cost_test - oracle_cost_test))")

# save results to CSV
results_df = DataFrame(
    Model = ["Pretrained", "ADL Gradient", "Oracle"],
    Train_G_RMSE = [ls_G_rmse_train, gd_G_rmse_train, 0.0],
    Train_Pi_RMSE = [ls_Pi_rmse_train, gd_Pi_rmse_train, 0.0],
    Train_Cost = [ls_cost_train, gd_cost_train, oracle_cost_train],
    Train_Profit = [mean(ls_profit_train), mean(gd_profit_train), mean(oracle_profit_train)],
    Train_Arbitrage = [mean(ls_arb_train), mean(gd_arb_train), mean(oracle_arb_train)],
    Test_G_RMSE = [ls_G_rmse_test, gd_G_rmse_test, 0.0],
    Test_Pi_RMSE = [ls_Pi_rmse_test, gd_Pi_rmse_test, 0.0],
    Test_Cost = [ls_cost_test, gd_cost_test, oracle_cost_test],
    Test_Profit = [mean(ls_profit_test), mean(gd_profit_test), mean(oracle_profit_test)],
    Test_Arbitrage = [mean(ls_arb_test), mean(gd_arb_test), mean(oracle_arb_test)],
)
CSV.write(
    joinpath(
        result_path, 
        "results_summary_$(ARCH_NAME).csv"
    ), 
    results_df
)

# dataframe with LS, GD (train and test) forecasts
forecast_train_df = DataFrame()
forecast_test_df = DataFrame()
for t in 1:size(X_test, 1)
    for pidx in 1:N_PROJ
        for h in 1:H
            push!(forecast_train_df, (
                t = t,
                project = pidx,
                hour = h,
                G_actual = deflatten(Y_train_clean)[t, h, 1, pidx],
                Pi_actual = deflatten(Y_train_clean)[t, h, 2, pidx],
                G_LS = ls_pred_train[t, h, 1, pidx],
                Pi_LS = ls_pred_train[t, h, 2, pidx],
                G_GD = gd_pred_train[t, h, 1, pidx],
                Pi_GD = gd_pred_train[t, h, 2, pidx],
                Profit_LS = ls_profit_train[t, pidx],
                Profit_GD = gd_profit_train[t, pidx],
                Arb_LS = ls_arb_train[t, pidx],
                Arb_GD = gd_arb_train[t, pidx],
            ))
            push!(forecast_test_df, (
                t = t,
                project = pidx,
                hour = h,
                G_actual = deflatten(Y_test_clean)[t, h, 1, pidx],
                Pi_actual = deflatten(Y_test_clean)[t, h, 2, pidx],
                G_LS = ls_pred_test[t, h, 1, pidx],
                Pi_LS = ls_pred_test[t, h, 2, pidx],
                G_GD = gd_pred_test[t, h, 1, pidx],
                Pi_GD = gd_pred_test[t, h, 2, pidx],
                Profit_LS = ls_profit_test[t, pidx],
                Profit_GD = gd_profit_test[t, pidx],
                Arb_LS = ls_arb_test[t, pidx],
                Arb_GD = gd_arb_test[t, pidx],
            ))
        end
    end
end
CSV.write(
    joinpath(
        result_path, 
        "forecasts_train_$(ARCH_NAME).csv"
    ), 
    forecast_train_df
)
CSV.write(
    joinpath(
        result_path, 
        "forecasts_test_$(ARCH_NAME).csv"
    ), 
    forecast_test_df
)
