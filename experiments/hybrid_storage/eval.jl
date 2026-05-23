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

function compute_sales_profit(model, X, Y_dict)
    Y_matrix = ADL.dict_to_var_indexed_matrix(Y_dict, model.forecast_vars)
    sales = Matrix{Float64}(undef, size(X, 1), N_PROJ)
    for t in 1:size(X, 1)
        if t % 1000 == 0
            println("Computing sales for sample $t / $(size(X, 1))")
        end
        y = Y_matrix[t, :]
        ADL.compute_single_step_cost(
            model, 
            y, 
            model.forecast(X[t, :])
        )
        sales[t, :] = sum(
            deflatten(y)[:, 2, :] .* 
            JuMP.value.(ADL.Assess(model)[:assess_sale]),
            dims = 1
        )
    end
    return sales
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
ls_sales_train = compute_sales_profit(model, X_train, Y_dict_train)

ls_pred_test = deflatten_model_forecast(model.forecast(X_test')', model.forecast_vars)
ls_G_err_test = ls_pred_test[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
ls_G_rmse_test = mean(ls_G_err_test .^ 2) .^ 0.5
ls_Pi_err_test = ls_pred_test[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
ls_Pi_rmse_test = mean(ls_Pi_err_test .^ 2) .^ 0.5
ls_cost_test = ADL.compute_cost(model, X_test, Y_dict_test)
ls_sales_test = compute_sales_profit(model, X_test, Y_dict_test)

println("Pretrained predictor train RMSE: G = $ls_G_rmse_train, Pi = $ls_Pi_rmse_train")
println("Pretrained predictor train Cost: $ls_cost_train")
println("Pretrained predictor train Sales: $(mean(ls_sales_train))")
println("Pretrained predictor test RMSE: G = $ls_G_rmse_test, Pi = $ls_Pi_rmse_test")
println("Pretrained predictor test Cost: $ls_cost_test")
println("Pretrained predictor test Sales: $(mean(ls_sales_test))")

# ADL
pred_model = build_pred_model_from(final_model_state)
ApplicationDrivenLearning.set_forecast_model(model, pred_model)

gd_pred_train = deflatten_model_forecast(model.forecast(X_train')', model.forecast_vars)
gd_G_err_train = gd_pred_train[:, :, 1, :] - deflatten(Y_train_clean)[:, :, 1, :]
gd_G_rmse_train = mean(gd_G_err_train .^ 2) .^ 0.5
gd_Pi_err_train = gd_pred_train[:, :, 2, :] - deflatten(Y_train_clean)[:, :, 2, :]
gd_Pi_rmse_train = mean(gd_Pi_err_train .^ 2) .^ 0.5
gd_cost_train = ADL.compute_cost(model, X_train, Y_dict_train)
gd_sales_train = compute_sales_profit(model, X_train, Y_dict_train)

gd_pred_test = deflatten_model_forecast(model.forecast(X_test')', model.forecast_vars)
gd_G_err_test = gd_pred_test[:, :, 1, :] - deflatten(Y_test_clean)[:, :, 1, :]
gd_G_rmse_test = mean(gd_G_err_test .^ 2) .^ 0.5
gd_Pi_err_test = gd_pred_test[:, :, 2, :] - deflatten(Y_test_clean)[:, :, 2, :]
gd_Pi_rmse_test = mean(gd_Pi_err_test .^ 2) .^ 0.5
gd_cost_test = ADL.compute_cost(model, X_test, Y_dict_test)
gd_sales_test = compute_sales_profit(model, X_test, Y_dict_test)

println("Gradient-mode predictor train RMSE: G = $gd_G_rmse_train, Pi = $gd_Pi_rmse_train")
println("Gradient-mode predictor train Cost: $gd_cost_train")
println("Gradient-mode predictor train Sales: $(mean(gd_sales_train))")
println("Gradient-mode predictor test RMSE: G = $gd_G_rmse_test, Pi = $gd_Pi_rmse_test")
println("Gradient-mode predictor test Cost: $gd_cost_test")
println("Gradient-mode predictor test Sales: $(mean(gd_sales_test))")

# save results to CSV
results_df = DataFrame(
    Model = ["Pretrained", "ADL Gradient"],
    Train_G_RMSE = [ls_G_rmse_train, gd_G_rmse_train],
    Train_Pi_RMSE = [ls_Pi_rmse_train, gd_Pi_rmse_train],
    Train_Cost = [ls_cost_train, gd_cost_train],
    Train_Sales = [mean(ls_sales_train), mean(gd_sales_train)],
    Test_G_RMSE = [ls_G_rmse_test, gd_G_rmse_test],
    Test_Pi_RMSE = [ls_Pi_rmse_test, gd_Pi_rmse_test],
    Test_Cost = [ls_cost_test, gd_cost_test],
    Test_Sales = [mean(ls_sales_test), mean(gd_sales_test)],
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
                Sales_LS = ls_sales_train[t, pidx],
                Sales_GD = gd_sales_train[t, pidx],
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
                Sales_LS = ls_sales_test[t, pidx],
                Sales_GD = gd_sales_test[t, pidx],
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

# # Visualize forecasts vs actuals for a single test sample

# t = 1

# t_y = deflatten(Y_test_clean)[t, :, :, :]
# g = t_y[:, 1, :]
# pi = t_y[:, 2, :]

# ls_ghat = ls_pred[t, :, 1, :]
# ls_pihat = ls_pred[t, :, 2, :]
# gd_ghat = gd_pred[t, :, 1, :]
# gd_pihat = gd_pred[t, :, 2, :]

# import Plots

# # 3 x 2 grid with generation and actuals vs forecast for all three projects
# fig_1_1 = Plots.plot(g[:, 1], label = "Actual G", title = "Project 1", xlabel = "Hour", ylabel = "MW")
# Plots.plot!(ls_ghat[:, 1], label = "LS G", linestyle = :dash)
# Plots.plot!(gd_ghat[:, 1], label = "GD G", linestyle = :dashdot)

# fig_1_2 = Plots.plot(pi[:, 1], label = "Actual Pi", title = "Project 1", xlabel = "Hour", ylabel = "\$/MWh")
# Plots.plot!(ls_pihat[:, 1], label = "LS Pi", linestyle = :dash)
# Plots.plot!(gd_pihat[:, 1], label = "GD Pi", linestyle = :dashdot)

# fig_2_1 = Plots.plot(g[:, 2], label = "Actual G", title = "Project 2", xlabel = "Hour", ylabel = "MW")
# Plots.plot!(ls_ghat[:, 2], label = "LS G", linestyle = :dash)
# Plots.plot!(gd_ghat[:, 2], label = "GD G", linestyle = :dashdot)

# fig_2_2 = Plots.plot(pi[:, 2], label = "Actual Pi", title = "Project 2", xlabel = "Hour", ylabel = "\$/MWh")
# Plots.plot!(ls_pihat[:, 2], label = "LS Pi", linestyle = :dash)
# Plots.plot!(gd_pihat[:, 2], label = "GD Pi", linestyle = :dashdot)

# fig_3_1 = Plots.plot(g[:, 3], label = "Actual G", title = "Project 3", xlabel = "Hour", ylabel = "MW")
# Plots.plot!(ls_ghat[:, 3], label = "LS G", linestyle = :dash)
# Plots.plot!(gd_ghat[:, 3], label = "GD G", linestyle = :dashdot)

# fig_3_2 = Plots.plot(pi[:, 3], label = "Actual Pi", title = "Project 3", xlabel = "Hour", ylabel = "\$/MWh")
# Plots.plot!(ls_pihat[:, 3], label = "LS Pi", linestyle = :dash)
# Plots.plot!(gd_pihat[:, 3], label = "GD Pi", linestyle = :dashdot)

# final_fig = Plots.plot(
#     fig_1_1,
#     fig_1_2,
#     fig_2_1,
#     fig_2_2,
#     fig_3_1,
#     fig_3_2;
#     layout = (3, 2),
#     size = (1200, 800),
#     title = "Forecasts vs Actuals for Test Sample t=$t",
# )