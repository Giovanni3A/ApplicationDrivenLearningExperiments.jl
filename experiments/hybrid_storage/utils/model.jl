# ADL.Model for the hybrid_storage LP: 3 projects, 24-hour horizon, cyclic SoC.

ADL = ApplicationDrivenLearning

model = ADL.Model()

# --- Forecast variables (predicted by the network in Plan; realized in Assess) ---
# Indexed by (h, p): h ∈ 1..H, p ∈ 1..N_PROJ. Network output is in standardized
# space; the LP un-standardizes inside each constraint, mirroring matpower.
@variable(model, G_fc[1:H, 1:N_PROJ], ADL.Forecast)
@variable(model, pi_fc[1:H, 1:N_PROJ], ADL.Forecast)

# --- Decision (Policy) variables — values in both Plan and Assess stages ---
@variable(model, charge[1:H, 1:N_PROJ] >= 0.0, ADL.Policy)   # charge from renewables (MWh)
@variable(model, discharge[1:H, 1:N_PROJ] >= 0.0, ADL.Policy)   # discharge to grid    (MWh)
@variable(model, fc[i = 1:2, h = 1:H, pidx = 1:N_PROJ], ADL.Policy)

@constraints(
    ADL.Plan(model),
    begin
        [h = 1:H, pidx = 1:N_PROJ],
        fc[1,h,pidx].plan == G_fc[h, pidx].plan
        [h = 1:H, pidx = 1:N_PROJ],
        fc[2,h,pidx].plan == pi_fc[h, pidx].plan
    end
)

# --- Plan stage ---

@variables(
    ADL.Plan(model),
    begin
        plan_shortfall[1:H, 1:N_PROJ] >= 0.0
        plan_soc[1:H, 1:N_PROJ] >= 0.0
        plan_sale[h = 1:H, pidx = 1:N_PROJ] >= 0.0
    end
)

# Generator output: predicted generation less optional curtailment (plus shortfall when needed)
@expression(
    ADL.Plan(model),
    plan_generator_output[h = 1:H, pidx = 1:N_PROJ],
    G_fc[h, pidx].plan + plan_shortfall[h, pidx]
)
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    plan_generator_output[h, pidx] >= 0.0,
)

# Balance constraint
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    charge[h, pidx].plan + plan_sale[h, pidx] ==
    plan_generator_output[h, pidx] + discharge[h, pidx].plan
)

# SoC dynamics (1h timestep, so MW × 1h = MWh and c/d are already MWh).
@constraint(
    ADL.Plan(model),
    [pidx = 1:N_PROJ],
    plan_soc[1, pidx] ==
    BESS[PROJECT_IDS[pidx]].soc_0 + η * charge[1, pidx].plan -
    discharge[1, pidx].plan / η
)
@constraint(
    ADL.Plan(model),
    [h = 2:H, pidx = 1:N_PROJ],
    plan_soc[h, pidx] ==
    plan_soc[h-1, pidx] + η * charge[h, pidx].plan -
    discharge[h, pidx].plan / η
)

# battery state upper limit
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    plan_soc[h, pidx] <= BESS[PROJECT_IDS[pidx]].e_max
)

# power limit
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    charge[h, pidx].plan + discharge[h, pidx].plan <=
    BESS[PROJECT_IDS[pidx]].p_max
)

# cycle limit
@constraint(
    ADL.Plan(model),
    [pidx = 1:N_PROJ],
    sum(discharge[h, pidx].plan for h = 1:H) <=
    n_cycles * BESS[PROJECT_IDS[pidx]].e_max
)

# cyclic SoC constraint
@constraint(
    ADL.Plan(model),
    [pidx = 1:N_PROJ],
    plan_soc[H, pidx] == BESS[PROJECT_IDS[pidx]].soc_0
)

# Plan objective: minimize -revenue (Min problem, matching matpower's convention).
@objective(
    ADL.Plan(model),
    Min,
    -sum(
        pi_fc[h, pidx].plan * plan_sale[h, pidx] for h = 1:H, pidx = 1:N_PROJ
    ) +
    PHANTOM_PENALTY * sum(plan_shortfall[h, pidx] for h = 1:H, pidx = 1:N_PROJ)
)

# --- Assess stage (same structure with .assess on forecast variables) ---
@variables(
    ADL.Assess(model),
    begin
        fc_err[i = 1:2, h = 1:H, pidx = 1:N_PROJ]
    end
)

@constraints(
    ADL.Assess(model),
    begin
        [h = 1:H, pidx = 1:N_PROJ],
        fc_err[1, h, pidx] >= G_fc[h, pidx].assess - fc[1, h, pidx].assess
        [h = 1:H, pidx = 1:N_PROJ],
        fc_err[1, h, pidx] >= -(G_fc[h, pidx].assess - fc[1, h, pidx].assess)
        [h = 1:H, pidx = 1:N_PROJ],
        fc_err[2, h, pidx] >= pi_fc[h, pidx].assess - fc[2, h, pidx].assess
        [h = 1:H, pidx = 1:N_PROJ],
        fc_err[2, h, pidx] >= -(pi_fc[h, pidx].assess - fc[2, h, pidx].assess)
    end
)

# Balance constraint
@expression(
    ADL.Assess(model),
    assess_sale[h = 1:H, pidx = 1:N_PROJ],
    G_fc[h, pidx].assess + discharge[h, pidx].assess - charge[h, pidx].assess
)

@expression(
    ADL.Assess(model),
    assess_profit[h = 1:H, pidx = 1:N_PROJ],
        pi_fc[h, pidx].assess * assess_sale[h, pidx]
)

LAMBDA = 0.01

@objective(
    ADL.Assess(model),
    Min,
    sum(
        -assess_profit[h, pidx] for h = 1:H, pidx = 1:N_PROJ
    ) + LAMBDA * sum(fc_err)
)

set_optimizer(model, Gurobi.Optimizer)
set_silent(model)
ADL._build(model)

# --- Y_dict: map each Forecast variable to its column of Y_{train,test} ---
Y_dict_train = Dict{ADL.Forecast,Vector{Float32}}()
Y_dict_test = Dict{ADL.Forecast,Vector{Float32}}()
for (pidx, pid) in enumerate(PROJECT_IDS)
    for h = 1:H
        Y_dict_train[G_fc[h, pidx]] = Y_train_clean[:, y_idx_G(pidx, h)]
        Y_dict_train[pi_fc[h, pidx]] = Y_train_clean[:, y_idx_pi(pidx, h)]
        Y_dict_test[G_fc[h, pidx]] = Y_test_clean[:, y_idx_G(pidx, h)]
        Y_dict_test[pi_fc[h, pidx]] = Y_test_clean[:, y_idx_pi(pidx, h)]
    end
end

# --- forecast units per archetype ---
# Forecast vector order must match the corresponding network output order: column-major
# iteration over [h, p], identical to the iteration used to build _G_TARGET_IDX / _PI_TARGET_IDX.
# `nets` is the single chain under :uni and the vector of subnets otherwise; it is
# an argument because callers reload weights from a saved state before assembling.
function forecast_units(nets)
    if model_type == :uni
        # one network over the whole input row, emitting every forecast variable
        return [ADL.ForecastModel(architecture = nets, outputs = model.forecast_vars)]
    elseif model_type == :type
        # one unit per network - each subnet emits all projects' targets
        return [
            ADL.ForecastModel(
                inputs = g_feat_idx,
                architecture = nets[1],
                outputs = vec([G_fc[h, p] for h = 1:H, p = 1:N_PROJ]),
            ),
            ADL.ForecastModel(
                inputs = pi_feat_idx,
                architecture = nets[2],
                outputs = vec([pi_fc[h, p] for h = 1:H, p = 1:N_PROJ]),
            ),
        ]
    elseif model_type == :many
        error(
            ":many applies one shared subnet once per project, which a vector " *
            "of ForecastModel units cannot express: a unit reads a single input " *
            "group, and two units may not share an architecture object.",
        )
    else
        error("model_type $model_type not implemented")
    end
end

function deflatten_model_forecast(y, forecast_vars::Vector{<:ADL.Forecast})
    # Convert vector of Forecasts (ordered by h, then pidx) into a 4D array (T x N_PROJ x 2 x N_SAMPLES).
    # We don't know the ordering, so we have to look up the forecast variables vector.
    var_to_idx = Dict(var => idx for (idx, var) in enumerate(forecast_vars))

    y_out = zeros(Float32, size(y, 1), H, 2, N_PROJ)
    for pidx in 1:N_PROJ
        for h in 1:H
            g_var = G_fc[h, pidx]
            pi_var = pi_fc[h, pidx]
            g_idx = var_to_idx[g_var]
            pi_idx = var_to_idx[pi_var]
            y_out[:, h, 1, pidx] = y[:, g_idx]
            y_out[:, h, 2, pidx] = y[:, pi_idx]
        end
    end
    return y_out
end