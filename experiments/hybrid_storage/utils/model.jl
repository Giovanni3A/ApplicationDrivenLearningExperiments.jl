# ADL.Model for the hybrid_storage LP: 3 projects, 24-hour horizon, cyclic SoC.
# Variables and constraints match PLAN.md §4.

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

# --- Plan stage ---

@variables(
    ADL.Plan(model),
    begin
        plan_shortfall[1:H, 1:N_PROJ] >= 0.0
        plan_soc[1:H, 1:N_PROJ] >= 0.0
        plan_sale[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        plan_curtail[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        plan_generator_output[h = 1:H, pidx = 1:N_PROJ] >= 0.0
    end
)

# Generator output: predicted generation less optional curtailment. G_phys is
# guaranteed non-negative by the NN output reparametrization.
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    plan_generator_output[h, pidx] ==
    G_fc[h, pidx].plan + plan_shortfall[h, pidx] - plan_curtail[h, pidx]
)

# Balance constraint
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    charge[h, pidx].plan + plan_sale[h, pidx] ==
    plan_generator_output[h, pidx] + discharge[h, pidx].plan
)
@constraint(
    ADL.Plan(model),
    [h = 1:H, pidx = 1:N_PROJ],
    charge[h, pidx].plan <= plan_generator_output[h, pidx]
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
        assess_soc[1:H, 1:N_PROJ] >= 0.0
        assess_sale[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_curtail[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_generator_output[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_charge_slack_pos[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_charge_slack_neg[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_discharge_slack_pos[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_discharge_slack_neg[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_charge[h = 1:H, pidx = 1:N_PROJ] >= 0.0
        assess_discharge[h = 1:H, pidx = 1:N_PROJ] >= 0.0
    end
)

# slack variables for charge/discharge to enable subgradient flow when the battery is full/empty or at power limits. 
# Penalized in the objective to ensure they're only used when necessary.
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_charge[h, pidx] <=
    charge[h, pidx].assess
)
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_discharge[h, pidx] <=
    discharge[h, pidx].assess
)

# Generator output: predicted generation less optional curtailment. G_phys is
# guaranteed non-negative by the NN output reparametrization.
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_generator_output[h, pidx] ==
    G_fc[h, pidx].assess - assess_curtail[h, pidx]
)

# Balance constraint
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_charge[h, pidx] + assess_sale[h, pidx] ==
    assess_generator_output[h, pidx] + assess_discharge[h, pidx]
)
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_charge[h, pidx] <= assess_generator_output[h, pidx]
)

# SoC dynamics (1h timestep, so MW × 1h = MWh and c/d are already MWh).
@constraint(
    ADL.Assess(model),
    [pidx = 1:N_PROJ],
    assess_soc[1, pidx] ==
    BESS[PROJECT_IDS[pidx]].soc_0 + η * assess_charge[1, pidx] -
    assess_discharge[1, pidx] / η
)
@constraint(
    ADL.Assess(model),
    [h = 2:H, pidx = 1:N_PROJ],
    assess_soc[h, pidx] ==
    assess_soc[h-1, pidx] + η * assess_charge[h, pidx] -
    assess_discharge[h, pidx] / η
)

# battery state upper limit
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_soc[h, pidx] <= BESS[PROJECT_IDS[pidx]].e_max
)

# power limit
@constraint(
    ADL.Assess(model),
    [h = 1:H, pidx = 1:N_PROJ],
    assess_charge[h, pidx] + assess_discharge[h, pidx] <=
    BESS[PROJECT_IDS[pidx]].p_max
)

# cycle limit
@constraint(
    ADL.Assess(model),
    [pidx = 1:N_PROJ],
    sum(assess_discharge[h, pidx] for h = 1:H) <=
    n_cycles * BESS[PROJECT_IDS[pidx]].e_max
)

# cyclic SoC constraint
@constraint(
    ADL.Assess(model),
    [pidx = 1:N_PROJ],
    assess_soc[H, pidx] == BESS[PROJECT_IDS[pidx]].soc_0
)

@objective(
    ADL.Assess(model),
    Min,
    -sum(
        pi_fc[h, pidx].assess * assess_sale[h, pidx] for h = 1:H,
        pidx = 1:N_PROJ
    ) +
    PHANTOM_PENALTY * sum(
        assess_charge_slack_pos[h, pidx] +
        assess_charge_slack_neg[h, pidx] +
        assess_discharge_slack_pos[h, pidx] +
        assess_discharge_slack_neg[h, pidx] for h = 1:H, pidx = 1:N_PROJ
    )
)

set_optimizer(model, Gurobi.Optimizer)
set_silent(model)
ADL.build(model)

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
