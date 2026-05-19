# Project portfolio, BESS sizing, and problem dimensions for the hybrid_storage
# experiment. Single source of truth — referenced by data.jl, model.jl, and the
# pretrain_*.jl files. Values locked in PLAN.md.

using Dates

const PROJECTS = (
    wind_a = (tech = :wind, nameplate = 122.0),
    wind_b = (tech = :wind, nameplate = 160.0),
    solar_a = (tech = :solar, nameplate = 200.0),
)

# 2-hour duration, 85% round-trip efficiency, 1 cycle / 24h horizon, half-charged start.
η = 0.922
n_cycles = 1.0
const BESS = (
    wind_a = (p_max = 40.0, e_max = 80.0, soc_0 = 40.0),
    wind_b = (p_max = 50.0, e_max = 100.0, soc_0 = 50.0),
    solar_a = (p_max = 100.0, e_max = 200.0, soc_0 = 100.0),
)

const PROJECT_IDS = collect(keys(PROJECTS))           # [:wind_a, :wind_b, :solar_a]
const N_PROJ = length(PROJECT_IDS)

const H = 24                       # planning horizon (hours)
const T_ACT = 24                       # action window (= H, no lookahead)
const N_OUTPUTS_PER_PROJ = 2 * H                    # G̃ and π̃ over H hours = 48
const N_OUTPUTS_TOTAL = N_PROJ * N_OUTPUTS_PER_PROJ   # 144

# Chronological split boundaries (UTC, half-open intervals)
const TRAIN_START = DateTime("2023-08-28T00:00:00")
const TRAIN_END = DateTime("2025-09-01T00:00:00")
const TEST_START = DateTime("2025-09-01T00:00:00")
const TEST_END = DateTime("2026-03-14T00:00:00")

const PHANTOM_PENALTY = 99999   # $/MWh penalty for phantom generation (infeasible production shortfall)