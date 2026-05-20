# Archetype :type — two networks (G and π) with feature subsetting.
# G net sees calendar + own-project generation lags.
# π net sees calendar + price lags + DAM prices.

const G_INPUT_SIZE = length(g_feat_idx)
const PI_INPUT_SIZE = length(pi_feat_idx)
const G_OUTPUT_SIZE = length(_G_TARGET_IDX)
const PI_OUTPUT_SIZE = length(_PI_TARGET_IDX)

function _build_subnet(input_size, output_size, output_activation)
    if N_HIDDEN_LAYERS == 0
        return Flux.Chain(
            Flux.Dense(input_size => output_size),
            output_activation,
        )
    end
    layers = Any[Flux.Dense(input_size => HIDDEN_SIZE), Flux.relu]
    for _ = 1:(N_HIDDEN_LAYERS-1)
        push!(layers, Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE))
        push!(layers, Flux.relu)
    end
    push!(layers, Flux.Dense(HIDDEN_SIZE => output_size))
    push!(layers, output_activation)
    return Flux.Chain(layers...)
end

nns = [
    _build_subnet(G_INPUT_SIZE, G_OUTPUT_SIZE, g_output_activation),
    _build_subnet(PI_INPUT_SIZE, PI_OUTPUT_SIZE, pi_output_activation),
]

if pretrain
    batch_sz =
        PRETRAIN_BATCH_SIZE == -1 ? size(X_train, 1) : PRETRAIN_BATCH_SIZE
    train_data = Flux.DataLoader(
        (X_train', Y_train_clean'),
        batchsize = batch_sz,
        shuffle = true,
    )
    opt_state = Flux.setup(Flux.Adam(PRETRAIN_LEARNING_RATE), nns)
    local epoch = 1
    local init_time = time()
    local err = 1e18
    local stable_iters = 0
    while epoch <= PRETRAIN_EPOCHS
        Flux.train!(nns, train_data, opt_state) do m, x, y
            return mean((m[1](x[g_feat_idx, :]) .- y[_G_TARGET_IDX, :]) .^ 2) +
                   mean((m[2](x[pi_feat_idx, :]) .- y[_PI_TARGET_IDX, :]) .^ 2)
        end

        g_err = mean(
            (nns[1](X_train'[g_feat_idx, :]) .- Y_train_clean'[_G_TARGET_IDX, :]) .^ 2,
        )
        pi_err = mean(
            (nns[2](X_train'[pi_feat_idx, :]) .- Y_train_clean'[_PI_TARGET_IDX, :]) .^ 2,
        )
        err2 = g_err + pi_err
        err_var = abs(err - err2)
        err = err2

        if err_var < PRETRAIN_CONV_TOL
            stable_iters += 1
        else
            stable_iters = 0
        end
        if stable_iters > PRETRAIN_CONV_PATIENCE
            println("Pre-training converged after $epoch epochs.")
            break
        end

        if epoch % 10 == 0
            println(
                "Epoch $epoch | G MSE = $(round(g_err, digits=4)) | π MSE = $(round(pi_err, digits=4))",
            )
        end
        if (time() - init_time) > PRETRAIN_MAX_TIME
            println("Pre-training time limit reached at epoch $epoch.")
            break
        end

        epoch += 1
    end

    JLD2.jldsave(pretrained_model_state; state = Flux.state(nns))
    println("Pretrained state saved to $pretrained_model_state")
else
    models_state = JLD2.load(pretrained_model_state, "state")
    Flux.loadmodel!(nns, models_state)
    println("Loaded pretrained state from $pretrained_model_state")
end
