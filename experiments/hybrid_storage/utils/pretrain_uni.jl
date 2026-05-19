# Archetype 1: uni — one Dense network applied to the full dataset

INPUT_SIZE = length(feature_names)
OUTPUT_SIZE = length(target_names)

if N_HIDDEN_LAYERS == 0
    nn = Flux.Chain(Flux.Dense(INPUT_SIZE => OUTPUT_SIZE))
else
    layers = Any[Flux.Dense(INPUT_SIZE => HIDDEN_SIZE), Flux.relu]
    for _ = 1:(N_HIDDEN_LAYERS-1)
        push!(layers, Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE))
        push!(layers, Flux.relu)
    end
    push!(layers, Flux.Dense(HIDDEN_SIZE => OUTPUT_SIZE))
    nn = Flux.Chain(layers...)
end

# MSE pre-training loop (matpower-style: loss = sum of per-project per-output squared error).
if pretrain
    batch_sz =
        PRETRAIN_BATCH_SIZE == -1 ? size(X_train, 1) : PRETRAIN_BATCH_SIZE
    train_data = Flux.DataLoader(
        (X_train', Y_train'),
        batchsize = batch_sz,
        shuffle = true,
    )
    opt_state = Flux.setup(Flux.Adam(PRETRAIN_LEARNING_RATE), nn)
    local epoch = 1
    local init_time = time()
    local err = 1e18
    local stable_iters = 0
    while epoch <= PRETRAIN_EPOCHS
        Flux.train!(nn, train_data, opt_state) do m, x, y
            return mean((m(x) .- y) .^ 2)
        end

        err2 = mean((nn(X_train')' .- Y_train) .^ 2)
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
            println("Epoch $epoch | MSE = $(round(err, digits=4))")
        end
        if (time() - init_time) > PRETRAIN_MAX_TIME
            println("Pre-training time limit reached at epoch $epoch.")
            break
        end

        epoch += 1
    end

    JLD2.jldsave(pretrained_model_state; state = Flux.state(nn))
    println("Pretrained state saved to $pretrained_model_state")
else
    models_state = JLD2.load(pretrained_model_state, "state")
    Flux.loadmodel!(nn, models_state)
    println("Loaded pretrained state from $pretrained_model_state")
end
