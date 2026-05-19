if model_type == 1
    include("pretrain_uni.jl")
elseif model_type == 2
    include("pretrain_many.jl")
elseif model_type == 3
    include("pretrain_multi.jl")
else
    error("Invalid model_type (1=uni, 2=many, 3=multi)")
end
