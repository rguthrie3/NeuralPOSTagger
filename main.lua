require 'torch'
require 'optim'
require 'paths'
require 'nn'
require 'rnn'


function sample_minibatch(shuffle_idxs, idx_start, dataset, context_size)
    local word_feats = torch.IntTensor(args.minibatch_size, context_size)
    local targets = torch.IntTensor(args.minibatch_size)
    local idx = idx_start
    for j = 1, args.minibatch_size do
        word_feats[j] = dataset.contexts[shuffle_idxs[idx]]
        targets[j] = dataset.targets[shuffle_idxs[idx]]
        idx = idx + 1
    end

    if args.cuda then
        word_feats = word_feats:cuda()
        targets = targets:cuda()
    end

    return word_feats, targets
end


-- ===-------------------------------------------------------------------------------------------------------===
-- PARAMETER PARSING
-- ===-------------------------------------------------------------------------------------------------------===
local cmd = torch.CmdLine()
cmd:text("Train a part of speech tagger using embeddings for contexts.")

-- Data options
cmd:option("--dataset", "", "data as output by make_dataset.lua")
cmd:option("--embeddings", "", "Filename with embeddings for each word in training data vocab")

-- Training configuration
cmd:option("--num_epochs", 15, "The number of full passes over the training data")
cmd:option("--hidden_layers", 2, "Number of hidden layers")
cmd:option("--hidden_size", 625, "Dimensionality of hidden layer")
cmd:option("--minibatch_size", 25, "The number of elements per minibatch")
cmd:option("--learning_rate", 0.001, "Set the learning rate")
cmd:option("--learning_rate_decay", 0.97, "Amount to decay the learning rate by per epoch (enter a value <= 0 for no decay")
cmd:option("--dropout", 0.5, "Apply dropout after linear layers")
cmd:option("--step_clipping", 1.0, "Gradient norm to clip at (enter a value <= 0 for no clipping")

-- Runtime configuration
cmd:option("--logging_dir_name", "", "Directory where to dump all logging information (if not supplied it will default to a timestamp)")
cmd:option("--cuda", false, "Use CUDA")

args = cmd:parse(arg or {})
if args.cuda then
    require 'cunn'
end


-- ===-------------------------------------------------------------------------------------------------------===
-- LOGGING CONFIGURATION
-- ===-------------------------------------------------------------------------------------------------------===
local logging_dir = cmd:string('experiment', args, {}) .. sys.clock()
if args.logging_dir_name ~= "" then
    logging_dir = args.logging_dir_name
end
print("Logs being dumped in " .. logging_dir)
paths.mkdir(logging_dir)
cmd:log(logging_dir .. "/log", args)
print("===> Dataset: " .. args.dataset)


-- ===-------------------------------------------------------------------------------------------------------===
-- LOAD DATASETS
-- ===-------------------------------------------------------------------------------------------------------===
print("Loading datasets...")
local dataset = torch.load(args.dataset)
local training_data   = dataset.training_data
print("Loaded training data")
local dev_data        = dataset.dev_data
print("Loaded dev data")
local word_vocab_size = dataset.vocab_size
local tag_set_size    = dataset.tag_set_size
local word_indices    = dataset.word_to_ix
local context_size    = training_data.contexts:size()[2]
local ix_to_tag       = {}
for tag, idx in pairs(dataset.tag_to_ix) do
    ix_to_tag[idx] = tag
end
local ix_to_word = {}
for word, idx in pairs(dataset.word_to_ix) do
    ix_to_word[idx] = word
end


-- ===-------------------------------------------------------------------------------------------------------===
-- READ IN PRETRAINED EMBEDDINGS
-- ===-------------------------------------------------------------------------------------------------------===
assert(args.embeddings ~= "", "ERROR: Please supply a file with embeddings for each word in the vocab")
print("===> Using embeddings from file " .. args.embeddings)
local embeddings = {}
local file       = torch.DiskFile(args.embeddings, "r"):quiet()
local line       = file:readString("*l")
while not file:hasError() do
    local word_and_embed = line:split(' ')
    local word = word_and_embed[1]
    local embed = {}
    for i = 2, #word_and_embed do
        embed[i - 1] = word_and_embed[i]
    end
    embeddings[word_indices[word]] = torch.Tensor(embed)
    line = file:readString("*l")
end
local embedding_size = embeddings[1]:size()[1]


-- ===-------------------------------------------------------------------------------------------------------===
-- BUILD MODEL AND CONFIGURE
-- ===-------------------------------------------------------------------------------------------------------===
local network = nn.Sequential()
local lookup  = nn.LookupTable(word_vocab_size, embedding_size)
network:add(lookup)
network:add(nn.Reshape(args.minibatch_size, context_size * embedding_size))
network:add(nn.Linear(context_size * embedding_size, args.hidden_size))
network:add(nn.Tanh())
network:add(nn.Dropout(args.dropout))
for i = 1, args.hidden_layers - 1 do
    network:add(nn.Linear(args.hidden_size, args.hidden_size))
           :add(nn.Tanh())
           :add(nn.Dropout(args.dropout))
end
network:add(nn.Linear(args.hidden_size, tag_set_size))
network:add(nn.LogSoftMax())
local trainable_params, grad_params = network:getParameters()

local criterion = nn.CrossEntropyCriterion()

if args.cuda then
    print("===> Using CUDA")
    network:cuda()
    criterion:cuda()
    trainable_params, grad_params = network:getParameters()
end


-- ===-------------------------------------------------------------------------------------------------------===
-- OPTIMIZATION CONFIGURATION AND LOGGING
-- ===-------------------------------------------------------------------------------------------------------===
local optim_logger = optim.Logger(logging_dir .. "/minibatch_cost.log")
optim_logger:setNames{"Training.cost", "Grad.norm"}
local train_dev_logger = optim.Logger(logging_dir .. "/train_dev_cost.log")
train_dev_logger:setNames{"Training.cost", "Dev.cost"}
rmsprop_config = { learningRate=args.learning_rate }


-- ===-------------------------------------------------------------------------------------------------------===
-- SERIALIZATION CONFIGURATION
-- ===-------------------------------------------------------------------------------------------------------===
print("Configuring Serialization...")
local serialize_data = {}
serialize_data.network      = nn.Serial(network)
serialize_data.epoch        = epoch
serialize_data.args         = args
serialize_data.dev_cost     = -1
serialize_data.min_dev_cost = -1


-- ===-------------------------------------------------------------------------------------------------------===
-- MODEL TRAINING
-- ===-------------------------------------------------------------------------------------------------------===
local num_minibatches = training_data.contexts:size()[1] / args.minibatch_size
local num_dev_minibatches = dev_data.contexts:size()[1] / args.minibatch_size

local epoch = 1 
while epoch <= args.num_epochs do
    print("")
    print("")
    print("===> Beginning Epoch #" .. epoch .. ":")

    local epoch_timer = torch.Timer()

    -- Marks the network as begin trained, because some modules like dropout behave differently in training and evaluation
    network:training() 

    local shuffle     = torch.randperm(training_data.contexts:size()[1])
    local shuffle_ind = 1
    local train_cost  = 0

    -- Iterate through the training set once
    for i = 1, num_minibatches do
        
        local feats, targets = sample_minibatch(shuffle, shuffle_ind, training_data, context_size)
        shuffle_ind = shuffle_ind + args.minibatch_size

        -- Optimization with optim and rmsprop
        local feval = function(x)
            if x ~= trainable_params then
                trainable_params:copy(x)
            end

            grad_params:zero()

            -- Forward pass
            local outputs = network:forward(feats)
            local cost    = criterion:forward(outputs, targets)
            train_cost    = train_cost + cost

            -- Backward pass
            local dcost_doutputs = criterion:backward(outputs, targets)
            network:backward(feats, dcost_doutputs)
            local norm    = torch.norm(grad_params)

            -- Clip grad params
            if args.step_clipping > 0 then
                norm = network:gradParamClip(args.step_clipping)
            end

            optim_logger:add{cost, norm}

            return cost, grad_params
        end

        optim.rmsprop(feval, trainable_params, rmsprop_config)

        xlua.progress(i, num_minibatches)
    end

    if cutorch then cutorch.synchronize() end -- Wait till all GPU code is finished
    time = epoch_timer:time()
    print("===> Epoch # " .. epoch .. " finished: " .. time['real'] .. "seconds passed")

    -- Marks network as being evaluated, since some modules like dropout behave differently
    network:evaluate()

    -- Get development cost
    print("===> Evaluating development data cost")
    local dev_cost = 0
    shuffle     = torch.randperm(dev_data.contexts:size()[1])
    shuffle_ind = 1
    local confusion = optim.ConfusionMatrix(ix_to_tag)
    for i = 1, num_dev_minibatches do
        local feats, targets = sample_minibatch(shuffle, shuffle_ind, dev_data, context_size)

        local output = network:forward(feats)
        local err    = criterion:forward(output, targets)
        dev_cost     = dev_cost + err

        for j = 1, args.minibatch_size do
            confusion:add(output[j], targets[j])
        end
    end

    confusion:render("score", true)

    -- Log costs
    train_cost = train_cost / num_minibatches
    dev_cost   = dev_cost / num_dev_minibatches 
    train_dev_logger:add{train_cost, dev_cost}
    print("\tTrain Cost: " .. train_cost)
    print("\tDev Cost: " .. dev_cost)

    -- Serialize
    serialize_data.epoch        = epoch
    serialize_data.dev_cost     = dev_cost
    serialize_data.args         = args
    serialize_data.min_dev_cost = min_dev_cost
    local filename = logging_dir .. "/model_epoch" .. epoch .. ".t7"
    torch.save(filename, serialize_data)

    -- Learning rate decay
    if args.learning_rate_decay > 0 and args.learning_rate_decay < 1 then
        local new_lr                = args.learning_rate * args.learning_rate_decay
        print("===> Learning rate decayed from " .. args.learning_rate .. " to " .. new_lr)
        args.learning_rate          = new_lr
        rmsprop_config.learningRate = new_lr
    end

    epoch = epoch + 1
    collectgarbage()
end 
