require 'torch'
require 'pl'

function make_dataset_from_file(filename, word_to_ix, vocab_size, tag_to_ix, tag_set_size)
    local input_file = torch.DiskFile(filename, "r"):quiet()
    local line       = input_file:readString("*l")

    local instance_tensor_size = 10000
    local instance_counter = 1
    local contexts = torch.IntTensor(instance_tensor_size, 2 * args.context_size + 1) -- Start at size 10000, double space as needed
    local targets  = torch.IntTensor(instance_tensor_size)

    while not input_file:hasError() do
        local words = {} 
        local tags  = {}

        -- Read the next document and its tags from the file
        while line ~= args.stop_token do
            local word, _, tag = stringx.rpartition(line, "/")
            words[#words + 1] = word
            tags[#tags + 1]   = tag
            line = input_file:readString("*l")
        end

        -- Point line to the next line to read (past the stop token)
        while line == args.stop_token do
            line = input_file:readString("*l")
        end

        -- Add all words to vocab
        for i = 1, #words do
            local word = words[i]
            if not word_to_ix[word] then
                word_to_ix[word] = vocab_size + 1
                vocab_size       = vocab_size + 1
            end
        end

        -- Add all tags to the set
        for i = 1, #tags do
            local tag = tags[i]
            if not tag_to_ix[tag] then
                tag_to_ix[tag] = tag_set_size + 1
                tag_set_size   = tag_set_size + 1
            end
        end

        -- Make the contexts and add them to the instance tensor
        for i = args.context_size+1, #words - args.context_size do
            local context = {}
            for k = args.context_size, 1, -1 do
                context[args.context_size - k + 1] = word_to_ix[words[i - k]]
            end
            local target_word = word_to_ix[words[i]]
            local target_tag  = tag_to_ix[tags[i]]
            context[args.context_size + 1] = target_word
            for k = 1, args.context_size do
                context[args.context_size + k + 1] = word_to_ix[words[i + k]]
            end

            -- Add the new context to the instance tensor
            contexts[instance_counter] = torch.IntTensor(context)
            targets[instance_counter]  = target_tag
            instance_counter = instance_counter + 1
            if instance_counter >= contexts:size()[1] then
                -- Need to resize the instances tensors, double the space we have
                contexts:resize(contexts:size()[1] * 2, contexts:size()[2])
                targets:resize(targets:size()[1] * 2)
            end
        end
    end

    -- Slice off the excess space
    contexts = contexts[{ {1, instance_counter - 1}, {} }]
    targets  = targets[{ {1, instance_counter - 1} }]

    return contexts, targets, word_to_ix, vocab_size, tag_to_ix, tag_set_size
end

cmd = torch.CmdLine()

cmd:option("--training_data", "", "Training data text file")
cmd:option("--dev_data", "", "Dev data text file")
cmd:option("--test_data", "", "Test data text file")
cmd:option("--output", "", "Output data file (.t7 extension)")
cmd:option("--context_size", 2, "Number of words on each side of a target word to include in context")
cmd:option("--stop_token", "=====", "Token that separates documents (contexts should not cross this token)")

args = cmd:parse(arg or {})

local word_to_ix = {}
local tag_to_ix  = {}
local vocab_size = 0
local tag_set_size = 0

local training_contexts, training_targets, word_to_ix, vocab_size, tag_to_ix, tag_set_size = make_dataset_from_file(args.training_data, word_to_ix, vocab_size, tag_to_ix, tag_set_size)
local dev_contexts, dev_targets, word_to_ix, vocab_size, tag_to_ix, tag_set_size = make_dataset_from_file(args.dev_data, word_to_ix, vocab_size, tag_to_ix, tag_set_size)
local test_contexts, test_targets, word_to_ix, vocab_size, tag_to_ix, tag_set_size = make_dataset_from_file(args.test_data, word_to_ix, vocab_size, tag_to_ix, tag_set_size)

local dataset = {
    training_data={
        contexts=training_contexts,
        targets=training_targets
    },
    dev_data={
        contexts=dev_contexts,
        targets=dev_targets
    },
    test_data={
        contexts=test_contexts,
        targets=test_targets
    },
    word_to_ix=word_to_ix,
    vocab_size=vocab_size,
    tag_to_ix=tag_to_ix,
    tag_set_size=tag_set_size
}

torch.save(args.output, dataset)

print("# of words: " .. vocab_size)
print("# of tags: " .. tag_set_size)
print("Writing vocab list to vocab.txt.  Use this with output_word_vectors script to initialize the Lookup table weights in the model")
print("Writing dataset file to " .. args.output)

local outfile = torch.DiskFile("vocab.txt", "w")
for word, idx in pairs(word_to_ix) do
    outfile:writeString(word .. "\n")
end
