 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'cudnn'
require 'optim'
require 'paths'
require 'hdf5'

package.path = package.path .. ';src/?.lua' .. ';src/utils/?.lua' .. ';src/model/?.lua' .. ';src/optim/?.lua'
require 'cnn'
require 'LSTM'
require 'reshaper'
require 'output_projector'
require 'criterion'
require 'model_utils'
require 'optim_adadelta'
require 'optim_sgd'
require 'optim_checkgrad'
require 'memory'
require 'mybatchnorm'

local model = torch.class('Model')

--[[ Args: 
-- config.load_model
-- config.model_dir
-- config.dropout
-- config.encoder_num_hidden
-- config.encoder_num_layers
-- config.decoder_num_layers
-- config.target_vocab_size
-- config.target_embedding_size
-- config.max_encoder_l_w
-- config.max_decoder_l
-- config.input_feed
-- config.batch_size
--]]

-- init
function model:__init()
    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
end

-- load model from model_path
function model:load(model_path, config)
    config = config or {}

    -- Build model

    assert(paths.filep(model_path), string.format('Model %s does not exist!', model_path))

    local checkpoint = torch.load(model_path)
    local model, model_config = checkpoint[1], checkpoint[2]
    preallocateMemory(config.prealloc)
    self.cnn_model = model[1]:double()
    self.encoder_fine_fw = model[2]:double()
    self.encoder_fine_bw = model[3]:double()
    self.encoder_coarse_fw = model[4]:double()
    self.encoder_coarse_bw = model[5]:double()
    self.reshaper = model[6]:double()
    self.decoder = model[7]:double()      
    self.output_projector = model[8]:double()
    self.global_step = checkpoint[3]
    self.optim_state = checkpoint[4]
    id2vocab = checkpoint[5]
    local reward_baselines = checkpoint[6]
    id2vocab_source = checkpoint[7]
    self.reward_baselines = {}--reward_baselines or {}

    -- evaluate batch norm layers
    --local bn_nodes, container_nodes = self.cnn_model:findModules('nn.SpatialBatchNormalization')
    --assert (#bn_nodes == 5 or #bn_nodes == 0)
    --for i = 1, #bn_nodes do
    --    --Search the container for the current threshold node
    --    for j = 1, #(container_nodes[i].modules) do
    --        if container_nodes[i].modules[j] == bn_nodes[i] then
    --            -- Replace with a new instance
    --            container_nodes[i].modules[j] = nn.BatchNorm(bn_nodes[i])
    --            print ('Fixing Batch Normalization')
    --        end
    --    end
    --end
    -- Load model structure parameters
    self.cnn_feature_size = 100
    self.dropout = model_config.dropout
    self.encoder_num_hidden = model_config.encoder_num_hidden
    self.encoder_num_layers = model_config.encoder_num_layers
    self.decoder_num_hidden = self.encoder_num_hidden * 2
    self.decoder_num_layers = model_config.decoder_num_layers
    self.source_vocab_size = #id2vocab_source+4
    self.target_vocab_size = #id2vocab+4
    self.source_embedding_size = model_config.source_embedding_size
    self.target_embedding_size = model_config.target_embedding_size
    self.input_feed = model_config.input_feed
    self.prealloc = config.prealloc
    self.fine = model_config.fine or {1,32}

    self.entropy_scale = config.entropy_scale or model_config.entropy_scale
    self.semi_sampling_p = config.semi_sampling_p or model_config.semi_sampling_p
    self.baseline_lr = config.baseline_lr or model_config.baseline_lr
    self.discount = config.discount or model_config.discount

    self.max_encoder_fine_l_w = config.max_encoder_fine_l_w or model_config.max_encoder_fine_l_w
    self.max_encoder_fine_l_h = config.max_encoder_fine_l_h or model_config.max_encoder_fine_l_h
    self.max_encoder_coarse_l_w = config.max_encoder_coarse_l_w or model_config.max_encoder_coarse_l_w
    self.max_encoder_coarse_l_h = config.max_encoder_coarse_l_h or model_config.max_encoder_coarse_l_h
    self.max_decoder_l = config.max_decoder_l or model_config.max_decoder_l
    self.batch_size = config.batch_size or model_config.batch_size
    self.pre_word_vecs_enc = ''--config.pre_word_vecs_enc
    self.pre_word_vecs_dec = ''--config.pre_word_vecs_dec

    --self.decoder = createLSTM(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers, self.dropout, true, self.input_feed, true, self.target_vocab_size, self.batch_size, {self.fine[1]*self.fine[2], self.max_encoder_coarse_l_h*self.max_encoder_coarse_l_w}, 'decoder',
        --os.exit(1)
    self:_build()
end

-- create model with fresh parameters
function model:create(config)
    self.cnn_feature_size = 100
    self.dropout = config.dropout
    self.encoder_num_hidden = config.encoder_num_hidden
    self.encoder_num_layers = config.encoder_num_layers
    self.decoder_num_hidden = config.encoder_num_hidden * 2
    self.decoder_num_layers = config.decoder_num_layers
    self.source_vocab_size = config.source_vocab_size
    self.target_vocab_size = config.target_vocab_size
    self.source_embedding_size = config.source_embedding_size
    self.target_embedding_size = config.target_embedding_size
    self.max_encoder_fine_l_w = config.max_encoder_fine_l_w
    self.max_encoder_fine_l_h = config.max_encoder_fine_l_h
    self.max_encoder_coarse_l_w = config.max_encoder_coarse_l_w
    self.max_encoder_coarse_l_h = config.max_encoder_coarse_l_h
    self.max_decoder_l = config.max_decoder_l
    self.input_feed = config.input_feed
    self.batch_size = config.batch_size
    self.entropy_scale = config.entropy_scale
    self.semi_sampling_p = config.semi_sampling_p
    self.baseline_lr = config.baseline_lr
    self.discount = config.discount
    self.prealloc = config.prealloc
    self.fine = {1,32}
    self.pre_word_vecs_enc = config.pre_word_vecs_enc
    self.pre_word_vecs_dec = config.pre_word_vecs_dec

    preallocateMemory(config.prealloc)

    -- Word model, input size: (batch_size, 1, 32, width), output size: (batch_size, 1, 32, word_embedding_size)
    self.word_model = createWordModel(self.source_vocab_size, self.source_embedding_size)
    -- CNN model, input size: (batch_size, 1, 32, width), output size: (batch_size, sequence_length, 100)
    self.cnn_model = createCNNModel(self.encoder_num_hidden * 2)
    -- createLSTM(input_size, num_hidden, num_layers, dropout, use_attention, input_feed, use_lookup, vocab_size)
    self.encoder_fine_fw = createLSTM(self.source_embedding_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, {}, 'encoder-fine-fw')
    self.encoder_fine_bw = createLSTM(self.source_embedding_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, {}, 'encoder-fine-bw')
    self.encoder_coarse_fw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, {}, 'encoder-coarse-fw')
    self.encoder_coarse_bw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, {}, 'encoder-coarse-bw')
    self.decoder = createLSTM(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers, self.dropout, true, self.input_feed, true, self.target_vocab_size, self.batch_size, {self.fine[1]*self.fine[2], self.max_encoder_coarse_l_h*self.max_encoder_coarse_l_w}, 'decoder',
        self.entropy_scale, self.semi_sampling_p)
    self.output_projector = createOutputUnit(self.decoder_num_hidden, self.target_vocab_size)
    self.reshaper = nn.Reshaper(self.fine)
    self.global_step = 0
    self._init = true

    self.optim_state = {}
    self.optim_state.learningRate = config.learning_rate
    self.reward_baselines = {}
    self:_build()
end

-- build
function model:_build()
    log(string.format('cnn_featuer_size: %d', self.cnn_feature_size))
    log(string.format('dropout: %f', self.dropout))
    log(string.format('encoder_num_hidden: %d', self.encoder_num_hidden))
    log(string.format('encoder_num_layers: %d', self.encoder_num_layers))
    log(string.format('decoder_num_hidden: %d', self.decoder_num_hidden))
    log(string.format('decoder_num_layers: %d', self.decoder_num_layers))
    log(string.format('target_vocab_size: %d', self.target_vocab_size))
    log(string.format('target_embedding_size: %d', self.target_embedding_size))
    log(string.format('max_encoder_fine_l_w: %d', self.max_encoder_fine_l_w))
    log(string.format('max_encoder_fine_l_h: %d', self.max_encoder_fine_l_h))
    log(string.format('max_encoder_coarse_l_w: %d', self.max_encoder_coarse_l_w))
    log(string.format('max_encoder_coarse_l_h: %d', self.max_encoder_coarse_l_h))
    log(string.format('max_decoder_l: %d', self.max_decoder_l))
    log(string.format('input_feed: %s', self.input_feed))
    log(string.format('batch_size: %d', self.batch_size))
    log(string.format('entropy_scale: %f', self.entropy_scale))
    log(string.format('semi_sampling_p: %f', self.semi_sampling_p))
    log(string.format('baseline_lr: %f', self.baseline_lr))
    log(string.format('discount: %f', self.discount))
    log(string.format('prealloc: %s', self.prealloc))


    self.config = {}
    self.config.dropout = self.dropout
    self.config.encoder_num_hidden = self.encoder_num_hidden
    self.config.encoder_num_layers = self.encoder_num_layers
    self.config.decoder_num_hidden = self.decoder_num_hidden
    self.config.decoder_num_layers = self.decoder_num_layers
    self.config.target_vocab_size = self.target_vocab_size
    self.config.source_vocab_size = self.source_vocab_size
    self.config.source_embedding_size = self.source_embedding_size
    self.config.target_embedding_size = self.target_embedding_size
    self.config.max_encoder_fine_l_w = self.max_encoder_fine_l_w
    self.config.max_encoder_fine_l_h = self.max_encoder_fine_l_h
    self.config.max_encoder_coarse_l_w = self.max_encoder_coarse_l_w
    self.config.max_encoder_coarse_l_h = self.max_encoder_coarse_l_h
    self.config.max_decoder_l = self.max_decoder_l
    self.config.input_feed = self.input_feed
    self.config.batch_size = self.batch_size
    self.config.entropy_scale = self.entropy_scale
    self.config.semi_sampling_p = self.semi_sampling_p
    self.config.baseline_lr = self.baseline_lr
    self.config.discount = self.discount
    self.config.fine = self.fine
    self.config.prealloc = self.prealloc


    if self.optim_state == nil then
        self.optim_state = {}
    end
    self.criterion = createCriterion(self.target_vocab_size)

    -- convert to cuda if use gpu
    self.layers = {self.cnn_model, self.word_model, self.encoder_fine_fw, self.encoder_fine_bw, self.encoder_coarse_fw, self.encoder_coarse_bw, self.reshaper, self.decoder, self.output_projector}
    for i = 1, #self.layers do
        localize(self.layers[i])
    end
    localize(self.criterion)

    self.context_fine_proto = localize(torch.zeros(self.batch_size, self.max_encoder_fine_l_w*self.max_encoder_fine_l_h, 2*self.encoder_num_hidden))
    self.context_coarse_proto = localize(torch.zeros(self.batch_size, self.max_encoder_coarse_l_w*self.max_encoder_coarse_l_h, 2*self.encoder_num_hidden))
    self.encoder_fine_fw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_fine_l_w*self.max_encoder_fine_l_h, self.encoder_num_hidden))
    self.encoder_fine_bw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_fine_l_w*self.max_encoder_fine_l_h, self.encoder_num_hidden))
    self.encoder_coarse_fw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_coarse_l_w*self.max_encoder_coarse_l_h, self.encoder_num_hidden))
    self.encoder_coarse_bw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_coarse_l_w*self.max_encoder_coarse_l_h, self.encoder_num_hidden))
    self.reshaper_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_coarse_l_w*self.max_encoder_coarse_l_h, self.fine[1]*self.fine[2], 2*self.encoder_num_hidden))
    self.cnn_fine_grad_proto = localize(torch.zeros(self.max_encoder_fine_l_h, self.batch_size, self.max_encoder_fine_l_w, self.source_embedding_size))
    self.cnn_coarse_grad_proto = localize(torch.zeros(self.max_encoder_coarse_l_h, self.batch_size, self.max_encoder_coarse_l_w, self.cnn_feature_size))

    local num_params = 0
    self.params, self.grad_params = {}, {}
    for i = 1, #self.layers do
        local p, gp = self.layers[i]:getParameters()
        if p:dim() ~= 0 then
            if self._init then
                p:uniform(-0.05,0.05)
            end
            num_params = num_params + p:size(1)
            self.params[i] = p
            self.grad_params[i] = gp
        end
    end
    word_vec_layers = {}
    self.word_model:apply(function (m)
        if m.name == 'word_vecs_encoder' then word_vec_layers[1] = m end
    end)
    self.decoder:apply(function (m)
        if m.name == 'word_vecs_decoder' then word_vec_layers[2] = m end
    end)
    assert (word_vec_layers[1] ~= nil)
    assert (word_vec_layers[2] ~= nil)
    if self.pre_word_vecs_enc:len() > 0 then   
        log(string.format('Load source word embeddings from: %s', self.pre_word_vecs_enc))
        local f = hdf5.open(self.pre_word_vecs_enc)     
        local pre_word_vecs = f:read('word_vecs'):all()
        for i = 1, pre_word_vecs:size(1) do
            word_vec_layers[1].weight[i]:copy(pre_word_vecs[i])
        end      
    end
    if self.pre_word_vecs_dec:len() > 0 then      
        log(string.format('Load target word embeddings from: %s', self.pre_word_vecs_dec))
        local f = hdf5.open(self.pre_word_vecs_dec)     
        local pre_word_vecs = f:read('word_vecs'):all()
        for i = 1, pre_word_vecs:size(1) do
            word_vec_layers[2].weight[i]:copy(pre_word_vecs[i])
        end      
    end
    log(string.format('Number of parameters: %d', num_params))

    self.decoder_clones = clone_many_times(self.decoder, self.max_decoder_l)
    self.encoder_fine_fw_clones = clone_many_times(self.encoder_fine_fw, self.max_encoder_fine_l_w)
    self.encoder_fine_bw_clones = clone_many_times(self.encoder_fine_bw, self.max_encoder_fine_l_w)
    self.encoder_coarse_fw_clones = clone_many_times(self.encoder_coarse_fw, self.max_encoder_coarse_l_w)
    self.encoder_coarse_bw_clones = clone_many_times(self.encoder_coarse_bw, self.max_encoder_coarse_l_w)
   
    self.softmax_attn_clones={}
    for i = 1, #self.decoder_clones do
        local decoder = self.decoder_clones[i]
        local decoder_attn
        decoder:apply(function (layer) 
            if layer.name == 'decoder_attn' then
                decoder_attn = layer
            end
        end)
        assert (decoder_attn)
        decoder:apply(function (layer)
            if layer.name == 'mul_attn' then
                self.softmax_attn_clones[i] = layer
            end
        end)
        assert (self.softmax_attn_clones[i])
    end
    for i = 1, #self.encoder_fine_fw_clones do
        if self.encoder_fine_fw_clones[i].apply then
            self.encoder_fine_fw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_fine_fw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.encoder_fine_bw_clones do
        if self.encoder_fine_bw_clones[i].apply then
            self.encoder_fine_bw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_fine_bw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.encoder_coarse_fw_clones do
        if self.encoder_coarse_fw_clones[i].apply then
            self.encoder_coarse_fw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_coarse_fw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.encoder_coarse_bw_clones do
        if self.encoder_coarse_bw_clones[i].apply then
            self.encoder_coarse_bw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_coarse_bw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    self.sampler_coarse_clones = {}
    self.sampler_fine_clones = {}
    for i = 1, #self.decoder_clones do
        if self.decoder_clones[i].apply then
            self.decoder_clones[i]:apply(function (m) 
                if m.name == 'sampler_coarse' then
                    m.prealloc = nil
                    self.sampler_coarse_clones[i] = m
                    m.semi_sampling_p = self.semi_sampling_p
                    m.entropy_scale = self.entropy_scale
                elseif m.name == 'sampler_fine' then
                    m.prealloc = nil
                    self.sampler_fine_clones[i] = m
                    m.semi_sampling_p = 1.0
                    m.entropy_scale = self.entropy_scale
                end
            end)
            self.decoder_clones[i]:apply(function (m) m:setReuse() end)
            if self.prealloc then self.decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    -- initalial states
    local encoder_h_init = localize(torch.zeros(self.batch_size, self.encoder_num_hidden))
    local decoder_h_init = localize(torch.zeros(self.batch_size, self.decoder_num_hidden))

    self.init_fwd_enc = {}
    self.init_bwd_enc = {}
    self.init_fwd_dec = {}
    self.init_bwd_dec = {}
    for L = 1, self.encoder_num_layers do
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
    end
    if self.input_feed then
        table.insert(self.init_fwd_dec, decoder_h_init:clone())
    end
    table.insert(self.init_bwd_dec, decoder_h_init:clone())
    for L = 1, self.decoder_num_layers do
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- memory cell
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- hidden state
        table.insert(self.init_bwd_dec, decoder_h_init:clone())
        table.insert(self.init_bwd_dec, decoder_h_init:clone()) 
    end
    self.dec_offset = 4 -- offset depends on input feeding
    if self.input_feed then
        self.dec_offset = self.dec_offset + 1
    end
    self.init_beam = false
    self.visualize = false
    collectgarbage()
end

-- one step 
function model:step(batch, forward_only, beam_size, trie)
    if forward_only then
        self.val_batch_size = self.batch_size
        beam_size = beam_size or 1 -- default argmax
        beam_size = math.min(beam_size, self.target_vocab_size)
        if not self.init_beam then
            self.init_beam = true
            local beam_decoder_h_init = localize(torch.zeros(self.val_batch_size*beam_size, self.decoder_num_hidden))
            self.beam_scores = localize(torch.zeros(self.val_batch_size, beam_size))
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.beam_init_fwd_dec = {}
            if self.input_feed then
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone())
            end
            for L = 1, self.decoder_num_layers do
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- memory cell
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- hidden state
            end
            self.trie_locations = {}
        else
            self.beam_scores:zero()
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
        end
    else
        if self.init_beam then
            self.init_beam = false
            self.trie_locations = {}
            self.beam_init_fwd_dec = {}
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
            self.beam_scores = nil
            collectgarbage()
        end
    end
    local input_batch = localize(batch[1])
    local target_batch = localize(batch[2])
    local target_eval_batch = localize(batch[3])
    local num_nonzeros = batch[4]
    local img_paths
    if self.visualize then
        img_paths = batch[5]
    end

    local batch_size = input_batch:size()[1]
    local target_l = target_batch:size()[2]

    assert(target_l <= self.max_decoder_l, string.format('max_decoder_l (%d) < target_l (%d)!', self.max_decoder_l, target_l))
    -- if forward only, then re-generate the target batch
    if forward_only then
        local target_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_batch)
        target_batch = target_batch_new
        local target_eval_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_eval_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_eval_batch)
        target_eval_batch = target_eval_batch_new
        target_l = self.max_decoder_l
    end

    if not forward_only then
        self.word_model:training()
        self.cnn_model:training()
        self.output_projector:training()
    else
        self.word_model:evaluate()
        self.cnn_model:evaluate()
        --self.cnn_model:training()
        self.output_projector:evaluate()
    end

    local feval = function(p) --cut off when evaluate
        --for i = 1, #self.params do
        --    if p[i] ~= nil then
        --        self.params[i]:copy(p[i])
        --    end
        --end
        target = target_batch:transpose(1,2)
        target_eval = target_eval_batch:transpose(1,2)
        local word_embeddings_list = self.word_model:forward(input_batch)
        -- encode fine
        local counter = 1
        local imgH_fine = #word_embeddings_list
        local imgW_fine = word_embeddings_list[1]:size()[2]
        local context_fine = self.context_fine_proto[{{1, batch_size}, {1, imgW_fine*imgH_fine}}]
        assert(imgW_fine <= self.max_encoder_fine_l_w, string.format('max_encoder_fine_l_w (%d) < imgW_fine (%d)!', self.max_encoder_fine_l_w, imgW_fine))
        for i = 1, imgH_fine do
            local cnn_output = word_embeddings_list[i] --1, imgW, 100
            local source = cnn_output:transpose(1,2) -- imgW,1,100
            -- forward encoder
            local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
            for t = 1, imgW_fine do
                counter = (i-1)*imgW_fine + t
                if not forward_only then
                    self.encoder_fine_fw_clones[t]:training()
                else
                    self.encoder_fine_fw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = self.encoder_fine_fw_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context_fine[{{},counter, {1, self.encoder_num_hidden}}]:copy(out[#out])
            end
            local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, imgW_fine+1)
            for t = imgW_fine, 1, -1 do
                counter = (i-1)*imgW_fine + t
                if not forward_only then
                    self.encoder_fine_bw_clones[t]:training()
                else
                    self.encoder_fine_bw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                local out = self.encoder_fine_bw_clones[t]:forward(encoder_input)
                rnn_state_enc_bwd[t] = out
                context_fine[{{},counter, {1+self.encoder_num_hidden, 2*self.encoder_num_hidden}}]:copy(out[#out])
            end
        end
        -- context_fine ready, batch_size, imgH_fine*imgW_fine, 2*self.encoder_num_hidden
    --local raw_features = inputs[1] --(batch_size, imgH, imgW, embedding_size)
        local cnn_output_coarse_list = self.cnn_model:forward(context_fine:contiguous():view(-1, imgH_fine, imgW_fine, 2*self.encoder_num_hidden)) -- list of (batch_size, W, 100)
        -- encode coarse
        local counter = 1
        local imgH_coarse = #cnn_output_coarse_list
        local imgW_coarse = cnn_output_coarse_list[1]:size()[2]
        local context_coarse = self.context_coarse_proto[{{1, batch_size}, {1, imgW_coarse*imgH_coarse}}]
        assert(imgW_coarse <= self.max_encoder_coarse_l_w, string.format('max_encoder_coarse_l_w (%d) < imgW_coarse (%d)!', self.max_encoder_coarse_l_w, imgW_coarse))
        for i = 1, imgH_coarse do
            local cnn_output = cnn_output_coarse_list[i] --1, imgW, 100
            local source = cnn_output:transpose(1,2) -- imgW,1,100
            -- forward encoder
            local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
            for t = 1, imgW_coarse do
                counter = (i-1)*imgW_coarse + t
                if not forward_only then
                    self.encoder_coarse_fw_clones[t]:training()
                else
                    self.encoder_coarse_fw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = self.encoder_coarse_fw_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context_coarse[{{},counter, {1, self.encoder_num_hidden}}]:copy(out[#out])
            end
            local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, imgW_coarse+1)
            for t = imgW_coarse, 1, -1 do
                counter = (i-1)*imgW_coarse + t
                if not forward_only then
                    self.encoder_coarse_bw_clones[t]:training()
                else
                    self.encoder_coarse_bw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                local out = self.encoder_coarse_bw_clones[t]:forward(encoder_input)
                rnn_state_enc_bwd[t] = out
                context_coarse[{{},counter, {1+self.encoder_num_hidden, 2*self.encoder_num_hidden}}]:copy(out[#out])
            end
        end -- context_coarse and conext_fine ready
        -- input: imgH_coarse, imgW_coarse, imgH_fine, imgW_fine, fine
        -- context_fine: batch_size, imgH_fine*imgW_fine, 2*encoder_num_hidden
        -- desired output: context_coarse: batch_size, imgH_coarse*imgW_coarse, fine, 2*encoder_num_hidden
        self.reshaper.imgW_coarse = imgW_coarse
        self.reshaper.imgH_coarse = imgH_coarse
        self.reshaper.imgW_fine = imgW_fine
        self.reshaper.imgH_fine = imgH_fine
        local reshape_context_fine = self.reshaper:forward(context_fine)
        -- context_coarse and reshape_context_fine ready
        local preds = {}
        local indices
        local rnn_state_dec
        -- forward_only == true, beam search
        if forward_only then
            local beam_replicate = function(hidden_state)
                if hidden_state:dim() == 1 then
                    local batch_size = hidden_state:size()[1]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1):expand(batch_size, beam_size)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(-1)
                elseif hidden_state:dim() == 2 then
                    local batch_size = hidden_state:size()[1]
                    local num_hidden = hidden_state:size()[2]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, num_hidden):expand(batch_size, beam_size, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, num_hidden)
                elseif hidden_state:dim() == 3 then
                    local batch_size = hidden_state:size()[1]
                    local source_l = hidden_state:size()[2]
                    local num_hidden = hidden_state:size()[3]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, source_l, num_hidden):expand(batch_size, beam_size, source_l, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, source_l, num_hidden)
                elseif hidden_state:dim() == 4 then
                    local batch_size = hidden_state:size()[1]
                    local source_l = hidden_state:size()[2]
                    local source_l2 = hidden_state:size()[3]
                    local num_hidden = hidden_state:size()[4]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, source_l, source_l2, num_hidden):expand(batch_size, beam_size, source_l, source_l2, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, source_l, source_l2, num_hidden)
                else
                    assert(false, 'does not support ndim except for 2 and 3 and 4')
                end
            end
            rnn_state_dec = reset_state(self.beam_init_fwd_dec, batch_size, 0)
            local beam_context_coarse = beam_replicate(context_coarse)
            local beam_context_fine = beam_replicate(reshape_context_fine)
            local decoder_input
            local beam_input
            for t = 1, target_l do
                self.decoder_clones[t]:evaluate()
                if t == 1 then
                    beam_input = target[t]
                    decoder_input = {beam_input, context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                else
                    decoder_input = {beam_input, beam_context_coarse, beam_context_fine, table.unpack(rnn_state_dec[t-1])}
                end
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                local top_out = out[#out]
                local probs = self.output_projector:forward(top_out) -- t~=0, batch_size*beam_size, vocab_size; t=0, batch_size,vocab_size
                local current_indices, raw_indices
                local beam_parents
                if t == 1 then
                    -- probs batch_size, vocab_size
                    self.beam_scores, raw_indices = probs:topk(beam_size, true)
                    raw_indices = localize(raw_indices:double())
                    current_indices = raw_indices
                else
                    -- batch_size*beam_size, vocab_size
                    probs:select(2,1):maskedFill(beam_input:eq(1), 0) -- once padding or EOS encountered, stuck at that point
                    probs:select(2,1):maskedFill(beam_input:eq(3), 0)
                    local total_scores = (probs:view(batch_size, beam_size, self.target_vocab_size) + self.beam_scores[{{1,batch_size}, {}}]:view(batch_size, beam_size, 1):expand(batch_size, beam_size, self.target_vocab_size)):view(batch_size, beam_size*self.target_vocab_size) -- batch_size, beam_size * target_vocab_size
                    self.beam_scores, raw_indices = total_scores:topk(beam_size, true) --batch_size, beam_size
                    raw_indices = localize(raw_indices:double())
                    raw_indices:add(-1)
                    if use_cuda then
                        current_indices = raw_indices:double():fmod(self.target_vocab_size):cuda()+1 -- batch_size, beam_size for current vocab
                    else
                        current_indices = raw_indices:fmod(self.target_vocab_size)+1 -- batch_size, beam_size for current vocab
                    end
                end
                beam_parents = localize(raw_indices:int()/self.target_vocab_size+1) -- batch_size, beam_size for number of beam in each batch
                beam_input = current_indices:view(batch_size*beam_size)
                table.insert(self.current_indices_history, current_indices:clone())
                table.insert(self.beam_parents_history, beam_parents:clone())

                if self.input_feed then
                    local top_out = out[#out] -- batch_size*beam_size, hidden_dim
                    if t == 1 then
                        top_out = beam_replicate(top_out)
                    end
                    table.insert(next_state, top_out:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1)))
                end
                for j = 1, #out-1 do
                    local out_j = out[j] -- batch_size*beam_size, hidden_dim
                    if t == 1 then
                        out_j = beam_replicate(out_j)
                    end
                    table.insert(next_state, out_j:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1)))
                end
                rnn_state_dec[t] = next_state
            end
        else -- forward_only == false
            -- set decoder states
            rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
            for t = 1, target_l do
                self.decoder_clones[t]:training()
                --if t == 1 then 
                --    output_flag= true
                --else
                --    output_flag = false
                --end
                local decoder_input
                decoder_input = {target[t], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                if self.input_feed then
                    table.insert(next_state, out[#out])
                end
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end
        end
        local loss, accuracy = 0.0, 0.0
        if forward_only then
            -- final decoding
            local labels = localize(torch.zeros(batch_size, target_l)):fill(1)
            local scores, indices = torch.max(self.beam_scores[{{1,batch_size},{}}], 2) -- batch_size, 1
            indices = localize(indices:double())
            scores = scores:view(-1) -- batch_size
            indices = indices:view(-1) -- batch_size
            local current_indices = self.current_indices_history[#self.current_indices_history]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
            for t = target_l, 1, -1 do
                labels[{{1,batch_size}, t}]:copy(current_indices)
                indices = self.beam_parents_history[t]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                if t > 1 then
                    current_indices = self.current_indices_history[t-1]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                end
            end
            local word_err, labels_pred, labels_gold, labels_list_pred, labels_list_gold = evalHTMLErrRate(labels, target_eval_batch, self.visualize)
            accuracy = batch_size - word_err
            if self.visualize then
                -- get gold score
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                local gold_scores = localize(torch.zeros(batch_size))
                for t = 1, target_l do
                    self.decoder_clones[t]:evaluate()
                    --self.decoder_clones[t]:training()
                    local decoder_input
                    decoder_input = {target[t], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    local next_state = {}
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    for j = 1, batch_size do
                        if target_eval[t][j] ~= 1 then
                            gold_scores[j] = gold_scores[j] + pred[j][target_eval[t][j]]
                        end
                    end

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
                -- use predictions to visualize attns
                local attn_probs = localize(torch.zeros(batch_size, target_l, imgW_fine*imgH_fine))
                local attn_positions_h = localize(torch.zeros(batch_size, target_l))
                local attn_positions_w = localize(torch.zeros(batch_size, target_l))
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                for t = 1, target_l do
                    --self.decoder_clones[t]:training()
                    self.decoder_clones[t]:evaluate()
                    local decoder_input
                    if t == 1 then
                        decoder_input = {target[t], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                    else
                        decoder_input = {labels[{{1,batch_size},t-1}], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                    end
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    -- print attn
                    --attn_probs[{{}, t, {}}]:copy(self.softmax_attn_clones[t].output)
                    local _, attn_inds = torch.max(self.softmax_attn_clones[t].output:view(-1,imgH_coarse*imgW_coarse*self.fine[1]*self.fine[2]), 2) --batch_size, 1
                    attn_inds = attn_inds:view(-1) --batch_size
                    local i_H = math.floor((attn_inds[1]-1) / self.fine[1] / self.fine[2] / imgW_coarse) + 1
                    local i_W = math.floor((attn_inds[1]-1) / self.fine[1] / self.fine[2] - (i_H-1) * imgW_coarse) + 1
                    local i_H_fine = math.floor((attn_inds[1]-1 - self.fine[1]*self.fine[2]*((i_H-1)*imgW_coarse + i_W-1)) / self.fine[2]) + 1
                    local i_W_fine = math.floor(attn_inds[1]-1 - self.fine[1]*self.fine[2]*((i_H-1)*imgW_coarse+i_W-1)) - (i_H_fine-1)*self.fine[2] + 1
                    local score = pred[1][target_eval[t][1]]
                    if t == 11 then
                        output_flag = true
                    else
                        output_flag = false
                    end
                    --print (string.format('t:%d, label: %s, score: %f', t, split(labels_gold[1])[t], score))
                    --if true then
                    --    print (string.format('t:%d, coarse h:%d, coarse w:%d fine h:%d, fine w:%d', t, i_H, i_W, i_H_fine, i_W_fine))
                    --end
                    --for kk = 1, batch_size do
                    --    local counter = attn_inds[kk]
                    --    local p_i = math.floor((counter-1) / imgW_fine) + 1
                    --    local p_t = counter-1 - (p_i-1)*imgW_fine + 1
                    --    attn_positions_h[kk][t] = p_i
                    --    attn_positions_w[kk][t] = p_t
                    --    --print (string.format('%d, %d', p_i, p_t))
                    --end
                        --for kk = 1, fea_inds:size(1) do
                    local next_state = {}
                    --table.insert(preds, out[#out])
                    -- target_eval[t] --batch_size
                    --for j = 1, batch_size do
                    --    if target_eval[t][j] ~= 1 then
                    --        gold_scores[j] = gold_scores[j] + pred[j][target_eval[t][j]]
                    --    end
                    --end

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
                for i = 1, #img_paths do
                    self.visualize_file:write(string.format('%s\t%s\t%s\t%f\t%f\t\n', img_paths[i], labels_gold[i], labels_pred[i], scores[i], gold_scores[i]))
                    --for j = 1, target_l do
                    --    if labels[i][j] == 3 then
                    --        break
                    --    end
                    --    self.visualize_attn_file:write(string.format('%s\t%d\t%d\t', labels_list_pred[i][j], attn_positions_h[i][j], attn_positions_w[i][j]))
                    --    for k = 1, source_l*imgH do
                    --        self.visualize_attn_file:write(string.format('%f\t', attn_probs[i][j][k]))
                    --    end
                    --end
                    --self.visualize_attn_file:write('\n')
                end
                self.visualize_file:flush()
            end
                --self.visualize_attn_file:flush()
            --else -- forward_only and not visualize
                -- get gold score
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                local gold_scores = localize(torch.zeros(batch_size))
                for t = 1, target_l do
                    self.decoder_clones[t]:evaluate()
                    --self.decoder_clones[t]:training()
                    local decoder_input
                    decoder_input = {target[t], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    local next_state = {}
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    loss = loss + self.criterion:forward(pred, target_eval[t])/batch_size

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
        else
            local encoder_fine_fw_grads = self.encoder_fine_fw_grad_proto[{{1, batch_size}, {1, imgW_fine*imgH_fine}}]
            local encoder_fine_bw_grads = self.encoder_fine_bw_grad_proto[{{1, batch_size}, {1, imgW_fine*imgH_fine}}]
            local encoder_coarse_fw_grads = self.encoder_coarse_fw_grad_proto[{{1, batch_size}, {1, imgW_coarse*imgH_coarse}}]
            local encoder_coarse_bw_grads = self.encoder_coarse_bw_grad_proto[{{1, batch_size}, {1, imgW_coarse*imgH_coarse}}]
            local reshaper_grads = self.reshaper_grad_proto[{{1,batch_size}, {1,imgW_coarse*imgH_coarse}, {}, {}}]
            for i = 1, #self.grad_params do
                if self.grad_params[i] ~= nil then
                    self.grad_params[i]:zero()
                end
            end
            encoder_fine_fw_grads:zero()
            encoder_fine_bw_grads:zero()
            encoder_coarse_fw_grads:zero()
            encoder_coarse_bw_grads:zero()
            reshaper_grads:zero()
            local drnn_state_dec = reset_state(self.init_bwd_dec, batch_size)
            local rewards = nil -- current reward before subtracting baselines
            print (self.reward_baselines[1])
            for t = target_l, 1, -1 do
                if t == 1 then 
                    output_flag= true
                else
                    output_flag = false
                end
                    -- print attn
                    --attn_probs[{{}, t, {}}]:copy(self.softmax_attn_clones[t].output)
                    --local attn_vals, attn_inds = torch.min(self.softmax_attn_clones[t].output:view(-1,imgH_coarse*imgW_coarse*self.fine[1]*self.fine[2]), 2) --batch_size, 1
                    --attn_inds = attn_inds:view(-1) --batch_size
                    --attn_vals = attn_vals:view(-1)
                    --local i_H = math.floor((attn_inds[1]-1) / self.fine[1] / self.fine[2] / imgW_coarse) + 1
                    --local i_W = math.floor((attn_inds[1]-1) / self.fine[1] / self.fine[2] - (i_H-1) * imgW_coarse)
                    --if t==1 then
                    --    print (string.format('%d, %d', i_H, i_W))
                    --end
                local pred = self.output_projector:forward(preds[t]) -- batch_size, target_vocab_size
                pred:select(2,1):maskedFill(target_eval[t]:eq(1), 0)
                local rewards_raw = pred:gather(2, target_eval[t]:contiguous():view(batch_size,1))
                --if t== 1 then
                --    print (rewards_raw[1])
                --end
                local num_valid = batch_size - target_eval[t]:eq(1):sum()
                -- normed reward
                local rewards_norm
                if rewards == nil then
                    rewards = rewards_raw:clone()
                else
                    rewards = rewards:clone():mul(self.discount) + rewards_raw
                end
                rewards:maskedFill(target_eval[t]:eq(1), 0)
                if self.reward_baselines[t] == nil then
                    self.sampler_fine_clones[t]:reinforce(localize(torch.zeros(batch_size)))
                    self.sampler_coarse_clones[t]:reinforce(localize(torch.zeros(batch_size)))
                else
                    local rewards_norm = rewards:clone():add(-1.0*self.reward_baselines[t])
                    rewards_norm:maskedFill(target_eval[t]:eq(1), 0)
                    self.sampler_fine_clones[t]:reinforce(rewards_norm:clone():div(batch_size))
                    self.sampler_coarse_clones[t]:reinforce(rewards_norm:clone():div(batch_size))
                end
                    
                if num_valid > 0 then -- update baselines
                    local average_reward = rewards:sum() / num_valid
                    if self.reward_baselines[t] == nil then
                        self.reward_baselines[t] = average_reward
                    else
                        self.reward_baselines[t] = (1-self.baseline_lr)*self.reward_baselines[t] + self.baseline_lr*average_reward
                    end
                end
                local step_loss = self.criterion:forward(pred, target_eval[t])/batch_size
                loss = loss + step_loss
                local dl_dpred = self.criterion:backward(pred, target_eval[t])
                dl_dpred:div(batch_size)--:zero()
                local dl_dtarget = self.output_projector:backward(preds[t], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
                local decoder_input = {target[t], context_coarse, reshape_context_fine, table.unpack(rnn_state_dec[t-1])}
                local dlst = self.decoder_clones[t]:backward(decoder_input, drnn_state_dec)
                encoder_coarse_fw_grads:add(dlst[2][{{}, {}, {1,self.encoder_num_hidden}}])
                encoder_coarse_bw_grads:add(dlst[2][{{}, {}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
                reshaper_grads:add(dlst[3])
                drnn_state_dec[#drnn_state_dec]:zero()
                if self.input_feed then
                    drnn_state_dec[#drnn_state_dec]:copy(dlst[self.dec_offset-1])
                end     
                for j = self.dec_offset, #dlst do
                    drnn_state_dec[j-self.dec_offset+1]:copy(dlst[j])
                end
            end
            local cnn_fine_grad = self.cnn_fine_grad_proto[{{1,imgH_fine}, {1,batch_size}, {1,imgW_fine}, {}}]
            local cnn_coarse_grad = self.cnn_coarse_grad_proto[{{1,imgH_coarse}, {1,batch_size}, {1,imgW_coarse}, {}}]
            local encoder_fine_grads = self.reshaper:backward(context_fine, reshaper_grads)
            -- coarse
            -- forward directional encoder
            for i = 1, imgH_coarse do
                local cnn_output_coarse = cnn_output_coarse_list[i]
                local source = cnn_output_coarse:transpose(1,2) -- 128,1,100
                assert (imgW_coarse == cnn_output_coarse:size()[2])
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                -- forward encoder
                local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
                for t = 1, imgW_coarse do
                    if not forward_only then
                        self.encoder_coarse_fw_clones[t]:training()
                    else
                        self.encoder_coarse_fw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    local out = self.encoder_coarse_fw_clones[t]:forward(encoder_input)
                    rnn_state_enc[t] = out
                end
                local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, imgW_coarse+1)
                for t = imgW_coarse, 1, -1 do
                    if not forward_only then
                        self.encoder_coarse_bw_clones[t]:training()
                    else
                        self.encoder_coarse_bw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    local out = self.encoder_coarse_bw_clones[t]:forward(encoder_input)
                    rnn_state_enc_bwd[t] = out
                end
                for t = imgW_coarse, 1, -1 do
                    counter = (i-1)*imgW_coarse + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_coarse_fw_grads[{{},counter}])
                    local dlst = self.encoder_coarse_fw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    cnn_coarse_grad[{i, {}, t, {}}]:copy(dlst[1])
                end
                -- backward directional encoder
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                for t = 1, imgW_coarse do
                    counter = (i-1)*imgW_coarse + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_coarse_bw_grads[{{},counter}])
                    local dlst = self.encoder_coarse_bw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    --cnn_grad[{{}, t, {}}]:add(dlst[1])
                    cnn_coarse_grad[{i, {}, t, {}}]:add(dlst[1])
                end
            end
            -- cnn
            --cnn_coarse_grad:zero()
            local cnn_final_coarse_grad = cnn_coarse_grad:split(1, 1)
            for i = 1, #cnn_final_coarse_grad do
                cnn_final_coarse_grad[i] = cnn_final_coarse_grad[i]:contiguous():view(batch_size, imgW_coarse, -1)
            end

            encoder_fine_grads:add(self.cnn_model:backward(input_batch, cnn_final_coarse_grad))
            encoder_fine_fw_grads:add(encoder_fine_grads[{{}, {}, {1,self.encoder_num_hidden}}])
            encoder_fine_bw_grads:add(encoder_fine_grads[{{}, {}, {self.encoder_num_hidden+1,2*self.encoder_num_hidden}}])

            -- fine
            -- forward directional encoder
            for i = 1, imgH_fine do
                local cnn_output_fine = word_embeddings_list[i]
                local source = cnn_output_fine:transpose(1,2) -- 128,1,100
                assert (imgW_fine == cnn_output_fine:size()[2])
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                -- forward encoder
                local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
                for t = 1, imgW_fine do
                    if not forward_only then
                        self.encoder_fine_fw_clones[t]:training()
                    else
                        self.encoder_fine_fw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    local out = self.encoder_fine_fw_clones[t]:forward(encoder_input)
                    rnn_state_enc[t] = out
                end
                local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, imgW_fine+1)
                for t = imgW_fine, 1, -1 do
                    if not forward_only then
                        self.encoder_fine_bw_clones[t]:training()
                    else
                        self.encoder_fine_bw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    local out = self.encoder_fine_bw_clones[t]:forward(encoder_input)
                    rnn_state_enc_bwd[t] = out
                end
                for t = imgW_fine, 1, -1 do
                    counter = (i-1)*imgW_fine + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_fine_fw_grads[{{},counter}])
                    local dlst = self.encoder_fine_fw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    cnn_fine_grad[{i, {}, t, {}}]:copy(dlst[1])
                end
                -- backward directional encoder
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                for t = 1, imgW_fine do
                    counter = (i-1)*imgW_fine + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_fine_bw_grads[{{},counter}])
                    local dlst = self.encoder_fine_bw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    --cnn_grad[{{}, t, {}}]:add(dlst[1])
                    cnn_fine_grad[{i, {}, t, {}}]:add(dlst[1])
                end
            end
            -- cnn
            --cnn_fine_grad:zero()
            local cnn_final_fine_grad = cnn_fine_grad:split(1, 1)
            for i = 1, #cnn_final_fine_grad do
                cnn_final_fine_grad[i] = cnn_final_fine_grad[i]:contiguous():view(batch_size, imgW_fine, -1)
            end

            self.word_model:backward(input_batch, cnn_final_fine_grad)
            
            
            collectgarbage()
        end
        return loss, self.grad_params, {num_nonzeros, accuracy}
    end
    local optim_state = self.optim_state
    if not forward_only then
        --print ('*******')
        ------optim.checkgrad_list(feval, self.params, 1e-6)
        --local _, loss, stats = optim.sgd_list(feval, self.params, optim_state); loss = loss[1]
        --local loss, orig_grad_params, stats = feval(self.params)
        --local grad_params = {}
        --for i = 1, #orig_grad_params do
        --    if orig_grad_params[i] ~= nil then
        --        grad_params[i] = orig_grad_params[i]:clone()
        --    end
        --end
        --for i2 = 1, #self.params do
        --    local i = #self.params - i2 + 1
        --    if self.params[i] ~= nil then
        --        print ('i')
        --        print (i)
        --        local epsilon = 1e-4
        --        print ('j')
        --        
        --        local _, max_inds = torch.topk(-1*grad_params[i]:view(-1), 5)
        --        local _, min_inds = torch.topk(grad_params[i]:view(-1), 5)
        --        for k = 1, 10 do
        --            local j
        --            if k <= 5 then
        --                j = max_inds[k]
        --            else
        --                j = min_inds[k-5]
        --            end
        --            print (j)
        --            -- minus epsilon
        --            self.params[i][j] = self.params[i][j] - epsilon
        --            local loss1, tmpgrad_params, stats = feval(self.params)
        --            self.params[i][j] = self.params[i][j] + 2*epsilon
        --            local loss2, tmpgrad_params, stats = feval(self.params)
        --            print (string.format('ana: %f, numerical: %f', grad_params[i][j], (loss2-loss1) / 2.0 / epsilon))
        --            self.params[i][j] = self.params[i][j] - epsilon
        --        end
        --    end
        --end
        local _, loss, stats = optim.sgd_list(feval, self.params, optim_state); loss = loss[1]
        return loss*batch_size, stats
    else
        local loss, _, stats = feval(self.params)
        return loss*batch_size, stats -- todo: accuracy
    end
end
-- Set visualize phase
function model:vis(output_dir)
    self.visualize = true
    self.visualize_path = paths.concat(output_dir, 'results.txt')
    self.visualize_attn_path = paths.concat(output_dir, 'results_attn.txt')
    local file, err = io.open(self.visualize_path, "w")
    local file_attn, err_attn = io.open(self.visualize_attn_path, "w")
    self.visualize_file = file
    self.visualize_attn_file = file_attn
    if err then 
        log(string.format('Error: visualize file %s cannot be created', self.visualize_path))
        self.visualize  = false
        self.visualize_file = nil
    elseif err_attn then
        log(string.format('Error: visualize attention file %s cannot be created', self.visualize_attn_path))
        self.visualize  = false
        self.visualize_attn_file = nil
    else
        self.softmax_attn_clones = {}
        for i = 1, #self.decoder_clones do
            local decoder = self.decoder_clones[i]
            local decoder_attn
            decoder:apply(function (layer) 
                if layer.name == 'decoder_attn' then
                    decoder_attn = layer
                end
            end)
            assert (decoder_attn)
            decoder:apply(function (layer)
                if layer.name == 'mul_attn' then
                    self.softmax_attn_clones[i] = layer
                end
            end)
            assert (self.softmax_attn_clones[i])
        end
    end
end
-- Save model to model_path
function model:save(model_path)
    for i = 1, #self.layers do
        self.layers[i]:clearState()
    end
    torch.save(model_path, {{self.cnn_model, self.encoder_fine_fw, self.encoder_fine_bw, self.encoder_coarse_fw, self.encoder_coarse_bw, self.reshaper, self.decoder, self.output_projector}, self.config, self.global_step, self.optim_state, id2vocab, self.reward_baselines, id2vocab_source})
end

function model:shutdown()
    if self.visualize_file then
        self.visualize_file:close()
    end
    if self.visualize_attn_file then
        self.visualize_attn_file:close()
    end
end
