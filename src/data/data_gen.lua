 --[[ Load image data. Adapted from https://github.com/da03/Attention-OCR/blob/master/src/data_util/data_gen.py. 
 --    ARGS:
 --        - `data_base_dir`      : string, The base directory of the image path in data_path. If the image path in data_path is absolute path, set it to /.
 --        - `data_path`  : string, The path containing data file names and labels. Format per line: image_path characters. Note that the image_path is the relative path to data_base_dir
 --]]
require 'image'
require 'paths'
require 'utils'
require 'class'
tds = require('tds')

local DataGen = torch.class('DataGen')

function DataGen:__init(data_base_dir, data_path, label_path, max_encoder_l_h, max_encoder_l_w, max_decoder_l)
    --self.imgW = 32
    self.data_base_dir = data_base_dir
    self.data_path = data_path
    self.label_path = label_path
    self.max_width = max_width
    self.max_encoder_l_h = max_encoder_l_h or math.huge
    self.max_encoder_l_w = max_encoder_l_w or math.huge
    self.max_decoder_l = max_decoder_l or math.huge
    self.min_aspect_ratio = 0.5

    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
    local file, err = io.open(self.data_path, "r")
    if err then 
        file, err = io.open(paths.concat(self.data_base_dir, self.data_path), "r")
        if err then
            log(string.format('Error: Data file %s not found ', self.data_path))
            os.exit()
            --return
        end
    end
    self.lines = tds.Hash()
    local idx = 0
    for line in file:lines() do
        idx = idx + 1
        if idx % 1000000==0 then
            log (string.format('%d lines read', idx))
        end
        local filename, label = unpack(split(line))
        self.lines[idx] = tds.Vec({filename, label})
    end
    collectgarbage()
    self.cursor = 1
    self.buffer = {}
end

function DataGen:shuffle()
    shuffle(self.lines)
end

function DataGen:size()
    return #self.lines
end

vocab2id_source = nil
function DataGen:nextBatch(batch_size)
    if vocab2id_source == nil then
        vocab2id_source = tds.Hash()
        for i = 1, #id2vocab_source do
            vocab2id_source[id2vocab_source[i]] = i+4
        end
    end
    while true do
        if self.cursor > #self.lines then
            break
        end
        local doc_path = self.lines[self.cursor][1]
        local file, err = io.open(paths.concat(self.data_base_dir, doc_path), "r")
        if err then
            self.cursor = self.cursor + 1
            log(doc_path)
        else
            local doc = tds.Hash()
            local sentence_idx = 1
            for line in file:readlines() do
                local strlist = split(trim(line))
                local numlist = tds.Hash()
                numlist[1] = 2
                for i = 1, #strlist do
                    local token = strlist[i]
                    if vocab2id_source[token] ~= nil then
                        numlist[#numlist+1] = vocab2id_source[token]
                    elseif token == '__PAD__' then
                        numlist[#numlist+1] = 1 -- PAD
                    else
                        numlist[#numlist+1] = 4 -- UNK
                    end
                end
                doc[sentence_idx] = numlist
            end
            local label_str = self.lines[self.cursor][2]
            local label_list = path2numlist(label_str, self.label_path)
            self.cursor = self.cursor + 1
            local origH = #doc
            local origW = #doc[1]
            if #label_list-1 > self.max_decoder_l then
                log(string.format('WARNING: %s\'s target sequence is too long, will be truncated. Consider using a larger max_num_tokens'%img_path))
                local temp = {}
                for i = 1, self.max_decoder_l+1 do
                    temp[i] = label_list[i]
                end
                label_list = temp
            end
            if #label_list-1 <= self.max_decoder_l and math.floor(origH/8.0) <= self.max_encoder_l_h and math.floor(origW/8.0) <= self.max_encoder_l_w then
                local imgW = origW
                local imgH = origH
                if self.buffer[imgW] == nil then
                    self.buffer[imgW] = {}
                end
                if self.buffer[imgW][imgH] == nil then
                    self.buffer[imgW][imgH] = {}
                end
                table.insert(self.buffer[imgW][imgH], {doc, label_list, doc_path})
                if #self.buffer[imgW][imgH] == batch_size then
                    local images = torch.Tensor(batch_size, 1, imgH, imgW)
                    local max_target_length = -math.huge
                    -- visualize
                    local img_paths = {}
                    for i = 1, #self.buffer[imgW][imgH] do
                        img_paths[i] = self.buffer[imgW][imgH][i][3]
                        for j = 1, imgH do
                            for k = 1, imgW do
                                images[i][j][k] = self.buffer[imgW][imgH][i][1][j][k]
                            end
                        end
                        max_target_length = math.max(max_target_length, #self.buffer[imgW][imgH][i][2])
                    end
                    -- targets: use as input. SOS, ch1, ch2, ..., chn
                    local targets = torch.IntTensor(batch_size, max_target_length-1):fill(1)
                    -- targets_eval: use for evaluation. ch1, ch2, ..., chn, EOS
                    local targets_eval = torch.IntTensor(batch_size, max_target_length-1):fill(1)
                    local num_nonzeros = 0
                    for i = 1, #self.buffer[imgW][imgH] do
                        num_nonzeros = num_nonzeros + #self.buffer[imgW][imgH][i][2] - 1
                        for j = 1, #self.buffer[imgW][imgH][i][2]-1 do
                            targets[i][j] = self.buffer[imgW][imgH][i][2][j] 
                            targets_eval[i][j] = self.buffer[imgW][imgH][i][2][j+1] 
                        end
                    end
                    self.buffer[imgW][imgH] = nil
                    --collectgarbage()
                    do return {images, targets, targets_eval, num_nonzeros, img_paths} end
                end
            else
                log(string.format('WARNING: %s is too large, will be ignored. Consider using a larger max_image_width or max_image_height'%img_path))
            end
        end
    end

    if next(self.buffer) == nil then
        self.cursor = 1
        collectgarbage()
        return nil
    end
    local imgW, v = next(self.buffer)
    while next(self.buffer[imgW]) == nil do -- if buffer[imgW] == {}
        if next(self.buffer, imgW) == nil then
            self.cursor = 1
            collectgarbage()
            return nil
        end
        imgW, v = next(self.buffer, imgW)
    end
    local imgH, v = next(self.buffer[imgW], nil) 
    real_batch_size = #self.buffer[imgW][imgH]
    local images = torch.Tensor(real_batch_size, 1, imgH, imgW)
    local max_target_length = -math.huge
    -- visualize
    local img_paths = {}
    for i = 1, #self.buffer[imgW][imgH] do
        img_paths[i] = self.buffer[imgW][imgH][i][3]
        for j = 1, imgH do
            for k = 1, imgW do
                images[i][j][k] = self.buffer[imgW][imgH][i][1][j][k]
            end
        end
        max_target_length = math.max(max_target_length, #self.buffer[imgW][imgH][i][2])
    end
    local targets = torch.IntTensor(real_batch_size, max_target_length-1):fill(1)
    local targets_eval = torch.IntTensor(real_batch_size, max_target_length-1):fill(1)
    local num_nonzeros = 0
    for i = 1, #self.buffer[imgW][imgH] do
        num_nonzeros = num_nonzeros + #self.buffer[imgW][imgH][i][2] - 1
        for j = 1, #self.buffer[imgW][imgH][i][2]-1 do
            targets[i][j] = self.buffer[imgW][imgH][i][2][j] 
            targets_eval[i][j] = self.buffer[imgW][imgH][i][2][j+1] 
        end
    end
    self.buffer[imgW][imgH] = nil
    --collectgarbage()
    return {images, targets, targets_eval, num_nonzeros, img_paths}
end
