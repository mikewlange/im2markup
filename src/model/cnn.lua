require 'cutorch'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'

function createWordModel(source_vocab_size, source_embedding_size)
    local model = nn.Sequential()

    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- input
    -- input shape: (batch_size, 1, imgH, imgW)
    model:add(nn.View(-1):setNumInputDims(3)) --(batch_size, imgH*imgW)
    local word_embedding_layer = nn.LookupTable(source_vocab_size, source_embedding_size)
    word_embedding_layer.name = 'word_vecs_encoder'
    model:add(word_embedding_layer) --(batch_size, imgH*imgW, embedding_size)
    local word_embeddings = model(inputs[1])
    local raw_features = nn.ReshapeAs()({word_embeddings, inputs[1]}) --(batch_size, imgH, imgW, embedding_size)
    local model_convert = nn.Sequential()
    model_convert:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, embedding_size)
    local fine_features = model_convert(raw_features)
    return nn.gModule(inputs, {fine_features})
end

function createCNNModel(source_embedding_size)
    local model = nn.Sequential()

    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- input
    local raw_features = inputs[1] --(batch_size, imgH, imgW, embedding_size)


    -- coarse model
    local model_coarse = nn.Sequential()
    model_coarse:add(nn.Transpose({3,4}, {2,3})) -- (batch_size, embedding_size, H, W)
    model_coarse:add(cudnn.SpatialConvolution(source_embedding_size, 100, 3, 1, 1, 1, 1, 0):setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')) -- (batch_size, source_embedding_size, imgH, imgW)
    --model_coarse:add(nn.SpatialBatchNormalization(100))
    model_coarse:add(cudnn.ReLU(true))
    model_coarse:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0)) -- (batch_size, source_embedding_size, imgH, imgW/2)

    model_coarse:add(cudnn.SpatialConvolution(100, 100, 3, 1, 1, 1, 1, 0):setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')) -- (batch_size, source_embedding_size, imgH, imgW)) -- (batch_size, source_embedding_size, imgH, imgW)
    --model_coarse:add(nn.SpatialBatchNormalization(100))
    model_coarse:add(cudnn.ReLU(true))
    model_coarse:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0)) -- (batch_size, source_embedding_size, imgH, imgW/4)

    model_coarse:add(cudnn.SpatialConvolution(100, 100, 3, 1, 1, 1, 1, 0):setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')) -- (batch_size, source_embedding_size, imgH, imgW)) -- (batch_size, source_embedding_size, imgH, imgW)
    --model_coarse:add(nn.SpatialBatchNormalization(100))
    model_coarse:add(cudnn.ReLU(true))
    model_coarse:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0)) -- (batch_size, source_embedding_size, imgH, imgW/8)

    model_coarse:add(cudnn.SpatialConvolution(100, 100, 3, 1, 1, 1, 1, 0):setMode('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')) -- (batch_size, source_embedding_size, imgH, imgW)) -- (batch_size, source_embedding_size, imgH, imgW)
    --model_coarse:add(nn.SpatialBatchNormalization(100))
    model_coarse:add(cudnn.ReLU(true))
    model_coarse:add(cudnn.SpatialMaxPooling(4, 1, 4, 1, 0, 0)) -- (batch_size, source_embedding_size, imgH, imgW/32)

    model_coarse:add(nn.Transpose({2,3}, {3,4})) -- (batch_size, H, W, source_embedding_size)
    model_coarse:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, source_embedding_size)

    local coarse_features = model_coarse(raw_features)
    local model = nn.gModule(inputs, {coarse_features})
    return model
end
