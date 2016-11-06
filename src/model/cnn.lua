require 'cutorch'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'cunn'

function createCNNModel(use_cuda)
    local model = nn.Sequential()

    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- input
    -- input shape: (batch_size, 1, imgH, imgW)
    -- CNN part
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))

    model:add(cudnn.SpatialConvolution(1, 64, 3, 3, 1, 1, 1, 1)) -- (batch_size, 64, imgH, imgW)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (batch_size, 64, imgH/2, imgW/2)

    model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- (batch_size, 128, imgH/2, imgW/2)
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) -- (batch_size, 128, imgH/2/2, imgW/2/2)

    model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- (batch_size, 256, imgH/2/2, imgW/2/2)
    model:add(nn.SpatialBatchNormalization(256))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)) -- (batch_size, 256, imgH/2/2, imgW/2/2)
    model:add(cudnn.ReLU(true))
    
    model:add(cudnn.SpatialMaxPooling(1, 2, 1, 2, 0, 0)) -- (batch_size, 256, imgH/2/2/2, imgW/2/2)

    model:add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    model:add(cudnn.SpatialMaxPooling(2, 1, 2, 1, 0, 0)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model:add(nn.SpatialBatchNormalization(512))
    model:add(cudnn.ReLU(true))

    local raw_features = model(inputs[1])
    -- (batch_size, 512, H, W)    

    local model_convert = nn.Sequential()
    model_convert:add(nn.Transpose({2, 3}, {3,4})) -- (batch_size, H, W, 512)
    model_convert:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512)
    --model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    --model:cuda()
    -- coarse model
    local fine_features = model_convert(raw_features)
    local model_coarse = nn.Sequential()
    model_coarse:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model_coarse:add(nn.SpatialBatchNormalization(512))
    model_coarse:add(cudnn.ReLU(true))

    model_coarse:add(cudnn.SpatialMaxPooling(4, 4, 4, 4, 0, 0)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model_coarse:add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)) -- (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
    model_coarse:add(nn.SpatialBatchNormalization(512))
    model_coarse:add(cudnn.ReLU(true))

    model_coarse:add(nn.Transpose({2, 3}, {3,4})) -- (batch_size, H, W, 512)
    model_coarse:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512)

    local coarse_features = model_coarse(raw_features)
    local model = nn.gModule(inputs, {fine_features,coarse_features})
    return model

end
