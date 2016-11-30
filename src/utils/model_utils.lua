 --[[ Adapted from https://github.com/karpathy/char-rnn/blob/master/model/model_utils.lua
--]]
function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

------------------------------------------------------------------------
--[[ LinearNoBias ]]--
-- Subclass of nn.Linear with no bias term
------------------------------------------------------------------------
nn = require 'nn'
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   nn.Module.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function LinearNoBias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function LinearNoBias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:mv(self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end
local ViewAs = torch.class('nn.ViewAs', 'nn.Module')
-- Views input[1] based on first ndim sizes of input[2]

function ViewAs:__init(ndim)
  nn.Module.__init(self)
  self.ndim = ndim
  self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
end

function ViewAs:updateOutput(input)
  self.output = self.output or input.new()

  assert(#input == 2, 'ViewAs can only take 2 inputs')
  if self.ndim then
    local sizes = {}
    for i = 1, self.ndim do
      sizes[#sizes+1] = input[2]:size(i)
    end
    self.output:view(input[1], table.unpack(sizes))
  else
    local sizes = input[2]:size()
    self.output:view(input[1], sizes)
  end
  return self.output
end

function ViewAs:updateGradInput(input, gradOutput)
  if self.gradInput == nil or self.gradInput[1] == nil then
    self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
  end
  self.gradInput[2]:resizeAs(input[2]):zero() -- unused

  self.gradInput[1] = self.gradInput[1] or gradOutput.new()
  self.gradInput[1]:view(gradOutput, input[1]:size())
  return self.gradInput
end


local ReshapeAs = torch.class('nn.ReshapeAs', 'nn.Module')
--(batch_size, imgH*imgW, embedding_size)
--(batch_size, 1, imgH, imgW)
-- output:
--(batch_size, imgH, imgW, embedding_size)

function ReshapeAs:__init()
  nn.Module.__init(self)
  self.ndim = ndim
  self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
end

function ReshapeAs:updateOutput(input)
  self.output = self.output or input.new()

  assert(#input == 2, 'ReshapeAs can only take 2 inputs')
    local sizes = {}
    sizes[1] = input[2]:size(1)
    sizes[2] = input[2]:size(3)
    sizes[3] = input[2]:size(4)
    sizes[4] = input[1]:size(3)
    self.output:view(input[1], table.unpack(sizes))
  return self.output
end

function ReshapeAs:updateGradInput(input, gradOutput)
  if self.gradInput == nil or self.gradInput[1] == nil then
    self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
  end
  self.gradInput[2]:resizeAs(input[2]):zero() -- unused

  self.gradInput[1] = self.gradInput[1] or gradOutput.new()
  self.gradInput[1]:view(gradOutput, input[1]:size())
  return self.gradInput
end

local ReplicateAs = torch.class('nn.ReplicateAs', 'nn.Module')
-- Replicates dim m of input[1] based on dim n of input[2]
-- basically copies Replicate

function ReplicateAs:__init(in_dim, template_dim)
  nn.Module.__init(self)
  self.in_dim = in_dim
  self.template_dim = template_dim
  self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
end

function ReplicateAs:updateOutput(input)
  assert(#input == 2, 'needs 2 inputs')
  local rdim = self.in_dim
  local ntimes = input[2]:size(self.template_dim)
  input = input[1]
  local sz = torch.LongStorage(input:dim() + 1)
  sz[rdim] = ntimes
  for i = 1,input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    sz[i+offset] = input:size(i)
  end
  local st = torch.LongStorage(input:dim() + 1)
  st[rdim] = 0
  for i = 1,input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    st[i+offset] = input:stride(i)
  end
  self.output:set(input:storage(), input:storageOffset(), sz, st)
  return self.output
end

function ReplicateAs:updateGradInput(input, gradOutput)
  if self.gradInput == nil or self.gradInput[1] == nil then
    self.gradInput = {localize(torch.Tensor()), localize(torch.Tensor())}
  end
  self.gradInput[2]:resizeAs(input[2]):zero() -- unused

  input = input[1]
  self.gradInput[1]:resizeAs(input):zero()
  local rdim = self.in_dim
  local sz = torch.LongStorage(input:dim() + 1)
  sz[rdim] = 1
  for i = 1, input:dim() do
    local offset = 0
    if i >= rdim then offset = 1 end
    sz[i+offset] = input:size(i)
  end
  local gradInput = self.gradInput[1]:view(sz)
  gradInput:sum(gradOutput, rdim)

  return self.gradInput
end
