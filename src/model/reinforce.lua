-- adapted from https://github.com/jeffreyling/seq2seq-hard/blob/summary/hard_attn.lua
-- From dpnn library
--
require 'nn'

function nn.Module:reinforce(reward)
  if self.modules then
    for i, module in ipairs(self.modules) do
      module:reinforce(reward)
    end
  end
end

--
------------------------------------------------------------------------
--[[ Reinforce ]]--
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Abstract class for modules that use the REINFORCE algorithm (ref A).
-- The reinforce(reward) method is called by a special Reward Criterion.
-- After which, when backward is called, the reward will be used to 
-- generate gradInputs. The gradOutput is usually ignored.
------------------------------------------------------------------------
local Reinforce, parent = torch.class("nn.Reinforce", "nn.Module")

function Reinforce:__init()
   parent.__init(self)
end

-- a Reward Criterion will call this
function Reinforce:reinforce(reward)
   self.reward = reward
end

function Reinforce:updateOutput(input)
   self.output:set(input)
end

function Reinforce:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end


------------------------------------------------------------------------
--[[ ReinforceCategorical ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are a vector of categorical prob : (p[1], p[2], ..., p[k]) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.230-236) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Reinforce")

function ReinforceCategorical:__init(entropy_scale, semi_sampling_p)
  parent.__init(self)
  self.semi_sampling_p = semi_sampling_p or 0
  self.entropy_scale = entropy_scale or 0
end

function ReinforceCategorical:_doArgmax(input)
   self.output:zero()
   _, self._index = input:max(2)
   self.output:scatter(2, self._index, 1)

   if type(self._index) ~= 'torch.CudaTensor' then
     self._index = self._index:cuda()
   end
end

function ReinforceCategorical:_doSample(input)
   self._do_through = (torch.uniform() < self.semi_sampling_p)
   if self._do_through == true then
      -- use p
      self.output:copy(input)
   else
      -- sample from categorical with p = input
      self._input = self._input or input.new()
      -- prevent division by zero error (see updateGradInput)
      self._input:resizeAs(input):copy(input):add(0.00000001) 

      input.multinomial(self._index, input, 1)
      -- one hot encoding
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   end
end

function ReinforceCategorical:updateOutput(input)
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())
   if self.train then
      --sample
      self:_doSample(input)
   else
     assert(self.train == false)
     -- do argmax at test time
     self:_doArgmax(input)
   end
   return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k]) 
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput:resizeAs(input):zero()
   if self._do_through == true then
     -- identity function
     self.gradInput:copy(gradOutput)
   else 
     self.gradInput:copy(self.output)
     self._input = self._input or input.new()
     -- prevent division by zero error
     self._input:resizeAs(input):copy(input):add(0.00000001) 
     self.gradInput:cdiv(self._input)
     
     -- multiply by reward 
     -- multiply by -1 ( gradient descent on input )

     local batch_size = self.reward:size(1)
     if self.gradInput:size(1) == batch_size then
        self.gradInput:cmul(-1*self.reward:view(batch_size,1):expandAs(self.gradInput))
     else
        local gradinput = self.gradInput:view(batch_size, -1, self.gradInput:size(2))
        gradinput:cmul(-1*self.reward:view(batch_size,1,1):expandAs(gradinput))
     end


     -- add entropy term
     --local grad_ent = self._input:log():add(1) / batch_size?
     --self.gradInput:add(self.entropy_scale, grad_ent)
   end
   return self.gradInput
end

function ReinforceCategorical:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end
