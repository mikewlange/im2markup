local Batchnorm, parent = torch.class("nn.BatchNorm", "nn.Module")

function Batchnorm:__init(s)
   parent.__init(self)
   self.running_mean = s.running_mean
   self.running_std = s.running_var:clone():sqrt()
   self.weight = s.weight
   self.bias = s.bias
   self.eps = 1e-5
   assert (self.running_mean ~= nil)
   assert (self.running_std ~= nil)
   assert (self.weight ~= nil)
   assert (self.bias ~= nil)
   self._parameters,self._gradParameters = s:parameters()
end

function Batchnorm:parameters()
    return self._parameters, self._gradParameters
end

--function Batchnorm:accGradParameters(input, gradOutput, scale)
--       return 
--end

function Batchnorm:updateOutput(input)
   local batch_size = input:size(1)
   self.output = self.output or input.new()
   self.output:resizeAs(input)
   self.output:copy(input)
   self.output:add(-1, self.running_mean:view(1, -1, 1, 1):expandAs(input))
   self.output:cdiv(self.running_std:view(1, -1, 1, 1):expandAs(input))
   self.output:cmul(self.weight:view(1, -1, 1, 1):expandAs(input))
   self.output:add(1, self.bias:view(1, -1, 1, 1):expandAs(input))
   return self.output
end

function Batchnorm:updateGradInput(input, gradOutput)
   self.gradInput = input or input.new()
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   self.gradInput:cmul(self.weight:view(1,-1,1,1):expandAs(input))
   self.gradInput:cdiv(self.running_std:view(1,-1,1,1):expandAs(input))
   for i = 1,#self._gradParameters do
       self._gradParameters[i]:zero()
   end
   return self.gradInput
end
