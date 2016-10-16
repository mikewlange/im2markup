require 'nn'

local Reshaper, parent = torch.class("nn.Reshaper", "nn.Module")

function Reshaper:__init(fine)
   parent.__init(self)
   self.fine = fine
   self.imgW_coarse = nil 
   self.imgH_coarse = nil
   self.imgW_fine = nil
   self.imgH_fine = nil
end


function Reshaper:updateOutput(input)
   -- input: batch_size, imgH_fine*imgW_fine, 2*encoder_num_hidden
   local batch_size = input:size(1)
   local num_hidden = input:size(3)
   local imgH_coarse = self.imgH_coarse
   local imgW_coarse = self.imgW_coarse
   local imgH_fine = self.imgH_fine
   local imgW_fine = self.imgW_fine
   local ratio_H = self.fine[1]--math.floor(imgH_fine/imgH_coarse)
   local ratio_W = self.fine[2]--math.floor(imgW_fine/imgW_coarse)
   --assert (ratio_H == self.fine[1], string.format('ratio_H %d vs fine[1] %d', ratio_H, self.fine[1]))
   --assert (ratio_W == self.fine[2], string.format('ratio_W %d vs fine[2] %d', ratio_W, self.fine[2]))
   local image = input:contiguous():view(batch_size, imgH_fine, imgW_fine, -1)
   self.output = self.output or localize(torch.Tensor(batch_size, imgH_coarse*imgW_coarse, self.fine[1]*self.fine[2], num_hidden)):fill(0)
   self.output:resize(batch_size, imgH_coarse*imgW_coarse, self.fine[1]*self.fine[2], num_hidden)
   for i = 1, imgH_coarse do
       for j = 1, imgW_coarse do
           local h_low = (i-1)*ratio_H + 1
           local h_high = i*ratio_H
           local w_low = (j-1)*ratio_W + 1
           local w_high = j*ratio_W
           self.output[{{}, (i-1)*imgW_coarse+j, {}, {}}]:copy(image[{{}, {h_low,h_high}, {w_low,w_high}, {}}]:contiguous():view(batch_size, self.fine[1]*self.fine[2], num_hidden))
       end
   end
   return self.output
end

function Reshaper:updateGradInput(input, gradOutput)
   local batch_size = gradOutput:size(1)
   local num_hidden = gradOutput:size(4)
   local imgH_coarse = self.imgH_coarse
   local imgW_coarse = self.imgW_coarse
   local imgH_fine = self.imgH_fine
   local imgW_fine = self.imgW_fine
   local ratio_H = self.fine[1]--math.floor(imgH_fine/imgH_coarse)
   local ratio_W = self.fine[2]--math.floor(imgW_fine/imgW_coarse)
   --assert (ratio_H == self.fine[1], string.format('ratio_H %d vs fine[1] %d', ratio_H, self.fine[1]))
   self.gradInput = self.gradInput or localize(torch.Tensor(batch_size, imgH_fine*imgW_fine, num_hidden))
   self.gradInput:resizeAs(input):zero()
   local gradImage = self.gradInput:view(batch_size, imgH_fine, imgW_fine, num_hidden)
   for i = 1, imgH_coarse do
       for j = 1, imgW_coarse do
           local h_low = (i-1)*ratio_H + 1
           local h_high = i*ratio_H
           local w_low = (j-1)*ratio_W + 1
           local w_high = j*ratio_W
           gradImage[{{}, {h_low,h_high}, {w_low,w_high}, {}}]:add(gradOutput[{{}, (i-1)*imgW_coarse+j, {}, {}}]:contiguous():view(batch_size, self.fine[1], self.fine[2], num_hidden))
       end
   end
   return self.gradInput
end
