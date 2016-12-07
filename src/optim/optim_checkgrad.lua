--[[ An implementation of a simple numerical gradient checker.
ARGS:
- `opfunc` : a function that takes a single input (X), the point of
         evaluation, and returns f(X) and df/dX
- `x` : the initial point
- `eps` : the epsilon to use for the numerical check (default is 1e-7)
RETURN:
- `diff` : error in the gradient, should be near tol
- `dC` : exact gradient at point 
- `dC_est` : numerically estimates gradient at point
]]--


-- function that numerically checks gradient of NCA loss:
function optim.checkgrad_list(opfunc, y, eps)
   
    print (y)
   for i = 1, #y do
       print ('----------')
       print (i)
       if y[i] ~= nil and i == 7 then
       print ('*******----------')
           local x = y[i]
            -- compute true gradient:
            local Corg,dD = opfunc(y)
            dC = dD[i]
            dC:resize(x:size())
            
            local Ctmp -- temporary value
            local isTensor = torch.isTensor(Corg)
            if isTensor then
                  Ctmp = Corg.new(Corg:size())
            end
            
            -- compute numeric approximations to gradient:
            local eps = eps or 1e-7
            local dC_est = torch.Tensor():typeAs(dC):resizeAs(dC)
            for i = 1,dC:size(1) do
              local tmp = x[i]
              x[i] = x[i] + eps
              local C1 = opfunc(y)
              if isTensor then
                  Ctmp:copy(C1)
                  C1 = Ctmp
              end
              x[i] = x[i] - 2 * eps
              local C2 = opfunc(y)
              x[i] = tmp
              dC_est[i] = (C1 - C2) / (2 * eps)
              print (dC[i])
              print (dC_est[i])
            end
            -- estimate error of gradient:
            local diff = torch.norm(dC - dC_est) / torch.norm(dC + dC_est)
            print (string.format('i: %d, diff: %f', i, diff))
        end
    end
end
