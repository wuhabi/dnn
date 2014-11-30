function [ modelParas ] = nngraddesc( hyperParas, modelParas, grads )
%NNGRADDESC 
%   

for i = 1:hyperParas.numLayer-1
    weight_inc = -hyperParas.learnRate*grads.weightsGrad{i};
    modelParas.weights{i} = modelParas.weights{i} + weight_inc;
    
    bias_inc = -hyperParas.learnRate*grads.biasGrad{i};
    modelParas.bias{i} = modelParas.bias{i} + bias_inc;
    
end

end

