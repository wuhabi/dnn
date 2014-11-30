function [ netState, loss] = nnfp( hyperParas, modelParas, batch_x, batch_y )
%NNFP 
% 
batchSize = size(batch_x,2);

netState.activity = cell(hyperParas.numLayer, 1);
netState.isPos = cell(hyperParas.numLayer, 1);

netState.activity{1} = batch_x;
for i = 2:hyperParas.numLayer-1
    inputs = modelParas.weights{i-1}'*netState.activity{i-1} + repmat(modelParas.bias{i-1}, 1, batchSize);
    if strcmp(hyperParas.actFunc, 'sigm')
        netState.activity{i} = 1./(1+exp(-inputs));
    elseif strcmp(hyperParas.actFunc, 'tanh')
        netState.activity{i} = (exp(inputs)-exp(-inputs))./(exp(inputs)+exp(-inputs));
    elseif strcmp(hyperParas.actFunc, 'relu')
        pos = double(inputs>0);
        netState.activity{i} = inputs.*pos;
        netState.isPos{i} = pos;
    else
        error('Check hyperParas.actFunc, only sigm, tanh or relu is valid.');
    end
end

inputs2out = modelParas.weights{hyperParas.numLayer-1}'*netState.activity{hyperParas.numLayer-1} + repmat(modelParas.bias{hyperParas.numLayer-1}, 1, batchSize);
inputs2out = inputs2out - repmat(max(inputs2out), hyperParas.outDim, 1);
netState.activity{hyperParas.numLayer} = exp(inputs2out)./repmat(sum(exp(inputs2out)), hyperParas.outDim, 1);

if strcmp(hyperParas.loss, 'crossEnt')
    tiny = exp(-30);
    loss = -sum(sum(batch_y.*log(netState.activity{hyperParas.numLayer}+tiny)))/batchSize;
elseif strcmp(hyperParas.loss, 'square')
    delta = 0.5*(batch_y-netState.activity{hyperParas.numLayer}).^2;
    loss = sum(sum(delta))/batchSize;
else
    error('Check hyperParas.loss, only crossEnt or square is valid.');
end

end

