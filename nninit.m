function [ modelParas ] = nninit( hyperParas )
%NNINIT 
%   

modelParas.weights = cell(hyperParas.numLayer-1, 1);
modelParas.bias = cell(hyperParas.numLayer-1, 1);
for i = 1:hyperParas.numLayer-1
    modelParas.weights{i} = 0.1*randn(hyperParas.arch(i), hyperParas.arch(i+1));
    modelParas.bias{i} = zeros(hyperParas.arch(i+1), 1);
%     initWeights = rand(hyperParas.arch(i)+1, hyperParas.arch(i+1))-0.5;
%     modelParas.weights{i} = initWeights(1:hyperParas.arch(i), :);
%     modelParas.bias{i} = 0*initWeights(hyperParas.arch(i)+1, :)';
end

end

