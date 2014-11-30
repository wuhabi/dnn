function [ err ] = nntest( hyperParas, modelparas, test_x, test_y )
%NNTEST 
%   

[netState, ~] = nnfp(hyperParas, modelparas, test_x, test_y);

probs = netState.activity{hyperParas.numLayer};

[~, predict] = max(probs);
[~, target] = max(test_y);

bad = find(predict ~= target);
err = numel(bad) / size(test_x, 2);

end