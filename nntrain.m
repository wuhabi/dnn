function [ modelParas,losses ] = nntrain( hyperParas, modelParas, train_x, train_y )
%NNTRAIN 
%   

trainSize = size(train_x, 2);
numBatches = ceil(trainSize/hyperParas.batchSize);

losses = zeros(hyperParas.numEpochs*numBatches, 1);
for epoch = 1:hyperParas.numEpochs
    tic;
    randinds = randperm(trainSize);
    for i = 1:numBatches
        batchSize = hyperParas.batchSize;
        if i==numBatches && mod(trainSize, hyperParas.batchSize)>0 
            batchSize = mod(trainSize, hyperParas.batchSize)>0;
        end
        batch_x = train_x(:, randinds((i-1)*batchSize+1:i*batchSize));
        batch_y = train_y(:, randinds((i-1)*batchSize+1:i*batchSize));
        
        [ netState, loss ] = nnfp( hyperParas, modelParas, batch_x, batch_y );
        losses((epoch-1)*numBatches+i) = loss;
        fprintf('Epoch %d/%d, batch %d, loss %f\n',epoch, hyperParas.numEpochs, i, loss);
        grads = nnbp( hyperParas, modelParas, netState, batch_x, batch_y );
        modelParas = nngraddesc(hyperParas, modelParas, grads);
        
    end
    toc;
end

end

