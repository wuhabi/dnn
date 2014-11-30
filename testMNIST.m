clear; clc;

[train_x, train_y, test_x, test_y] = loadMNIST();

hyperParas.debug = 0;

hyperParas.arch = [784, 1000, 10];
hyperParas.numLayer = numel(hyperParas.arch);
hyperParas.outDim = hyperParas.arch(end);
hyperParas.actFunc = 'sigm'; % sigm, tanh, relu
hyperParas.loss = 'crossEnt'; % crossEnt, square

hyperParas.learnRate = 0.1;
hyperParas.batchSize = 100;
hyperParas.numEpochs = 5;

modelParas = nninit(hyperParas);
[modelParas, losses] = nntrain(hyperParas, modelParas, train_x, train_y);
err = nntest(hyperParas, modelParas, test_x, test_y);
figure = figure('color',[1,1,1]); plot(losses(1:10:end));
