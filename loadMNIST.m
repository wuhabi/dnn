function [train_x, train_y, test_x, test_y] = loadMNIST()
%LOADMNIST 
%   

load mnist_uint8;
train_x = double(train_x')/255; train_y = double(train_y');
test_x = double(test_x')/255; test_y = double(test_y');

end

