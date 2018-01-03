% Config
img = imread('cat.jpg');
img = double(imresize(img, [227, 227]));
net = alexnet;
delta = 1e-10;
epsilon = 1/255;
% Base Fx
at = @(ary, idx) ary(idx);
% Real Fx
shape_noise = @(noise) 255 * clip(reshape(noise, [227, 227, 3]), epsilon);
eval_noise = @(img, noise) activations(net, img + shape_noise(noise), 24);
hypothesis = @(noise) at(double(eval_noise(img, noise)), 628);
%hypo_grad = @(noise) (hypothesis(noise+delta)-hypothesis(noise))/delta;
% Optimization
cost = @(noise) -hypothesis(noise);
inital = rand(227*227*3,1);
final = fminunc(cost, inital);