img = imread('cat.jpg');
img = double(imresize(img, [227, 227]));
net = alexnet;

cost = @(nois)costFx(net,img,nois);
init = zeros(227*227*3,1);
options = optimset('MaxIter', 1, 'MaxFunEvals', 1, 'TolX',1e-5);
[noise, ~] =  fminunc(cost,init,options);
