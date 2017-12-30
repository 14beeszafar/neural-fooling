function prob = costFx(net, img, nois)
    nois = reshape(nois,[227,227,3]);
    temp = activations(net, img +nois, 24);
    prob = double(-temp(628) - 0.5);
end