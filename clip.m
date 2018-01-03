function clipped = clip(noise, epsilon)
    clipped = noise;
    clipped(abs(clipped)>epsilon) = epsilon;
end