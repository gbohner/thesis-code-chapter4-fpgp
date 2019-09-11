% % Get spikes per time bin for data(i).spikesx
thesis_initpar_decision_spiking;
% Simulate a neuron with given excitatory synaptic input and changing
% inhibition, then count the spikes to figure out the "x" -> "firing rate"
% curve
close all;

T = 3000000;
% With decision input
gI = par.x;
for k=1:length(gI)
  if mod(k,10)==0
    disp(k)
  end
  [V,spikes] = iafsim( par, T, par.gE_deci, gI(k) );
  fr_from_inhib_f(k) = length(spikes)/T*1000;
end
fr_from_inhib_x = gI;
save('fr_from_inhib_deci.mat', 'fr_from_inhib_x', 'fr_from_inhib_f')
figure; plot(fr_from_inhib_x,fr_from_inhib_f)

% With loading input
gI = par.x;
for k=1:length(gI)
  if mod(k,10)==0
    disp(k)
  end
  [V,spikes] = iafsim( par, T, par.gE_load, gI(k) );
  fr_from_inhib_f(k) = length(spikes)/T*1000;
end
fr_from_inhib_x = gI;
save('fr_from_inhib_load.mat', 'fr_from_inhib_x', 'fr_from_inhib_f')
figure; plot(fr_from_inhib_x,fr_from_inhib_f)