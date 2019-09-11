% Create experiment data for ICML 2018 paper

addpath(genpath('twonode'))

cur_sim_time = datestr(now,30);


% Create decision type trials
Ntrials = 20;
 
disp 'Decision type trials'
% Store all the trajectories in the format we expect them for our python
% code:
y_python = [];
initpar_decision;
figure; hold on;
for l=1:Ntrials
  %disp(l)
  [xx, yy, tt, nullcline1, nullcline2] = mfsim_special(par, 4, 4, 0, 0); 
  y_python(1,:,l) = xx;
  y_python(2,:,l) = yy;
  scatter(xx, yy, 5, 1:numel(xx)); colormap(jet(numel(xx))); colorbar
end

% 
% tic
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create loading type trials
% Ntrials = 200;
% 
% % Store all the trajectories in the format we expect them for our python
% % code:
% y_python = [];
% initpar_loading;
% for l=1:Ntrials
%   disp(l)
%   toc
%   [xx, yy, tt, nullcline1, nullcline2] = mfsim_special(par, 4, 4, 0, 0); 
%   y_python(1,:,l) = xx;
%   y_python(2,:,l) = yy;
% end
% 
% save(['MachensBrodySim_' cur_sim_time '_loading.mat'], 'y_python', 'nullcline1', 'nullcline2', 'par', 'tt')
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create decision type trials
% Ntrials = 200;
%  
% disp 'Decision type trials'
% % Store all the trajectories in the format we expect them for our python
% % code:
% y_python = [];
% initpar_decision;
% for l=1:Ntrials
%   disp(l)
%   toc
%   [xx, yy, tt, nullcline1, nullcline2] = mfsim_special(par, 4, 4, 0, 0); 
%   y_python(1,:,l) = xx;
%   y_python(2,:,l) = yy;
% end
% 
% save(['MachensBrodySim_' cur_sim_time '_decision.mat'], 'y_python', 'nullcline1', 'nullcline2', 'par', 'tt')
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create mfsim dynamics trials with varying excitatory input
% Ntrials = 200;
%  
% disp 'Varying E input trials'
% % Store all the trajectories in the format we expect them for our python
% % code:
% initpar_decision;
% par.TSF2=4000;
% gE_deci_orig = par.gE_deci;
% 
% for E_extra = [0., (25:5:75)*1e-5, 100e-5]
%   disp(E_extra)
%   y_python = [];
%   par.gE_deci = gE_deci_orig + E_extra;
% %   figure; hold on; title(E_extra)
%   for l=1:Ntrials
%     [xx, yy, tt, nullcline1, nullcline2] = mfsim_special(par, 4, 4, 0, 0); 
%     y_python(1,:,l) = xx;
%     y_python(2,:,l) = yy;
%   end
% %   plot(nullcline1(1,:), nullcline1(2,:))
% %   plot(nullcline2(1,:), nullcline2(2,:))
% %   drawnow;
% %   pause(0.1)
%   
%   save(['MachensBrodySim_' cur_sim_time '_mfsim_noise_001_exc_', sprintf('%0.3d', E_extra*1e5), '.mat'], ...
%   'y_python', 'nullcline1', 'nullcline2', 'par', 'tt')
% end
% 
% 
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Create tanh dynamics trials with varying baseline
% % Ntrials = 200;
% %  
% % disp 'Decision type trials'
% % % Store all the trajectories in the format we expect them for our python
% % % code:
% % initpar_decision;
% % 
% % for bl = 0.0:0.05:0.8
% %   disp(bl)
% %   y_python = [];
% %   par.iobaseline = bl;
% %   for l=1:Ntrials
% %     [xx, yy, tt, nullcline1, nullcline2] = tanhsim_special(par, 4, 4, 0, 0); 
% %     y_python(1,:,l) = xx;
% %     y_python(2,:,l) = yy;
% %   end
% % 
% %   save(['MachensBrodySim_' cur_sim_time '_tanh_baseline_', sprintf('%0.2d', par.iobaseline*100), '.mat'], ...
% %   'y_python', 'nullcline1', 'nullcline2', 'par', 'tt')
% % end
% 
% 
