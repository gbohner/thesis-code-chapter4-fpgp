%initpar - initialize parameters
%
%   This files creates the structure 'par' which contains
%   all the necessary parameters (integrate-and-fire neurons,
%   inhibitory synapses, external inputs, task timing etc.)
%
%   Straightforward changes to this file include
%   the number of neurons in the spiking simulation
%   ('Nneurons'), the noise in the mean-field simulation
%   ('mfnoise'), and the timing of the three
%   task phases ('T', 'TSO1', 'TSF1', 'TSO2', 'TSF2').
%   Changing the inhibitory synaptic weight ('wI')
%   allows to investigate the breakdown of the maintenance
%   mode.
%
%   Less obvious changes include changes to the parameters
%   of the integrate-and-fire neuron and the synapse.
%   Any changes here will usually require to retune
%   the synaptic weight ( see autotuner ).

%--------------noise in simulations
par.Nneurons = 30;    %Number of neurons/node in spiking simulation
par.mfnoise  =   sqrt(0.25);    %Variance of additive noise in mean-field simulation
par.noiseTimeStep = 50; % (ms) Add the additive noise directly to x and y every this many timesteps
par.noise_eps = [0.01, 0.01]; %Variance of additive noise in mean-field simulation
par.init_mu = [0.0, 0.0]; %Initial state in mean-field simulation
par.init_Sigma = [0.01, 0.01]*sqrt(par.Nneurons); %Variance of initial state in mean-field simulation

par.iobaseline = 0.0; % Ratio of max firing rate added as baseline firing

%--------------stimulus onset/offset times
par.T       = 2000;  % task length
par.TSO1    =  0;  % onset of f1 (msec)
par.TSF1    = 2000;  % offset of f1 (msec)
par.TSO2    = 2000;  % onset of f2 (msec)
par.TSF2    = 2000;  % offset of f2 (msec)

%--------------mean field i/o curve
load 'fcurve.mat';
par.x = x;           % grid of conductance inputs
par.fx = fx;         % synaptic outputs
par.lambda = lambda; % exc/inh scaling factor
clear x; clear fx; clear lambda;

%--------------integrate-and-fire neuron
par.C       =   0.2 ;  %capacitance (nF)
par.Vthresh = -55   ;  %treshold (mV)
par.Vreset  = -61   ;  %reset potential (mV)
par.refrac  =   2   ;  %refractory period (msec)
par.EL      = -60   ;  %cell's resting potential (mV)
par.EE      = -5    ;  %reversal potential of excitatory input (mV)
par.EI      = -75   ;  %reversal potential of inhibitory input (mV)
par.gL      =  0.01 ;  %leak conductance (nS)
par.gaussnoise  = 0.6; %gaussian nose (mV)
			 
%--------------inhibitory synapses
par.Isatmax =   7   ; %synaptic saturation     
par.Itau    =  80;  ; %synaptic time constant (msec)
par.wI      = 0.0011575; %inhibitory synaptic weight,
                      %this is the fine-tuned parameter
%par.wI = par.wI*1.07; %overstrong GABA

%--------------constant excitatory inputs (uS)
par.gE        = 0.002;            %during maintenance
par.gE_load   = par.gE + 0.0003;  %during loading
par.gE_deci   = par.gE  -0.0005;  %during decision

%--------------stimulus dependent excitatory inputs (uS)
par.gE_loadpm = (-3:3) * 0.000035; %during loading
par.gE_decipm = (-3:3) * 0.000035; %during decision

%--------------sanity checks
if ( par.TSO1>par.TSF1 | par.TSF1>par.TSO2 | ...
     par.TSO2>par.TSF2 | par.TSF2>par.T )
  error('error in task times');
end

%--------------alternate (faster) loading
%note: the dynamics of the mean-field model are a bit
%slower than those of the spiking model during the
%loading mode. For f1=1 or f1=7 the loading process
%is therefore not really finished at the onset
%of the delay period. To speed up the loading process,
%the following parameters can also be used:

%par.gE_load   = par.gE + 0.0009;  %during loading
%par.gE_loadpm = (-3:3) * 0.00007; %during loading

%   (c) 2004 CK Machens & CD Brody
