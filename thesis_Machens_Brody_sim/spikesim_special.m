%spikesim - simulate the two-node spiking network
%
%   [gsx, gsy, t] = spikesim( f1, f2 ) simulates
%   two-interval discrimination task in the spiking
%   two-node network with first stimulus 'f1' and
%   second stimulus 'f2'. By default, 'f1' and 'f2' have
%   to be integers within the interval [1,7]. spikesim
%   uses the parameters provided in the function
%   'initpar.m'.
%   
%   spikesim returns the average inhibitory input
%   conductances of the plus ('gsx') and minus
%   ('gsy') neurons at the time points 't'.
%   [gsx, gsy, t, spikesx, spikesy] = spikesim( f1, f2 )
%   also returns two matrices of spike times for
%   the plus ('spikesx') and minus ('spikesy') neurons.
%   Each row of these matrices contains the spike
%   times of a single neuron. By default, rows are
%   filled with zeros after the last spike time.

% (c) 2004 CK Machens & CD Brody

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [gsx, gsy, rt, spikesx, spikesy] = spikesim_special(par, f1, f2 )

%--------------check frequencies
if (f1<1 | f1>length(par.gE_loadpm))
  error('f1 out of range');
end
if (f2<1 | f2>length(par.gE_decipm))
  error('f2 out of range');
end

%--------------simulation parameters
Nneurons = 2*par.Nneurons;       % Total number of neurons
dt = 0.1;                        % simulation time step
t = 0:dt:par.T;
Nt = length(t);

%--------------external excitatory inputs

%spontaneous activity
gEx = 0.0008 * ones( Nt, 1 ); 
gEy = 0.0008 * ones( Nt, 1 );

%during loading
gEx( t>par.TSO1 & t<=par.TSF1 ) = par.gE_load + par.gE_loadpm( f1 );
gEy( t>par.TSO1 & t<=par.TSF1 ) = par.gE_load - par.gE_loadpm( f1 );
gEx( t>par.TSF1 & t<=par.TSO2 ) = par.gE;
gEy( t>par.TSF1 & t<=par.TSO2 ) = par.gE;
gEx( t>par.TSO2 & t<=par.TSF2 ) = par.gE_deci - par.gE_decipm( f2 );
gEy( t>par.TSO2 & t<=par.TSF2 ) = par.gE_deci + par.gE_decipm( f2 );

%--------------connectivity
Iweight_ids  = cell(Nneurons,1);
Iweight_vals = cell(Nneurons,1);
initial_sinI = zeros(Nneurons,1);
for i=1:Nneurons/2,
  %note: ids are C-indices and run from 0...Nneurons-1
  Iweight_ids{i}  = (Nneurons/2+1:Nneurons) - 1;
  Iweight_vals{i} = ones(size(Iweight_ids{i}))/(Nneurons/2);
  initial_sinI(i) = rand(1) * 2;
end;
for i=Nneurons/2+1:Nneurons,
  Iweight_ids{i}  = (1:Nneurons/2) - 1;
  Iweight_vals{i} = ones(size(Iweight_ids{i}))/(Nneurons/2);
  initial_sinI(i) = rand(1) * 2;
end;

%--------------preparing structure for C-program

% general simulation parameters
G.T            = par.T;
G.Nt           = Nt;
G.dt           = dt;
G.t            = t;
G.report_every = 10;
G.rt           = 0:dt*G.report_every:par.T;
G.Nrt          = length( G.rt );

% iaf parameters
G.C            = par.C;
G.gleak        = par.gL;
G.Vleak        = par.EL;
G.Vthresh      = par.Vthresh;
G.Vreset       = par.Vreset;
G.refrac       = par.refrac;
G.EE           = par.EE;
G.EI           = par.EI;
%make gaussnoise independent of dt and membrane time constant
taum = par.C/par.gL;
var = par.gaussnoise.^2 * (1-dt/taum)/(dt/taum);
G.gaussnoise   = sqrt(var)*0.01418;

% synaptic parameters
G.Isyn_satmax  = par.Isatmax;
G.Itausyn      = par.Itau;
G.Igsyn        = par.wI;

% connectivity
G.Nneurons     = Nneurons;
G.Iweight_ids  = Iweight_ids;
G.Iweight_vals = Iweight_vals;

% external inputs
G.gE1          = gEx;
G.gE2          = gEy;

% Real inner variables, changed every timestep.
G.vv           = G.Vleak*ones(G.Nneurons,1);
%G.vv           = G.Vleak*ones(G.Nneurons,1) + 40*(G.Vthresh - G.Vleak) + randn(G.Nneurons,1);
G.ssoutI       = zeros(G.Nneurons,1);
G.ssinI        = zeros(G.Nneurons,1);
%Add noise here to "initial state"
% G.ssinI = [
%   par.init_mu(1) + sqrt(par.init_Sigma(1))*rand(par.Nneurons,1);
%   par.init_mu(2) + sqrt(par.init_Sigma(2))*rand(par.Nneurons,1)
%   ];
G.last_spike_time = -1e9*ones(G.Nneurons,1);

% reporting variables
G.v            = G.Vleak*ones(G.Nneurons,G.Nrt);
G.soutI        = zeros(G.Nneurons,G.Nrt);
G.sinI         = zeros(G.Nneurons,G.Nrt);
G.spikes       = zeros(G.Nneurons,G.Nrt);
G.nspikes      = zeros(G.Nneurons,1);

% C-program
ccspikesim(G);

%output
rt = G.rt;
gsx = par.wI * mean(G.sinI( Nneurons/2+1:Nneurons  , : ));
gsy = par.wI * mean(G.sinI(            1:Nneurons/2, : ));
spikesx   = G.spikes(Nneurons/2+1:Nneurons,   1:max(G.nspikes));
spikesy   = G.spikes(           1:Nneurons/2, 1:max(G.nspikes));
nspikes  = G.nspikes;

return;

%--------------create ROMO-data-like output

f = [10, 14 ,18, 22, 26, 30, 34];
%fill data into data array
for m=1:min(20,Nneurons/2)
  ind = find (spikes(m,:)>0);
  spikesarray(1,2*m-1) = {spikes(m,ind)};
  ind = find (spikes(Nneurons/2+m,:)>0);
  spikesarray(1,2*m)   = {spikes(Nneurons/2+m,ind)};
end
res = setvals( 'class', 0, 'trial', 0, 'hit', 1, ...
		'f1', f(f1), 'f2', f(f2), 'spikes', spikesarray, ...
		'PD', 0, 'KD', 0, 'SO1', par.TSO1, ...
		'SO2', par.TSO2, 'SF1', par.TSF1, ...
		'SF2', par.TSF2, 'KU', par.T, 'PU', par.T, ...
		'PK', par.T, 'RW', par.T, ...
		'of1', f(f1), 'of2', f(f2), ...
		'Nneurons', Nneurons/2 );
