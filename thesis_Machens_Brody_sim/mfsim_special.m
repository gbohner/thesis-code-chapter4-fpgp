%mfsim - mean field simulation
%
%   mfsim( f1, f2 ) displays an animated mean-field
%   simulation of the two-interval discrimination task
%   with first stimulus 'f1' and second stimulus 'f2'.
%   By default, 'f1' and 'f2' have to be integers
%   within the interval [1,7]. mfsim uses the parameters
%   and i/o function provided in the function 'initpar.m'
%   
%   mfsim( f1, f2, pausetime ) allows to control the
%   speed of the simulation by adjusting 'pausetime':
%   this is the time (in sec) paused after every
%   0.01 sec of simulated time (default = 0.01 sec).
%
%   [x,y,t] = mfsim( f1, f2, pausetime, doplot );
%   returns the traces of the plus ('x') and minus
%   'y' neurons at the time points 't'. Set 'pausetime'
%   and 'doplot' to zero if you want to skip the
%   animation.

%   (c) CK Machens & CD Brody

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xx, yy, tt, nullcline1, nullcline2] = mfsim(par, f1, f2, pausetime, doplot )

%--------------check frequencies

%initpar;


if (f1<1 | f1>length(par.gE_loadpm))
  error('f1 out of range');
end
if (f2<1 | f2>length(par.gE_decipm))
  error('f2 out of range');
end

%--------------default values
if (nargin==3)
  pausetime = 0.01;   %(in seconds)
  doplot = 1;
end
if (nargin==4)
  doplot = 1;
end

%--------------initialize parameters
global xgrd;
global fx;
xgrd = par.x;       % i/o function: x
fx   = par.fx;      % i/o function: y=f(x)
lambda= par.lambda; % exc/inh scaling factor
w    = par.wI;      % inhibitory weight
tau  = par.Itau;    % time constant of dynamics
gn   = par.mfnoise; % additive noise
t    = 1:par.T;     % note: time step dt = 1 msec

%--------------external inputs
Ex = zeros(1,par.T);
Ey = zeros(1,par.T);

%during loading
Ex( par.TSO1+1:par.TSF1 ) = par.gE_load + par.gE_loadpm( f1 );
Ey( par.TSO1+1:par.TSF1 ) = par.gE_load - par.gE_loadpm( f1 );

%during maintenance
Ex( par.TSF1+1:par.TSO2 ) = par.gE;
Ey( par.TSF1+1:par.TSO2 ) = par.gE;

%during decision
Ex( par.TSO2+1:par.TSF2 ) = par.gE_deci - par.gE_decipm( f2 );
Ey( par.TSO2+1:par.TSF2 ) = par.gE_deci + par.gE_decipm( f2 );

%post-decision
%Ex( par.TSF2+1:par.T ) = par.gE_deci;  % one possibility to store
%Ey( par.TSF2+1:par.T ) = par.gE_deci;  % the decision outcome


%--------------initialize plots
if (doplot)

  figure(2); clf;
  set(gcf,'MenuBar','none');
  pos = get(gcf,'Position');
  set(gcf,'Position', [pos(1) pos(2) 360 600] );

  subplot('Position', [0.15 0.5 0.7 0.4]);
  nc1 = plot( xgrd, xgrd ); hold on;
  nc2 = plot( xgrd, xgrd );
  posxy = plot(0,0, '.');
  set( nc1, 'Color', 'k', 'EraseMode', 'xor' );
  set( nc2, 'Color', 'g', 'EraseMode', 'xor' ); 
  set( posxy, 'Color', 'r', 'EraseMode', 'xor' );
  set( posxy, 'MarkerSize', 20 );
  xlabel('nS (plus neuron)');
  ylabel('nS (minus neuron)');
  tt = text( 3, 4.5, 't=0' );
  set( tt, 'EraseMode', 'xor' );
  axis( [0 5 0 5] );

  subplot('Position', [0.15 0.15 0.7 0.2]);
  traj = plot(0,0);
  set( traj, 'Color', 'r', 'EraseMode', 'xor' );
  xlabel('time (msec)');
  ylabel('nS (plus neuron)');
  axis( [0 par.T 0 5] );

end

%--------------simulation
x=0; y=0;
x = par.init_mu(1) + sqrt(par.init_Sigma(1))*randn(1);
y = par.init_mu(2) + sqrt(par.init_Sigma(2))*randn(1);

xx = []; yy = [];
xx = [xx, w*x]; yy = [yy, w*y];

% Save the nullclines
nullcline1 = [];
nullcline1(1,:) = 1000*xgrd;
nullcline1(2,:) = 1000*w*iofunc( -xgrd + lambda*Ey(1), par.iobaseline);
nullcline2 = [];
nullcline2(1,:) = 1000*w*iofunc( -xgrd + lambda*Ex(1), par.iobaseline);
nullcline2(2,:) = 1000*xgrd;

for k=1:length(t)

  % Add noise at discrete time steps
  if mod(k, par.noiseTimeStep) == 0
    x = x + sqrt(par.noise_eps(1))*randn(1);
    y = y + sqrt(par.noise_eps(2))*randn(1);
  end
  
  
  % ODE integration using Euler's method
  dx = -x + iofunc( -w*y + lambda*Ex(k), par.iobaseline );
  dy = -y + iofunc( -w*x + lambda*Ey(k), par.iobaseline );
  x = x+dx/tau; %note: dt = 1 and hence dx*dt = dx
  y = y+dy/tau; 
  
  % plot nullclines and (x,y)
  if ( ~mod(k,par.noiseTimeStep) )
    xx = [xx, w*x]; yy = [yy, w*y];
    if (doplot)
      set( nc1, 'XData', 1000*xgrd, ...
		'YData', 1000*w*iofunc( -xgrd + lambda*Ey(k), par.iobaseline) );
      set( nc2, 'XData', 1000*w*iofunc( -xgrd + lambda*Ex(k), par.iobaseline),...
		'YData', 1000*xgrd  );
      set( posxy, 'XData', 1000*w*x, 'YData', 1000*w*y );
      set( tt, 'String', sprintf( 't = %d msec', k ) );
      set( traj, 'XData', (1:k/10)*10, 'YData', 1000*xx )
      drawnow;
      pause( pausetime ); %slows down the dynamics;
    end
  end
end
tt = t(1:par.noiseTimeStep:end);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% iofunc simulates the neuron's io-function y=f(x)
% using the vectors xgrd and fx
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = iofunc ( x, baseline )

global xgrd;
global fx;

if nargin == 1
  baseline = 0;
end

N     = length( xgrd );
xmax  = max( xgrd );
fxmax = max( fx );

y  = zeros( size(x) );
u1 = find( xmax <=x );
u2 = find( 0<x & x<xmax );
u3 = find( x<=0 );

if ~isempty(u1); y(u1) = fxmax; end
if ~isempty(u2);
  ind = 1+round( N * (x(u2)-xgrd(1))/ ( xgrd(end)-xgrd(1) ));
  ind( ind>N ) = N;
  y(u2) = fx( ind );
end
if ~isempty(u3); y(u3) = 0; end;

% Also add baseline % firing rate to the i/o func (do not allow 0 output)
y = y+fxmax*baseline;

end
