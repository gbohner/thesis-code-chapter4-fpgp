function [nullcline1_fr, nullcline2_fr] = transform_nullclines_to_fr(nullcline1,nullcline2, xgrd, fx_fr_from_inhib)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

nullcline1_fr(1,:) = iofunc(nullcline1(1,:)*1e-3, xgrd, fx_fr_from_inhib);
nullcline1_fr(2,:) = iofunc(nullcline1(2,:)*1e-3, xgrd, fx_fr_from_inhib);
nullcline2_fr(1,:) = iofunc(nullcline2(1,:)*1e-3, xgrd, fx_fr_from_inhib);
nullcline2_fr(2,:) = iofunc(nullcline2(2,:)*1e-3, xgrd, fx_fr_from_inhib);


% Helper function for transform
function y = iofunc ( x, xgrd, fx)

N     = length( xgrd );
xmin  = min( xgrd );
xmax  = max( xgrd );
fxmax = max( fx );

y  = zeros( size(x) );
u1 = find( xmax <x );
u2 = find( xmin<=x & x<=xmax );
u3 = find( x<xmin );

if ~isempty(u1); y(u1) = fxmax; end
if ~isempty(u2);
  ind = 1+round( N * (x(u2)-xgrd(1))/ ( xgrd(end)-xgrd(1) ));
  ind( ind>N ) = N;
  y(u2) = fx( ind );
end
if ~isempty(u3); y(u3) = 0; end;

% Also add baseline % firing rate to the i/o func (do not allow 0 output)

end

end

