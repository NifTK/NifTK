% PLOT3D   Plots a matrix of 3D points  (D.C. Barratt 2001)
%
% plot3d(P,labelpts,varargin)  plots the matrix P = [p1 p2 ... pN] 
% labelpts = 0/1 or true/false
% varargin = style string, e.g. '.r' 
% Similar to plot3

function plot3d(P,labelpts,varargin)

if isempty(P) | (nargin == 0)
   return
end

holdon = ishold;

if (nargin >= 2)
  if labelpts
    for p = 1:size(P,2)
       text(P(1,p)+3,P(2,p)+3,P(3,p)+3,num2str(p));
       hold on;
    end
  end
end

plot3(P(1,:),P(2,:),P(3,:),varargin{:});

if holdon
    hold on;
else
    hold off;
end

axis equal
   
 

	