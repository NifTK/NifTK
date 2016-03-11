% MAGC Column vector magnitudes (D.C. Barratt)
%
% MAGC(M), where M = [c1 c2 ... cN], returns the magnitudes of the column vectors
% in the vector v = [ |c1| |c2| ... |cN| ]

function v = magc(M)

v = sqrt(sum(M.^2,1));
