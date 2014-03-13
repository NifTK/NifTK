% Compute the residual F = vMt.tMr.rMi.pI

function  D = CompCalResidual(params, tMrs, pIs)

   vMt = Comp_RigidBody_Matrix([params(7:9) 0 0 0]);
   rMi = Comp_RigidBody_Matrix(params(1:6));
   S = diag([params(10) params(11) 1 1]);

   D = [];
   
   for i= 1:size(pIs,1)
      v = vMt*tMrs{i}*rMi*S*pIs{i};
      D = [D;v(1:3)];
   end


   
   
