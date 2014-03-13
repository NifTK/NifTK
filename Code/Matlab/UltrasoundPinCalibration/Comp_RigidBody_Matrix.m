% Function to compute a 4x4 homogeneous rigid-body transformation matrix (rotation and translation)

function M = Comp_RigidBody_Matrix(params)

   x = params(1);       y = params(2);      z = params(3);
   alpha = params(4);   beta = params(5);   gamma = params(6);
    
   M = [cos(beta)*cos(gamma),   cos(gamma)*sin(beta)*sin(alpha)+sin(gamma)*cos(alpha),      sin(alpha)*sin(gamma) - cos(gamma)*sin(beta)*cos(alpha),  x;
		 -sin(gamma)*cos(beta),   cos(gamma)*cos(alpha) - sin(gamma)*sin(beta)*sin(alpha),    sin(beta)*cos(alpha)*sin(gamma)+cos(gamma)*sin(alpha),    y;
		  sin(beta),             -sin(alpha)*cos(beta),                                       cos(alpha)*cos(beta),                                     z;
          0,0,0,1];
      

      
   
      
   

