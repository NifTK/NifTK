/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-03-11 10:24:18 +0000 (Fri, 11 Mar 2011) $
 Revision          : $Revision: 5563 $
 Last modified by  : $Author: kkl $

 Original author   : 

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "math.h"
#include <cstdlib>
#include <limits>
#include <iostream>
using namespace std;

inline double cube_root(double a)
{
  if (a > 0)
    return pow(a, 1.0/3.0); 
  else
    return -pow(-a, 1.0/3.0); 
}


int main(int argc, char* argv[])
{
  const int numberOfImages = 3; 
  const double tol = 1e-15; 
  double b[numberOfImages][numberOfImages]; 
  double r[numberOfImages][numberOfImages]; 
  double prev_r[numberOfImages][numberOfImages]; 
  bool isWithinTolerance = false; 
  
  b[0][1] = atof(argv[1]); 
  b[1][2] = atof(argv[2]); 
  b[0][2] = atof(argv[3]); 
  b[1][0] = -b[0][1]; 
  b[2][1] = -b[1][2];
  b[2][0] = -b[0][2]; 
   
  r[1][0] = 1-b[0][1]; 
  r[2][0] = 1-b[0][2]; 
  r[2][1] = 1-b[1][2]; 
  r[0][1] = 1.0/r[1][0]; 
  r[0][2] = 1.0/r[2][0]; 
  r[1][2] = 1.0/r[2][1]; 
  
  int iteration = 0; 
  
  do 
  {  
    // save previous r.   
    for (int i = 0; i < numberOfImages; i++)
    {
      for (int j = i; j < numberOfImages; j++)
      {
        if (i != j)
          prev_r[i][j] = r[i][j]; 
      }
    }
    // compute new r. 
    for (int i = 0; i < numberOfImages; i++)
    {
      for (int j = i; j < numberOfImages; j++)
      {
        if (i != j)
        {
          double product = b[i][j]; 
          for (int k = 0; k < numberOfImages; k++)
          {
            if (i != k)
              product *= r[k][i]; 
          }
          r[j][i] = pow(1.0-product, 1.0/numberOfImages); 
          r[i][j] = 1.0/r[j][i]; 
        }
      }
    }
    std::cerr << "r=" << r[1][0] << "," << r[2][0] << "," << r[2][1] << std::endl; 
    
    isWithinTolerance = true; 
    for (int i = 0; i < numberOfImages; i++)
    {
      for (int j = i; j < numberOfImages; j++)
      {
        if (i != j)
        {
          if (fabs(prev_r[i][j]-r[i][j]) > tol)  
          {
            isWithinTolerance = false; 
          }
        }
      }
    }
    
    iteration++; 
  }
  while (iteration < 50 && !isWithinTolerance); 
  
  for (int i = 0; i < numberOfImages; i++)
  {
    for (int j = 0; j < numberOfImages; j++)
    {
      if (i != j)
        r[i][j] = pow(r[i][j], 3); 
    }
  }
  
#if 1
  #if 0
    //Transitive consistency. 
    // wrt 0m
    double l12=((1-r[1][0]) + ((r[0][1]-1)/r[0][1]) + ((r[0][2]-r[1][2])/r[0][2]))/3.0; 
    double l13=((1-r[2][0]) + ((r[0][2]-1)/r[0][2]) + ((r[0][1]-r[2][1])/r[0][1]))/3.0; 
    double l23=((r[1][0]-r[2][0]) + ((1-r[2][1])/r[0][1]) + ((r[1][2]-1)/r[0][2]))/3.0; 
    // wrt 24m. 
    double l12_wrt_24=( (1-r[1][0])/r[2][0] + (r[0][1]-1)/r[2][1] + (r[0][2]-r[1][2]) )/3.0; 
    double l13_wrt_24=( (1-r[2][0])/r[2][0] + (r[0][1]-r[2][1])/r[2][1] + (r[0][2]-1) )/3.0; 
    double l23_wrt_24=( (r[1][0]-r[2][0])/r[2][0] + (1-r[2][1])/r[2][1] + (r[1][2]-1) )/3.0; 
  #else
    //Transitive consistency. Right - we settle with the single one. 
    // wrt 0m
    double l12=(1-r[1][0]); 
    double l13=(1-r[2][0]); 
    double l23=(r[1][0]-r[2][0]); 
    // wrt 24m. 
    double l12_wrt_24=(r[0][2]-r[1][2]); 
    double l13_wrt_24=(r[0][2]-1); 
    double l23_wrt_24=(r[1][2]-1); 
  #endif
#else  
  // wrt 0m.
  double l12=cube_root((1-r[1][0]) * ((r[0][1]-1)/r[0][1]) * ((r[0][2]-r[1][2])/r[0][2])); 
  double l13=cube_root((1-r[2][0]) * ((r[0][2]-1)/r[0][2]) * ((r[0][1]-r[2][1])/r[0][1])); 
  double l23=cube_root((r[1][0]-r[2][0]) * ((1-r[2][1])/r[0][1]) * ((r[1][2]-1)/r[0][2])); 
  // wrt 24m. 
  double l12_wrt_24=cube_root( ((1-r[1][0])/r[2][0]) * ((r[0][1]-1)/r[2][1]) * (r[0][2]-r[1][2]) ); 
  double l13_wrt_24=cube_root( ((1-r[2][0])/r[2][0]) * ((r[0][1]-r[2][1])/r[2][1]) * (r[0][2]-1) ); 
  double l23_wrt_24=cube_root( ((r[1][0]-r[2][0])/r[2][0]) * ((1-r[2][1])/r[2][1]) * (r[1][2]-1) ); 
#endif   
  
  std::cout.precision(std::numeric_limits<double>::digits10 + 1);
  std::cout << "rates," << l12 << "," << l23 << "," << l13 << "," << l12_wrt_24 << "," << l23_wrt_24 << "," << l13_wrt_24 << std::endl; 
  
  std::cerr << 1-1/(1+l13_wrt_24) << "," << 1-1/(1+l13_wrt_24)-l13 << std::endl; 
  
  // Test input: 0.116650467 0.233300935 0.349951402
  // Test output: rates,0.0999999996293443,0.2000000001159507,0.299999999745295

  return 0;
}







