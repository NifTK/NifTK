/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include <cstdlib>
#include <niftkVTKIterativeClosestPoint.h>

/**
 * Runs ICP registration a known data set and checks the error
 */

int VTKIterativeClosestPointTest ( int argc, char * argv[] ) 
{

  niftk::IterativeClosestPoint tester;
    std::cerr << "Boing";
  for ( int i = 0 ; i < argc ; i ++ ) 
  {
    std::cerr << argv[i] << std::endl;
  }

  //return EXIT_SUCCESS;
  return EXIT_FAILURE;
 // return 0 ;
}
