/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-13 10:54:10 +0000 (Tue, 13 Dec 2011) $
 Revision          : $Revision: 8003 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "niftkFluid.txx"

int main(int argc, char** argv)
{
  time_t start = clock(); 
  
  int returnValue = fluid_main<3, short>(argc, argv); 
  time_t timeUsed = clock()-start; 
  std::cerr << std::endl << "Total time used=" << (timeUsed/CLOCKS_PER_SEC)/60 << "min " << (timeUsed/CLOCKS_PER_SEC)%60 << "s" << std::endl; 
      
  return returnValue; 
}




