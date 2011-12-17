/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include "itkCommandLineHelper.h"

/**
 * Simply runs the function itk::itkCommandLineHelper::GetImageDimension(filename)
 */
int CheckImageDimensionalityTest(int argc, char * argv[])
{
  if (argc != 3)
    {
      std::cerr << "Usage: CheckImageDimensionalityTest filename expectedNumberOfDimensions" << std::endl;
      return EXIT_FAILURE;
    }
    
  std::string filename = argv[1];
  int expected = atoi(argv[2]);
  int actual = itk::PeekAtImageDimension(filename);
  
  if (actual != expected)
    {
      std::cerr << "Expected:" << expected << ", but got:" << actual << std::endl;
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
