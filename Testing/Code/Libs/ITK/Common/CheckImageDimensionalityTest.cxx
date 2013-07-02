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
#include <itkCommandLineHelper.h>

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
