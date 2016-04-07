/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
//#pragma warning ( disable : 4786 )
#endif

#include <niftkAddTwoIntegers.h>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <sstream>

int niftkAddTwoIntegersKernelTests(int argc, char* argv[])
{
  bool result = true;

  try
  {
    int result = niftk::AddTwoIntegers(1, 2);
    if (result != 3)
    {
      std::stringstream oss;
      oss << "1+2 should equal 3, but it equals " << result << std::endl;
      throw std::runtime_error(oss.str());
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
