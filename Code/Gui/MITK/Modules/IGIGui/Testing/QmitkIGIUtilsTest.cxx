/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkTestingMacros.h>
#include <QString>
#include <QmitkIGIUtils.h>

/**
 * \brief Various tests for QmitkIGIUtils.
 */
int QmitkIGIUtilsTest(int argc, char* argv[])
{

  if (argc != 1)
  {
    std::cerr << "Usage: QmitkIGIUtilsTest" << std::endl;
    return EXIT_FAILURE;
  }

  QString tmp = ConvertNanoSecondsToString(1);
  MITK_TEST_CONDITION_REQUIRED(tmp == QString("000000001"), ".. Testing if 1 nanoseconds converts to 000000001, but actually we have " << tmp.toStdString());

  tmp = ConvertNanoSecondsToString(1234);
  MITK_TEST_CONDITION_REQUIRED(tmp == QString("000001234"), ".. Testing if 1234 nanoseconds converts to 000001234, but actually we have " << tmp.toStdString());

  return EXIT_SUCCESS;
}
