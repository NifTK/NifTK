/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLogHelper_h
#define niftkLogHelper_h

#include <niftkCommonWin32ExportHeader.h>

#include <iostream>


namespace niftk
{

/**
 * \class LogHelper
 * \brief This is a class to help with a few logging functions.
 */
class NIFTKCOMMON_WINEXPORT LogHelper
{

public:

  /// \brief Prints out command line header
  static void PrintCommandLineHeader(std::ostream& stream);

protected:
  LogHelper();
  virtual ~LogHelper();

private:
  LogHelper(const LogHelper&); //purposely not implemented
  void operator=(const LogHelper&); //purposely not implemented

};

}

#endif
