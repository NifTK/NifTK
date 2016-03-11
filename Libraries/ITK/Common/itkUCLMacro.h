/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUCLMacro_h
#define itkUCLMacro_h

#include <NifTKConfigure.h>
#include <niftkCommonWin32ExportHeader.h>

#include <iostream>
#include <itkObject.h>
#include <itkArray.h>
#include <itkMacro.h>

namespace niftk
{

#define niftkitkDebugMacro(x)     itkDebugMacro(x)

#define niftkitkWarningMacro(x)   itkWarningMacro(x)

#define niftkitkExceptionMacro(x) itkExceptionMacro(x)


#define niftkitkInfoMacro(x)                                          \
{                                                                     \
  if ( ::itk::Object::GetGlobalWarningDisplay() )                     \
  {                                                                   \
    std::ostringstream itkmsg;                                        \
    itkmsg << "Info: In " __FILE__ ", line " << __LINE__ << "\n"      \
       << this->GetNameOfClass() << " (" << this << "): " x           \
       << "\n\n";                                                     \
    ::itk::OutputWindowDisplayText( itkmsg.str().c_str() );           \
  }                                        \
}                                     \


#define niftkitkErrorMacro(x)                                        \
{                                                                    \
  if ( ::itk::Object::GetGlobalWarningDisplay() )                    \
  {                                                                  \
    std::ostringstream itkmsg;                                       \
    itkmsg << "Error: In " __FILE__ ", line " << __LINE__ << "\n"    \
      << this->GetNameOfClass() << " (" << this << "): " x           \
      << "\n\n";                                                     \
    ::itk::OutputWindowDisplayText( itkmsg.str().c_str() );          \
  }                                                                  \
}


/// This class ensures that the text output is kept in the shell instead of redirecting it
/// a text output window, what would happen on Windows by default. See #3497 for details.
class KeepTextOutputInShell
{

  KeepTextOutputInShell();

  static KeepTextOutputInShell s_KeepTextOutputInShell;

};


}

#endif
