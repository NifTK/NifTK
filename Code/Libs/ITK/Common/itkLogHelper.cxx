/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <iostream>
using namespace std;

#include "itkLogHelper.h"
#include "itkUCLMacro.h"
#include "ConversionUtils.h"

namespace niftk
{
  itkLogHelper::itkLogHelper()
  {
  }

  void itkLogHelper::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    Superclass::PrintSelf(os, indent);
  }

  itkLogHelper::~itkLogHelper()
  {
  }

  std::string itkLogHelper::ToString()
  {
    // No member variables yet, nothing to report.
    return "itkLogHelper[]";
  }

  void itkLogHelper::InfoMessage(const std::string& infoMessage)
  {
	  niftkitkInfoMacro(<< infoMessage);
  }

  void itkLogHelper::DebugMessage(const std::string& debugMessage)
  {
	  niftkitkDebugMacro(<< debugMessage);
  }

  void itkLogHelper::WarningMessage(const std::string& warningMessage)
  {
	  niftkitkWarningMacro(<< warningMessage);
  }

  void itkLogHelper::ErrorMessage(const std::string& errorMessage)
  {
	  niftkitkErrorMacro(<< errorMessage);
  }

  void itkLogHelper::ExceptionMessage(const std::string& exceptionMessage)
  {
	  niftkitkExceptionMacro(<< exceptionMessage);
  }

  std::string itkLogHelper::WriteParameterArray(const itk::Array<double>& array)
  {
    std::string result = std::string("array ");

    if (array.GetSize() < 20)
    {
      result += " contains[";
      for (unsigned int i = 0; i < array.GetSize(); i++)
      {
        result += (niftk::ConvertToString(array[i]) + ", ");
      }
      result += "]";
    }
    else
    {
      result += " is of size " + niftk::ConvertToString(((int)array.GetSize()));
    }

    return result;
  }

  void itkLogHelper::PrintCommandLineHeader(std::ostream& stream)
  {
    stream << NIFTK_COPYRIGHT << std::endl;
    stream << NIFTK_PLATFORM << ", " << NIFTK_VERSION_STRING << std::endl;
  }

}
