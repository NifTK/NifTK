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
#ifndef __ITKUCLMACRO_H
#define __ITKUCLMACRO_H

#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"

#include <iostream>
#include "itkObject.h"
#include "itkArray.h"
#include "itkMacro.h"

namespace niftk
{

#define niftkitkDebugMacro(x)     itkDebugMacro(x)	                         \

#define niftkitkWarningMacro(x)   itkWarningMacro(x) 				    	 \

#define niftkitkExceptionMacro(x) itkExceptionMacro(x)				         \


  #define niftkitkInfoMacro(x) 										      \
  {                               									      \
    if ( ::itk::Object::GetGlobalWarningDisplay() )                       \
    { 																      \
      std::ostringstream itkmsg;										  \
      itkmsg << "Info: In " __FILE__ ", line " << __LINE__ << "\n" 	      \
         << this->GetNameOfClass() << " (" << this << "): " x       	  \
         << "\n\n";												          \
      ::itk::OutputWindowDisplayText( itkmsg.str().c_str() );             \
    }        															  \
  } 																	  \



  #define niftkitkErrorMacro(x) 											  \
  {                               									      \
    if ( ::itk::Object::GetGlobalWarningDisplay() )                       \
    { 																      \
      std::ostringstream itkmsg;										  \
      itkmsg << "Error: In " __FILE__ ", line " << __LINE__ << "\n" 	  \
        << this->GetNameOfClass() << " (" << this << "): " x       	      \
        << "\n\n";												          \
      ::itk::OutputWindowDisplayText( itkmsg.str().c_str() );             \
    }        															  \
  } 																	  \


} //end of namespace niftk

#endif

