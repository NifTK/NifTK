/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-26 10:10:39 +0100 (Fri, 26 Aug 2011) $
 Revision          : $Revision: 7170 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewApplication.h"
#include "QmitkNiftyViewAppWorkbenchAdvisor.h"

//-----------------------------------------------------------------------------
QmitkNiftyViewApplication::QmitkNiftyViewApplication()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyViewApplication::QmitkNiftyViewApplication(const QmitkNiftyViewApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
berry::WorkbenchAdvisor* QmitkNiftyViewApplication::GetWorkbenchAdvisor()
{
  return new QmitkNiftyViewAppWorkbenchAdvisor();
}
