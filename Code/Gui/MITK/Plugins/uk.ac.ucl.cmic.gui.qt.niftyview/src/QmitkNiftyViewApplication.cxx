/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
