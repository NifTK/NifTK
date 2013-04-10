/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASApplication.h"
#include "QmitkNiftyMIDASAppWorkbenchAdvisor.h"

//-----------------------------------------------------------------------------
QmitkNiftyMIDASApplication::QmitkNiftyMIDASApplication()
{
}


//-----------------------------------------------------------------------------
QmitkNiftyMIDASApplication::QmitkNiftyMIDASApplication(const QmitkNiftyMIDASApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
berry::WorkbenchAdvisor* QmitkNiftyMIDASApplication::GetWorkbenchAdvisor()
{
  return new QmitkNiftyMIDASAppWorkbenchAdvisor();
}
