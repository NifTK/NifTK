/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseApplication.h"

#include <berryPlatformUI.h>


namespace niftk
{

//-----------------------------------------------------------------------------
BaseApplication::BaseApplication()
{
}


//-----------------------------------------------------------------------------
BaseApplication::BaseApplication(const BaseApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QVariant BaseApplication::Start(berry::IApplicationContext* context)
{
  Q_UNUSED(context);

  berry::Display* display = berry::PlatformUI::CreateDisplay();

  int code = berry::PlatformUI::CreateAndRunWorkbench(display, this->GetWorkbenchAdvisor());

  // exit the application with an appropriate return code
  return code == berry::PlatformUI::RETURN_RESTART ? EXIT_RESTART : EXIT_OK;
}


//-----------------------------------------------------------------------------
void BaseApplication::Stop()
{
}

}
