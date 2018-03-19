/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMITKViewPreferencePage.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyMITKViewPreferencePage::NiftyMITKViewPreferencePage()
: BaseApplicationPreferencePage()
{
}


//-----------------------------------------------------------------------------
NiftyMITKViewPreferencePage::NiftyMITKViewPreferencePage(const NiftyMITKViewPreferencePage& other)
: BaseApplicationPreferencePage(other)
{
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
NiftyMITKViewPreferencePage::~NiftyMITKViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void NiftyMITKViewPreferencePage::Init(berry::IWorkbench::Pointer workbench)
{
  BaseApplicationPreferencePage::Init(workbench);
}


//-----------------------------------------------------------------------------
void NiftyMITKViewPreferencePage::CreateQtControl(QWidget* parent)
{
  BaseApplicationPreferencePage::CreateQtControl(parent);

  /// You can add additional preferences by addRow calls to the parent form layout.
  //
  // QFormLayout* formLayout = qobject_cast<QFormLayout*>(m_MainControl->layout());
  // formLayout->addRow(...);

}


//-----------------------------------------------------------------------------
bool NiftyMITKViewPreferencePage::PerformOk()
{
  return BaseApplicationPreferencePage::PerformOk();
}


//-----------------------------------------------------------------------------
void NiftyMITKViewPreferencePage::PerformCancel()
{
  BaseApplicationPreferencePage::PerformCancel();
}


//-----------------------------------------------------------------------------
void NiftyMITKViewPreferencePage::Update()
{
  BaseApplicationPreferencePage::Update();
}

}
