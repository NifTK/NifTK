/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyViewPreferencePage.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyViewPreferencePage::NiftyViewPreferencePage()
: BaseApplicationPreferencePage()
{
}


//-----------------------------------------------------------------------------
NiftyViewPreferencePage::NiftyViewPreferencePage(const NiftyViewPreferencePage& other)
: BaseApplicationPreferencePage(other)
{
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
NiftyViewPreferencePage::~NiftyViewPreferencePage()
{
}


//-----------------------------------------------------------------------------
void NiftyViewPreferencePage::Init(berry::IWorkbench::Pointer workbench)
{
  BaseApplicationPreferencePage::Init(workbench);
}


//-----------------------------------------------------------------------------
void NiftyViewPreferencePage::CreateQtControl(QWidget* parent)
{
  BaseApplicationPreferencePage::CreateQtControl(parent);

  /// You can add additional preferences by addRow calls to the parent form layout.
  //
  // QFormLayout* formLayout = qobject_cast<QFormLayout*>(m_MainControl->layout());
  // formLayout->addRow(...);

}


//-----------------------------------------------------------------------------
bool NiftyViewPreferencePage::PerformOk()
{
  return BaseApplicationPreferencePage::PerformOk();
}


//-----------------------------------------------------------------------------
void NiftyViewPreferencePage::PerformCancel()
{
  BaseApplicationPreferencePage::PerformCancel();
}


//-----------------------------------------------------------------------------
void NiftyViewPreferencePage::Update()
{
  BaseApplicationPreferencePage::Update();
}

}
