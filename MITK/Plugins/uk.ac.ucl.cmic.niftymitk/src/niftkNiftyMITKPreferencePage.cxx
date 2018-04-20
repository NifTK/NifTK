/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMITKPreferencePage.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyMITKPreferencePage::NiftyMITKPreferencePage()
: BaseApplicationPreferencePage()
{
}


//-----------------------------------------------------------------------------
NiftyMITKPreferencePage::NiftyMITKPreferencePage(const NiftyMITKPreferencePage& other)
: BaseApplicationPreferencePage(other)
{
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
NiftyMITKPreferencePage::~NiftyMITKPreferencePage()
{
}


//-----------------------------------------------------------------------------
void NiftyMITKPreferencePage::Init(berry::IWorkbench::Pointer workbench)
{
  BaseApplicationPreferencePage::Init(workbench);
}


//-----------------------------------------------------------------------------
void NiftyMITKPreferencePage::CreateQtControl(QWidget* parent)
{
  BaseApplicationPreferencePage::CreateQtControl(parent);

  /// You can add additional preferences by addRow calls to the parent form layout.
  //
  // QFormLayout* formLayout = qobject_cast<QFormLayout*>(m_MainControl->layout());
  // formLayout->addRow(...);

}


//-----------------------------------------------------------------------------
bool NiftyMITKPreferencePage::PerformOk()
{
  return BaseApplicationPreferencePage::PerformOk();
}


//-----------------------------------------------------------------------------
void NiftyMITKPreferencePage::PerformCancel()
{
  BaseApplicationPreferencePage::PerformCancel();
}


//-----------------------------------------------------------------------------
void NiftyMITKPreferencePage::Update()
{
  BaseApplicationPreferencePage::Update();
}

}
