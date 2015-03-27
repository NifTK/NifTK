/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "IGIOverlayEditor2PreferencePage.h"
#include <IGIOverlayEditor2.h>

#include <QLabel>
#include <QPushButton>
#include <QFormLayout>
#include <QRadioButton>
#include <QColorDialog>
#include <QCheckBox>
#include <ctkPathLineEdit.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>


//-----------------------------------------------------------------------------
const char* IGIOverlayEditor2PreferencePage::BACKGROUND_COLOR_PREFSKEY = "background colour";


//-----------------------------------------------------------------------------
IGIOverlayEditor2PreferencePage::IGIOverlayEditor2PreferencePage()
  : m_MainControl(0)
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::Init(berry::IWorkbench::Pointer )
{
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::CreateQtControl(QWidget* parent)
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  m_IGIOverlayEditor2PreferencesNode = prefService->GetSystemPreferences()->Node(IGIOverlayEditor2::EDITOR_ID);

  m_MainControl = new QWidget(parent);

  this->Update();
}


//-----------------------------------------------------------------------------
QWidget* IGIOverlayEditor2PreferencePage::GetQtControl() const
{
  return m_MainControl;
}


//-----------------------------------------------------------------------------
bool IGIOverlayEditor2PreferencePage::PerformOk()
{
  return true;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::PerformCancel()
{

}


//-----------------------------------------------------------------------------
void IGIOverlayEditor2PreferencePage::Update()
{
}
