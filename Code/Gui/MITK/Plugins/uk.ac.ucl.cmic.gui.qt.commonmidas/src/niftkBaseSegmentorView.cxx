/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorView.h"

#include <berryPlatform.h>
#include <berryIBerryPreferences.h>
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>

#include <niftkBaseSegmentorController.h>

namespace niftk
{

const QString BaseSegmentorView::DEFAULT_COLOUR("midas editor default colour");
const QString BaseSegmentorView::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");


//-----------------------------------------------------------------------------
BaseSegmentorView::BaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
BaseSegmentorView::~BaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::Activated()
{
  QmitkBaseView::Activated();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsActivated();
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::Deactivated()
{
  QmitkBaseView::Deactivated();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsDeactivated();
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::Visible()
{
  QmitkBaseView::Visible();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsVisible();
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::Hidden()
{
  QmitkBaseView::Hidden();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsHidden();
}


//-----------------------------------------------------------------------------
BaseSegmentorView::BaseSegmentorView(const BaseSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_SegmentorController = this->CreateSegmentorController();
  m_SegmentorController->SetupGUI(parent);

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& nodes)
{
  Q_UNUSED(part);
  assert(m_SegmentorController);
  m_SegmentorController->OnDataManagerSelectionChanged(nodes);
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void BaseSegmentorView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  QString defaultColourName = prefs->Get(BaseSegmentorView::DEFAULT_COLOUR, "");
  QColor defaultSegmentationColour(defaultColourName);
  if (defaultColourName == "")
  {
    defaultSegmentationColour = QColor(0, 255, 0);
  }
  m_SegmentorController->SetDefaultSegmentationColour(defaultSegmentationColour);
}

}
