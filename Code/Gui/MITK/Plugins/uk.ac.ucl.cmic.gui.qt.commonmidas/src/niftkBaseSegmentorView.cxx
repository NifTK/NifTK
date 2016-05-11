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


const QString niftkBaseSegmentorView::DEFAULT_COLOUR("midas editor default colour");
const QString niftkBaseSegmentorView::DEFAULT_COLOUR_STYLE_SHEET("midas editor default colour style sheet");


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::~niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::Activated()
{
  QmitkBaseView::Activated();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsActivated();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::Deactivated()
{
  QmitkBaseView::Deactivated();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsDeactivated();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::Visible()
{
  QmitkBaseView::Visible();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsVisible();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::Hidden()
{
  QmitkBaseView::Hidden();

  assert(m_SegmentorController);
  m_SegmentorController->OnViewGetsHidden();
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorView::niftkBaseSegmentorView(const niftkBaseSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_SegmentorController = this->CreateSegmentorController();
  m_SegmentorController->SetupGUI(parent);

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::onVisibilityChanged(const mitk::DataNode* node)
{
  assert(m_SegmentorController);
  m_SegmentorController->OnNodeVisibilityChanged(node);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::NodeChanged(const mitk::DataNode* node)
{
  assert(m_SegmentorController);
  m_SegmentorController->OnNodeChanged(node);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::NodeRemoved(const mitk::DataNode* node)
{
  assert(m_SegmentorController);
  m_SegmentorController->OnNodeRemoved(node);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer>& nodes)
{
  Q_UNUSED(part);
  assert(m_SegmentorController);
  m_SegmentorController->OnDataManagerSelectionChanged(nodes);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  QString defaultColourName = prefs->Get(niftkBaseSegmentorView::DEFAULT_COLOUR, "");
  QColor defaultSegmentationColour(defaultColourName);
  if (defaultColourName == "")
  {
    defaultSegmentationColour = QColor(0, 255, 0);
  }
  m_SegmentorController->SetDefaultSegmentationColour(defaultSegmentationColour);
}
