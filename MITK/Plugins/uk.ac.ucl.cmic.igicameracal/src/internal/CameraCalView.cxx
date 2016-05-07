/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "CameraCalView.h"
#include "CameraCalViewPreferencePage.h"
#include <mitkNodePredicateDataType.h>
#include <mitkCoordinateAxesData.h>
#include <mitkImage.h>
#include <QMessageBox>
#include <QFileDialog>

namespace niftk
{

const std::string CameraCalView::VIEW_ID = "uk.ac.ucl.cmic.igicameracal";

//-----------------------------------------------------------------------------
CameraCalView::CameraCalView()
: m_Controls(NULL)
, m_NumberSuccessfulViews(0)
{
}


//-----------------------------------------------------------------------------
CameraCalView::~CameraCalView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string CameraCalView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void CameraCalView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls = new Ui::CameraCalView();
    m_Controls->setupUi(parent);

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    m_Controls->m_LeftCameraComboBox->SetPredicate(isImage);
    m_Controls->m_LeftCameraComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_RightCameraComboBox->SetPredicate(isImage);
    m_Controls->m_RightCameraComboBox->SetAutoSelectNewItems(false);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isMatrix = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_TrackerMatrixComboBox->SetPredicate(isMatrix);
    m_Controls->m_TrackerMatrixComboBox->SetAutoSelectNewItems(false);

    m_Controls->m_LeftCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_RightCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_TrackerMatrixComboBox->SetDataStorage(dataStorage);

    // I'm trying to stick to only 3 buttons, so we can easily link to foot switch.
    connect(m_Controls->m_GrabButton, SIGNAL(pressed()), this, SLOT(OnGrabButtonPressed()));
    connect(m_Controls->m_UndoButton, SIGNAL(pressed()), this, SLOT(OnUndoButtonPressed()));
    connect(m_Controls->m_SaveButton, SIGNAL(pressed()), this, SLOT(OnSaveButtonPressed()));

    // Hook up combo boxes, so we know when user changes node
    connect(m_Controls->m_LeftCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_RightCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_TrackerMatrixComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));

    m_Controls->m_UndoButton->setEnabled(false);
    m_Controls->m_SaveButton->setEnabled(false);

    RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void CameraCalView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::SetFocus()
{
  m_Controls->m_LeftCameraComboBox->setFocus();
}


//-----------------------------------------------------------------------------
void CameraCalView::OnGrabButtonPressed()
{
  mitk::DataNode::Pointer node = m_Controls->m_LeftCameraComboBox->GetSelectedNode();
  if (node.IsNull())
  {
    QMessageBox msgBox;
    msgBox.setText("The left camera image is non-existent, or not-selected.");
    msgBox.setInformativeText("Please select a left camera image.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  // Basically - each grab, we see if we can calibrate.
}


//-----------------------------------------------------------------------------
void CameraCalView::OnUndoButtonPressed()
{
  std::cout << "Matt, OnUndoButtonPressed" << std::endl;
}


//-----------------------------------------------------------------------------
void CameraCalView::OnSaveButtonPressed()
{
  std::cout << "Matt, OnSaveButtonPressed" << std::endl;
}


//-----------------------------------------------------------------------------
void CameraCalView::OnComboBoxChanged()
{
  std::cout << "Matt, OnComboBoxChanged" << std::endl;
}

} // end namespace
