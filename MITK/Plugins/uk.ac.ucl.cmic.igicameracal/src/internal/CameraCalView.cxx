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
#include <QPixmap>
#include <QtConcurrentRun>

namespace niftk
{

const std::string CameraCalView::VIEW_ID = "uk.ac.ucl.cmic.igicameracal";

//-----------------------------------------------------------------------------
CameraCalView::CameraCalView()
: m_Controls(nullptr)
, m_Manager(nullptr)
{
  bool ok = false;
  ok = connect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
CameraCalView::~CameraCalView()
{
  m_BackgroundProcessWatcher.waitForFinished();

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

    // Start these up as disabled, until we have enough images to calibrate.
    m_Controls->m_UndoButton->setEnabled(false);
    m_Controls->m_SaveButton->setEnabled(false);

    // Create manager, before we retrieve preferences which will populate it.
    m_Manager = niftk::NiftyCalVideoCalibrationManager::New();
    m_Manager->SetDataStorage(dataStorage);

    // Get user prefs, so we can decide if we doing chessboards/AprilTags etc.
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
void CameraCalView::OnComboBoxChanged()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

  mitk::DataNode::Pointer leftImageNode = m_Controls->m_LeftCameraComboBox->GetSelectedNode();
  mitk::DataNode::Pointer rightImageNode = m_Controls->m_RightCameraComboBox->GetSelectedNode();
  mitk::DataNode::Pointer trackingNode = m_Controls->m_TrackerMatrixComboBox->GetSelectedNode();

  int numberOfSnapshots = m_Manager->GetNumberOfSnapshots();
  if (numberOfSnapshots > 0)
  {
    bool needsReset = false;

    mitk::DataNode::Pointer leftImageNodeInManager = m_Manager->GetLeftImageNode();
    if (   leftImageNode.IsNotNull()
        && leftImageNodeInManager.IsNotNull()
        && leftImageNode != leftImageNodeInManager)
    {
      needsReset = true;
    }

    mitk::DataNode::Pointer rightImageNodeInManager = m_Manager->GetRightImageNode();
    if (   rightImageNode.IsNotNull()
        && rightImageNodeInManager.IsNotNull()
        && rightImageNode != rightImageNodeInManager)
    {
      needsReset = true;
    }

    mitk::DataNode::Pointer trackingNodeInManager = m_Manager->GetTrackingTransformNode();
    if (   trackingNode.IsNotNull()
        && trackingNodeInManager.IsNotNull()
        && trackingNode != trackingNodeInManager)
    {
      needsReset = true;
    }

    if (needsReset)
    {
      QMessageBox msgBox;
      msgBox.setText("Reset requested.");
      msgBox.setInformativeText("Changing an image or tracking transform while calibrating causes a reset.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();

      m_Manager->Restart();
    }
  }

  m_Manager->SetLeftImageNode(leftImageNode);
  m_Manager->SetRightImageNode(rightImageNode);
  m_Manager->SetTrackingTransformNode(trackingNode);
}


//-----------------------------------------------------------------------------
void CameraCalView::OnGrabButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

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

  bool successfullyGrabbed = m_Manager->Grab();

  if (m_Manager->GetNumberOfSnapshots() > 0)
  {
    m_Controls->m_UndoButton->setEnabled(true);
  }

  if (successfullyGrabbed)
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/green-tick-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();

    this->Calibrate();
  }
  else
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/red-cross-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::Calibrate()
{
  if (m_Manager->GetNumberOfSnapshots()
      >= m_Manager->GetMinimumNumberOfSnapshotsForCalibrating())
  {
    m_Controls->m_LeftCameraComboBox->setEnabled(false);
    m_Controls->m_RightCameraComboBox->setEnabled(false);
    m_Controls->m_TrackerMatrixComboBox->setEnabled(false);
    m_Controls->m_GrabButton->setEnabled(false);
    m_Controls->m_UndoButton->setEnabled(false);
    m_Controls->m_SaveButton->setEnabled(false);

    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/boobaloo-Don-t-Step-No-Gnome--300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();

    m_BackgroundProcess = QtConcurrent::run(this, &CameraCalView::RunCalibration);
    m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
  }
}


//-----------------------------------------------------------------------------
double CameraCalView::RunCalibration()
{
  return m_Manager->Calibrate();
}


//-----------------------------------------------------------------------------
void CameraCalView::OnBackgroundProcessFinished()
{
  QPixmap image(":/uk.ac.ucl.cmic.igicameracal/green-tick-300px.png");
  m_Controls->m_ImageLabel->setPixmap(image);
  m_Controls->m_ImageLabel->show();

  m_Controls->m_LeftCameraComboBox->setEnabled(true);
  m_Controls->m_RightCameraComboBox->setEnabled(true);
  m_Controls->m_TrackerMatrixComboBox->setEnabled(true);
  m_Controls->m_GrabButton->setEnabled(true);
  m_Controls->m_UndoButton->setEnabled(true);

  if (m_Manager->GetNumberOfSnapshots()
      >= m_Manager->GetMinimumNumberOfSnapshotsForCalibrating())
  {
    m_Controls->m_SaveButton->setEnabled(true);
  }

}


//-----------------------------------------------------------------------------
void CameraCalView::OnUndoButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

  m_Manager->Undo();

  if (m_Manager->GetNumberOfSnapshots()
      < m_Manager->GetMinimumNumberOfSnapshotsForCalibrating())
  {
    m_Controls->m_SaveButton->setEnabled(false);
  }

  this->Calibrate();
}


//-----------------------------------------------------------------------------
void CameraCalView::OnSaveButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundProcess.isRunning());

  QString dir = QFileDialog::getExistingDirectory(nullptr, tr("Save"),
                                                  m_DefaultSaveDirectory,
                                                  QFileDialog::ShowDirsOnly
                                                  | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty())
  {
    m_Manager->Save(dir.toStdString());
  }
}


} // end namespace
