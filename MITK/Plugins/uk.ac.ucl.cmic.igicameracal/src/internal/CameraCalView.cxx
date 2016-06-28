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
#include "CameraCalViewActivator.h"
#include <niftkNiftyCalException.h>
#include <mitkNodePredicateDataType.h>
#include <mitkCoordinateAxesData.h>
#include <mitkImage.h>
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QtConcurrentRun>
#include <ctkServiceReference.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <service/event/ctkEventConstants.h>

namespace niftk
{

const std::string CameraCalView::VIEW_ID = "uk.ac.ucl.cmic.igicameracal";

//-----------------------------------------------------------------------------
CameraCalView::CameraCalView()
: m_Controls(nullptr)
, m_Manager(nullptr)
{
  bool ok = false;
  ok = connect(&m_BackgroundGrabProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundGrabProcessFinished()));
  assert(ok);
  ok = connect(&m_BackgroundCalibrateProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundCalibrateProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
CameraCalView::~CameraCalView()
{
  m_BackgroundGrabProcessWatcher.waitForFinished();
  m_BackgroundCalibrateProcessWatcher.waitForFinished();

  if (m_Controls != NULL)
  {
    ctkPluginContext* context = niftk::CameraCalViewActivator::getContext();
    assert(context);

    ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
      if (eventAdmin)
      {
        eventAdmin->unpublishSignal(this, SIGNAL(PauseIGIUpdate(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATEPAUSE");
        eventAdmin->unpublishSignal(this, SIGNAL(RestartIGIUpdate(ctkDictionary)), "uk/ac/ucl/cmic/IGIUPDATERESTART");
      }
    }

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
    m_Controls->m_ReferenceTrackerMatrixComboBox->SetPredicate(isMatrix);
    m_Controls->m_ReferenceTrackerMatrixComboBox->SetAutoSelectNewItems(false);

    m_Controls->m_LeftCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_RightCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_TrackerMatrixComboBox->SetDataStorage(dataStorage);
    m_Controls->m_ReferenceTrackerMatrixComboBox->SetDataStorage(dataStorage);

    connect(m_Controls->m_GrabButton, SIGNAL(pressed()), this, SLOT(OnGrabButtonPressed()));
    connect(m_Controls->m_UndoButton, SIGNAL(pressed()), this, SLOT(OnUnGrabButtonPressed()));
    connect(m_Controls->m_ClearButton, SIGNAL(pressed()), this, SLOT(OnClearButtonPressed()));
    connect(m_Controls->m_SaveButton, SIGNAL(pressed()), this, SLOT(OnSaveButtonPressed()));

    // Hook up combo boxes, so we know when user changes node
    connect(m_Controls->m_LeftCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_RightCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_TrackerMatrixComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_ReferenceTrackerMatrixComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));

    // Start these up as disabled, until we have enough images to calibrate.
    m_Controls->m_UndoButton->setEnabled(false);
    m_Controls->m_SaveButton->setEnabled(false);

    // Create manager, before we retrieve preferences which will populate it.
    m_Manager = niftk::NiftyCalVideoCalibrationManager::New();
    m_Manager->SetDataStorage(dataStorage);

    // Get user prefs, so we can decide if we doing chessboards/AprilTags etc.
    RetrievePreferenceValues();

    // Here, we publish signals to ask the DataSourcesManager to pause momentarily.
    ctkPluginContext* context = niftk::CameraCalViewActivator::getContext();
    assert(context);

    ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
      eventAdmin->publishSignal(this, SIGNAL(PauseIGIUpdate(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATEPAUSE", Qt::DirectConnection);
      eventAdmin->publishSignal(this, SIGNAL(RestartIGIUpdate(ctkDictionary)), "uk/ac/ucl/cmic/IGIUPDATERESTART", Qt::DirectConnection);

      ctkDictionary properties1;
      properties1[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties1);

      ctkDictionary properties2;
      properties2[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH1START";
      eventAdmin->subscribeSlot(this, SLOT(OnGrab(ctkEvent)), properties2);

      ctkDictionary properties3;
      properties3[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH2START";
      eventAdmin->subscribeSlot(this, SLOT(OnGrab(ctkEvent)), properties3);

      ctkDictionary properties4;
      properties4[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH3START";
      eventAdmin->subscribeSlot(this, SLOT(OnGrab(ctkEvent)), properties4);
    }
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::SetButtonsEnabled(bool isEnabled)
{
  m_Controls->m_LeftCameraComboBox->setEnabled(isEnabled);
  m_Controls->m_RightCameraComboBox->setEnabled(isEnabled);
  m_Controls->m_TrackerMatrixComboBox->setEnabled(isEnabled);
  m_Controls->m_ReferenceTrackerMatrixComboBox->setEnabled(isEnabled);
  m_Controls->m_GrabButton->setEnabled(isEnabled);

  m_Controls->m_UndoButton->setEnabled(m_Manager->GetNumberOfSnapshots() > 0);
  m_Controls->m_SaveButton->setEnabled(m_Manager->GetNumberOfSnapshots()
                                       >= m_Manager->GetMinimumNumberOfSnapshotsForCalibrating());
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
    m_Manager->SetMinimumNumberOfSnapshotsForCalibrating(prefs->GetInt(CameraCalViewPreferencePage::MINIMUM_VIEWS_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating));

    std::string fileName = prefs->Get(CameraCalViewPreferencePage::MODEL_NODE_NAME, "").toStdString();
    if (!fileName.empty())
    {
      m_Manager->SetModelFileName(fileName);
    }

    m_Manager->SetScaleFactorX(prefs->GetDouble(CameraCalViewPreferencePage::SCALEX_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultScaleFactorX));
    m_Manager->SetScaleFactorY(prefs->GetDouble(CameraCalViewPreferencePage::SCALEY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultScaleFactorY));

    bool doIterative = prefs->GetBool(CameraCalViewPreferencePage::DO_ITERATIVE_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultDoIterative);
    m_Manager->SetDoIterative(doIterative);

    bool do3DOptimisation = prefs->GetBool(CameraCalViewPreferencePage::DO_3D_OPTIMISATION_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultDo3DOptimisation);
    m_Manager->SetDo3DOptimisation(do3DOptimisation);

    std::string refImage = prefs->Get(CameraCalViewPreferencePage::REFERENCE_IMAGE_NODE_NAME, "").toStdString();
    std::string refPoints = prefs->Get(CameraCalViewPreferencePage::REFERENCE_POINTS_NODE_NAME, "").toStdString();

    if (!refImage.empty() && !refPoints.empty())
    {
      m_Manager->SetReferenceDataFileNames(
            refImage,
            refPoints
            );
    }

    std::string templateImage = prefs->Get(CameraCalViewPreferencePage::TEMPLATE_IMAGE_NODE_NAME, "").toStdString();
    if (!templateImage.empty())
    {
      m_Manager->SetTemplateImageFileName(templateImage);
    }

    m_Manager->SetCalibrationPattern(
          static_cast<niftk::NiftyCalVideoCalibrationManager::CalibrationPatterns>(
            prefs->GetInt(CameraCalViewPreferencePage::PATTERN_NODE_NAME, static_cast<int>(niftk::NiftyCalVideoCalibrationManager::DefaultCalibrationPattern)))
          );
    m_Manager->SetGridSizeX(prefs->GetInt(CameraCalViewPreferencePage::GRIDX_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultGridSizeX));
    m_Manager->SetGridSizeY(prefs->GetInt(CameraCalViewPreferencePage::GRIDY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultGridSizeY));
    m_Manager->SetMinimumNumberOfPoints(prefs->GetInt(CameraCalViewPreferencePage::MINIMUM_NUMBER_POINTS_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfPoints));
    m_Manager->SetTagFamily(prefs->Get(CameraCalViewPreferencePage::TAG_FAMILY_NODE_NAME, QString::fromStdString(niftk::NiftyCalVideoCalibrationManager::DefaultTagFamily)).toStdString());

    niftk::NiftyCalVideoCalibrationManager::HandEyeMethod method = static_cast<niftk::NiftyCalVideoCalibrationManager::HandEyeMethod>(
          prefs->GetInt(CameraCalViewPreferencePage::HANDEYE_NODE_NAME, static_cast<int>(niftk::NiftyCalVideoCalibrationManager::DefaultHandEyeMethod)));
    m_Manager->SetHandeyeMethod(method);

    std::string modelToTrackerFileName = prefs->Get(CameraCalViewPreferencePage::MODEL_TO_TRACKER_NODE_NAME, "").toStdString();

    if (!modelToTrackerFileName.empty())
    {
      m_Manager->SetModelToTrackerFileName(modelToTrackerFileName);
    }
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
  assert(!m_BackgroundGrabProcess.isRunning());
  assert(!m_BackgroundCalibrateProcess.isRunning());

  mitk::DataNode::Pointer leftImageNode = m_Controls->m_LeftCameraComboBox->GetSelectedNode();
  mitk::DataNode::Pointer rightImageNode = m_Controls->m_RightCameraComboBox->GetSelectedNode();
  mitk::DataNode::Pointer trackingNode = m_Controls->m_TrackerMatrixComboBox->GetSelectedNode();
  mitk::DataNode::Pointer referenceTrackingNode = m_Controls->m_ReferenceTrackerMatrixComboBox->GetSelectedNode();

  int numberOfSnapshots = m_Manager->GetNumberOfSnapshots();
  if (numberOfSnapshots > 0)
  {
    bool needsReset = false;

    mitk::DataNode::Pointer leftImageNodeInManager = m_Manager->GetLeftImageNode();
    mitk::DataNode::Pointer rightImageNodeInManager = m_Manager->GetRightImageNode();
    mitk::DataNode::Pointer trackingNodeInManager = m_Manager->GetTrackingTransformNode();
    mitk::DataNode::Pointer referenceTrackingNodeInManager = m_Manager->GetReferenceTrackingTransformNode();

    if (   leftImageNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || referenceTrackingNodeInManager.IsNotNull())
        && leftImageNode != leftImageNodeInManager)
    {
      needsReset = true;
    }

    if (   rightImageNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || referenceTrackingNodeInManager.IsNotNull())
        && rightImageNode != rightImageNodeInManager)
    {
      needsReset = true;
    }

    if (   trackingNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || referenceTrackingNodeInManager.IsNotNull())
        && trackingNode != trackingNodeInManager)
    {
      needsReset = true;
    }

    if (   referenceTrackingNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || referenceTrackingNodeInManager.IsNotNull())
        && referenceTrackingNode != referenceTrackingNodeInManager)
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
  m_Manager->SetReferenceTrackingTransformNode(referenceTrackingNode);
}


//-----------------------------------------------------------------------------
void CameraCalView::OnClearButtonPressed()
{
  m_Manager->Restart();
  m_Controls->m_ProjectionErrorValue->setText("Too few images.");
}


//-----------------------------------------------------------------------------
void CameraCalView::OnGrab(const ctkEvent& event)
{
  if (!m_BackgroundGrabProcess.isRunning() && !m_BackgroundCalibrateProcess.isRunning())
  {
    this->OnGrabButtonPressed();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnUnGrab(const ctkEvent& event)
{
  if (!m_BackgroundGrabProcess.isRunning() && !m_BackgroundCalibrateProcess.isRunning())
  {
    this->OnUnGrabButtonPressed();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnClear(const ctkEvent& event)
{
  if (!m_BackgroundGrabProcess.isRunning() && !m_BackgroundCalibrateProcess.isRunning())
  {
    this->OnClearButtonPressed();
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnUpdate(const ctkEvent& event)
{
  m_Manager->UpdateCameraToWorldPosition();
}


//-----------------------------------------------------------------------------
void CameraCalView::OnGrabButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundGrabProcess.isRunning());
  assert(!m_BackgroundCalibrateProcess.isRunning());

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

  ctkDictionary dictionary;
  emit PauseIGIUpdate(dictionary);

  this->SetButtonsEnabled(false);

  QPixmap image(":/uk.ac.ucl.cmic.igicameracal/boobaloo-Don-t-Step-No-Gnome--300px.png");
  m_Controls->m_ImageLabel->setPixmap(image);
  m_Controls->m_ImageLabel->show();

  m_BackgroundGrabProcess = QtConcurrent::run(this, &CameraCalView::RunGrab);
  m_BackgroundGrabProcessWatcher.setFuture(m_BackgroundGrabProcess);
}


//-----------------------------------------------------------------------------
bool CameraCalView::RunGrab()
{

  bool isSuccessful = false;
  std::string errorMessage = "";

  // This happens in a separate thread, so try to catch everything.
  // Even if we understand where NiftyCal exceptions come from,
  // and in our own code, where MITK exceptions come from, we probably
  // have not searched through all of OpenCV/AprilTags etc.

  try
  {
    isSuccessful = m_Manager->Grab();
  }
  catch (niftk::NiftyCalException& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunGrab() failed:" << e.GetDescription();
  }
  catch (mitk::Exception& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunGrab() failed:" << e.GetDescription();
  }
  catch (std::exception& e)
  {
    errorMessage = e.what();
    MITK_ERROR << "CameraCalView::RunGrab() failed:" << e.what();
  }

  if (!errorMessage.empty())
  {
    QMessageBox msgBox;
    msgBox.setText("An Error Occurred.");
    msgBox.setInformativeText(QString::fromStdString(errorMessage));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
  }

  return isSuccessful;
}


//-----------------------------------------------------------------------------
void CameraCalView::OnBackgroundGrabProcessFinished()
{
  ctkDictionary dictionary;
  emit RestartIGIUpdate(dictionary);

  bool successfullyGrabbed = m_BackgroundGrabProcessWatcher.result();

  if (successfullyGrabbed)
  {
    this->Calibrate();
  }
  else
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/red-cross-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
  }

  if (!m_BackgroundCalibrateProcess.isRunning())
  {
    this->SetButtonsEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::Calibrate()
{
  if (m_Manager->GetNumberOfSnapshots()
      >= m_Manager->GetMinimumNumberOfSnapshotsForCalibrating())
  {
    this->SetButtonsEnabled(false);

    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/boobaloo-Don-t-Step-No-Gnome--300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();

    m_BackgroundCalibrateProcess = QtConcurrent::run(this, &CameraCalView::RunCalibration);
    m_BackgroundCalibrateProcessWatcher.setFuture(m_BackgroundCalibrateProcess);
  }
  else
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/green-tick-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
    m_Controls->m_ProjectionErrorValue->setText("Too few images.");
  }
}


//-----------------------------------------------------------------------------
double CameraCalView::RunCalibration()
{
  double rms = 0;
  std::string errorMessage = "";

  // This happens in a separate thread, so try to catch everything.
  // Even if we understand where NiftyCal exceptions come from,
  // and in our own code, where MITK exceptions come from, we probably
  // have not searched through all of OpenCV/AprilTags etc.

  try
  {
    rms = m_Manager->Calibrate();
  }
  catch (niftk::NiftyCalException& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.GetDescription();
  }
  catch (mitk::Exception& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.GetDescription();
  }
  catch (std::exception& e)
  {
    errorMessage = e.what();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.what();
  }

  if (!errorMessage.empty())
  {
    QMessageBox msgBox;
    msgBox.setText("An Error Occurred.");
    msgBox.setInformativeText(QString::fromStdString(errorMessage));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
  }

  return rms;
}


//-----------------------------------------------------------------------------
void CameraCalView::OnBackgroundCalibrateProcessFinished()
{
  double rms = m_BackgroundCalibrateProcessWatcher.result();

  if (rms < 1)
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/green-tick-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
  }
  else
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/red-cross-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
  }

  m_Controls->m_ProjectionErrorValue->setText(tr("%1 pixels (%2 images).").arg(rms).arg(m_Manager->GetNumberOfSnapshots()));

  this->SetButtonsEnabled(true);
}


//-----------------------------------------------------------------------------
void CameraCalView::OnUnGrabButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundGrabProcess.isRunning());
  assert(!m_BackgroundCalibrateProcess.isRunning());

  m_Manager->UnGrab();
  this->Calibrate();

  if (!m_BackgroundCalibrateProcess.isRunning())
  {
    this->SetButtonsEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::OnSaveButtonPressed()
{
  // should not be able to call/click here if it's still running.
  assert(!m_BackgroundGrabProcess.isRunning());
  assert(!m_BackgroundCalibrateProcess.isRunning());

  QString dir = QFileDialog::getExistingDirectory(nullptr, tr("Output Directory"),
                                                  "",
                                                  QFileDialog::ShowDirsOnly
                                                  | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty())
  {
    m_Manager->SetOutputDirName(dir.toStdString());
    m_Manager->Save();
  }
}


} // end namespace
