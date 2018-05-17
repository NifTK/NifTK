/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCameraCalView.h"
#include "niftkCameraCalViewPreferencePage.h"
#include "niftkCameraCalViewActivator.h"
#include <niftkNiftyCalException.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkIOUtil.h>
#include <QMessageBox>
#include <QFileDialog>
#include <QPixmap>
#include <QtConcurrentRun>

#include <ctkServiceReference.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <service/event/ctkEventConstants.h>

#include <niftkCoordinateAxesData.h>
#include <niftkFileHelper.h>

namespace niftk
{

const QString CameraCalView::VIEW_ID = "uk.ac.ucl.cmic.igicameracal";

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
void CameraCalView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls = new Ui::CameraCalView();
    m_Controls->setupUi(parent);

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    m_Controls->m_LeftCameraComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_LeftCameraComboBox->SetPredicate(isImage);
    m_Controls->m_LeftCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_LeftCameraComboBox->setCurrentIndex(0);

    m_Controls->m_RightCameraComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_RightCameraComboBox->SetPredicate(isImage);
    m_Controls->m_RightCameraComboBox->SetDataStorage(dataStorage);
    m_Controls->m_RightCameraComboBox->setCurrentIndex(0);

    mitk::TNodePredicateDataType<CoordinateAxesData>::Pointer isMatrix = mitk::TNodePredicateDataType<CoordinateAxesData>::New();
    m_Controls->m_TrackerMatrixComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_TrackerMatrixComboBox->SetPredicate(isMatrix);
    m_Controls->m_TrackerMatrixComboBox->SetDataStorage(dataStorage);
    m_Controls->m_TrackerMatrixComboBox->setCurrentIndex(0);

    m_Controls->m_ModelMatrixComboBox->SetAutoSelectNewItems(false);
    m_Controls->m_ModelMatrixComboBox->SetPredicate(isMatrix);
    m_Controls->m_ModelMatrixComboBox->SetDataStorage(dataStorage);
    m_Controls->m_ModelMatrixComboBox->setCurrentIndex(0);
    m_Controls->m_ModelMatrixComboBox->setVisible(true);
    m_Controls->m_ModelMatrixLabel->setVisible(true);

    connect(m_Controls->m_GrabButton, SIGNAL(pressed()), this, SLOT(OnGrabButtonPressed()));
    connect(m_Controls->m_UndoButton, SIGNAL(pressed()), this, SLOT(OnUnGrabButtonPressed()));
    connect(m_Controls->m_ClearButton, SIGNAL(pressed()), this, SLOT(OnClearButtonPressed()));

    // Hook up combo boxes, so we know when user changes node
    connect(m_Controls->m_LeftCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_RightCameraComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_TrackerMatrixComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));
    connect(m_Controls->m_ModelMatrixComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(OnComboBoxChanged()));

    // Start these up as disabled, until we have enough images to calibrate.
    m_Controls->m_UndoButton->setEnabled(false);
    m_Controls->m_ClearButton->setEnabled(false);

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

      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);

      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH3START";
      eventAdmin->subscribeSlot(this, SLOT(OnGrab(ctkEvent)), properties);
    }
  }
}


//-----------------------------------------------------------------------------
void CameraCalView::SetButtonsEnabled(bool isEnabled)
{
  m_Controls->m_LeftCameraComboBox->setEnabled(isEnabled);
  m_Controls->m_RightCameraComboBox->setEnabled(isEnabled);
  m_Controls->m_TrackerMatrixComboBox->setEnabled(isEnabled);
  m_Controls->m_ModelMatrixComboBox->setEnabled(isEnabled);
  m_Controls->m_GrabButton->setEnabled(isEnabled);

  if (isEnabled)
  {
    m_Controls->m_ClearButton->setEnabled(m_Manager->GetNumberOfSnapshots() > 0);
    m_Controls->m_UndoButton->setEnabled(m_Manager->GetNumberOfSnapshots() > 0);
  }
  else
  {
    m_Controls->m_ClearButton->setEnabled(isEnabled);
    m_Controls->m_UndoButton->setEnabled(isEnabled);
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
    m_Manager->SetNumberOfSnapshotsForCalibrating(prefs->GetInt(CameraCalViewPreferencePage::NUMBER_VIEWS_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultNumberOfSnapshotsForCalibrating));

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

    bool modelIsStationary = prefs->GetBool(CameraCalViewPreferencePage::MODEL_IS_STATIONARY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultModelIsStationary);
    m_Manager->SetModelIsStationary(modelIsStationary);

    bool cameraIsStationary = prefs->GetBool(CameraCalViewPreferencePage::CAMERA_IS_STATIONARY_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultCameraIsStationary);
    m_Manager->SetCameraIsStationary(cameraIsStationary);

    bool doClustering = prefs->GetBool(CameraCalViewPreferencePage::DO_CLUSTERING_NODE_NAME, niftk::NiftyCalVideoCalibrationManager::DefaultDoClustering);
    m_Manager->SetDoClustering(doClustering);

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

    std::string calibrationDir = prefs->Get(CameraCalViewPreferencePage::PREVIOUS_CALIBRATION_DIR_NODE_NAME, "").toStdString();
    if (!calibrationDir.empty())
    {
      QDir dirName(QString::fromStdString(calibrationDir));
      m_Manager->LoadCalibrationFromDirectory(dirName.absolutePath().toStdString());
    }

    std::string outputDir = prefs->Get(CameraCalViewPreferencePage::OUTPUT_DIR_NODE_NAME, "").toStdString();
    if (!outputDir.empty())
    {
      m_Manager->SetOutputPrefixName(outputDir);
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

    std::string modelTransformFileName = prefs->Get(CameraCalViewPreferencePage::MODEL_TRANSFORM_NODE_NAME, "").toStdString();

    if (!modelTransformFileName.empty())
    {
      m_Manager->SetModelTransformFileName(modelTransformFileName);
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
  mitk::DataNode::Pointer modelNode = m_Controls->m_ModelMatrixComboBox->GetSelectedNode();

  int numberOfSnapshots = m_Manager->GetNumberOfSnapshots();
  if (numberOfSnapshots > 0)
  {
    bool needsReset = false;

    mitk::DataNode::Pointer leftImageNodeInManager = m_Manager->GetLeftImageNode();
    mitk::DataNode::Pointer rightImageNodeInManager = m_Manager->GetRightImageNode();
    mitk::DataNode::Pointer trackingNodeInManager = m_Manager->GetTrackingTransformNode();
    mitk::DataNode::Pointer modelNodeInManager = m_Manager->GetModelTransformNode();

    if (   leftImageNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || modelNodeInManager.IsNotNull())
        && leftImageNode != leftImageNodeInManager)
    {
      needsReset = true;
    }

    if (   rightImageNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || modelNodeInManager.IsNotNull())
        && rightImageNode != rightImageNodeInManager)
    {
      needsReset = true;
    }

    if (   trackingNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || modelNodeInManager.IsNotNull())
        && trackingNode != trackingNodeInManager)
    {
      needsReset = true;
    }

    if (   modelNode.IsNotNull()
        && (rightImageNodeInManager.IsNotNull() || leftImageNodeInManager.IsNotNull() || trackingNodeInManager.IsNotNull() || modelNodeInManager.IsNotNull())
        && modelNode != modelNodeInManager)
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
  m_Manager->SetModelTransformNode(modelNode);
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
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/green-tick-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();

    // try calibrating - might not have enough images yet.
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
  int numberForCalibrating = m_Manager->GetNumberOfSnapshotsForCalibrating();
  int numberAcquired = m_Manager->GetNumberOfSnapshots();

  if (numberAcquired == numberForCalibrating)
  {
    if (m_Manager->GetModelFileName().empty())
    {
      QMessageBox msgBox;
      msgBox.setText("An Error Occurred.");
      msgBox.setInformativeText("The model file name is empty - check preferences.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
    }
    else
    {
      this->SetButtonsEnabled(false);

      QPixmap image(":/uk.ac.ucl.cmic.igicameracal/boobaloo-Don-t-Step-No-Gnome--300px.png");
      m_Controls->m_ImageLabel->setPixmap(image);
      m_Controls->m_ImageLabel->show();

      m_BackgroundCalibrateProcess = QtConcurrent::run(this, &CameraCalView::RunCalibration);
      m_BackgroundCalibrateProcessWatcher.setFuture(m_BackgroundCalibrateProcess);
    }
  }
  else
  {
    m_Controls->m_ProjectionErrorValue->setText(QObject::tr("Too few images: (%1/%2).").arg(numberAcquired).arg(numberForCalibrating));
  }
}


//-----------------------------------------------------------------------------
std::string CameraCalView::RunCalibration()
{
  std::string outputMessage = "";
  std::string errorMessage = "";

  // This happens in a separate thread, so try to catch everything.
  // Even if we understand where NiftyCal exceptions come from,
  // and in our own code, where MITK exceptions come from, we probably
  // have not searched through all of OpenCV/AprilTags etc.

  try
  {
    outputMessage = m_Manager->Calibrate();
  }
  catch (niftk::NiftyCalException& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.GetDescription();
    throw e;
  }
  catch (mitk::Exception& e)
  {
    errorMessage = e.GetDescription();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.GetDescription();
    throw e;
  }
  catch (std::exception& e)
  {
    errorMessage = e.what();
    MITK_ERROR << "CameraCalView::RunCalibration() failed:" << e.what();
    throw e;
  }

  return outputMessage;
}


//-----------------------------------------------------------------------------
void CameraCalView::OnBackgroundCalibrateProcessFinished()
{
  if (m_BackgroundCalibrateProcess.isCanceled()
      || m_BackgroundCalibrateProcess.resultCount() == 0)
  {
    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/thumb-down-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();

    QMessageBox msgBox;
    msgBox.setText("An Error Occurred.");
    msgBox.setInformativeText("The calibration itself failed - check log.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
  }
  else
  {
    std::string calibrationMessage = m_BackgroundCalibrateProcessWatcher.result();
    m_Controls->m_ProjectionErrorValue->setText(QString::fromStdString(calibrationMessage));
    m_Manager->UpdateCameraToWorldPosition();
    m_Manager->UpdateVisualisedPoints();

    QPixmap image(":/uk.ac.ucl.cmic.igicameracal/1465762629-300px.png");
    m_Controls->m_ImageLabel->setPixmap(image);
    m_Controls->m_ImageLabel->show();
    m_Manager->Save();
  }

  m_Manager->Restart();
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

} // end namespace
