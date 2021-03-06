/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SurfaceReconView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "SurfaceReconViewActivator.h"
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkNodePredicateDataType.h>
#include <mitkLogMacros.h>
#include <QFileDialog>
#include <QDateTime>
#include <QMessageBox>
#include <niftkCoordinateAxesData.h>
#include "SurfaceReconViewPreferencePage.h"
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryPlatform.h>
#include <QtConcurrentRun>
#include <cctype>


const QString SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";


//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
  : m_BackgroundOutputNodeIsVisible(true)
  , m_IGIUpdateSubscriptionID(-1)
  , m_IGIRecordingStartedSubscriptionID(-1)
{
  m_SurfaceReconstruction = niftk::SurfaceReconstruction::New();

  bool ok = false;
  ok = connect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
}


//-----------------------------------------------------------------------------
SurfaceReconView::~SurfaceReconView()
{
  bool ok = false;
  ok = disconnect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));
  assert(ok);


  // ctk event bus de-registration
  {
    ctkServiceReference ref = mitk::SurfaceReconViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::SurfaceReconViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      if (eventAdmin)
      {
        eventAdmin->unsubscribeSlot(m_IGIUpdateSubscriptionID);
        eventAdmin->unsubscribeSlot(m_IGIRecordingStartedSubscriptionID);
        eventAdmin->unsubscribeSlot(m_FootswitchSubscriptionID);
      }
    }
  }

  // wait for it to finish first and then disconnect?
  // or the other way around?
  // i'd say disconnect first then wait because at that time we no longer care about the result
  // and the finished-handler might access some half-destroyed objects.
  ok = disconnect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
  m_BackgroundProcessWatcher.waitForFinished();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CreateQtPartControl( QWidget *parent )
{
  setupUi(parent);

  bool ok = false;
  ok = connect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));
  assert(ok);

  ctkServiceReference ref = mitk::SurfaceReconViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::SurfaceReconViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    m_IGIUpdateSubscriptionID = eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);

    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTARTED";
    m_IGIRecordingStartedSubscriptionID = eventAdmin->subscribeSlot(this, SLOT(OnRecordingStarted(ctkEvent)), properties);

    ctkDictionary footswitchProperties;
    footswitchProperties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH2START";
    m_FootswitchSubscriptionID = eventAdmin->subscribeSlot(this, SLOT(OnFootSwitchPressed(ctkEvent)), footswitchProperties);
  }

  this->RetrievePreferenceValues();

  m_StereoImageAndCameraSelectionWidget->SetDataStorage(this->GetDataStorage());
  m_StereoImageAndCameraSelectionWidget->UpdateNodeNameComboBox();

  // populate the method combobox with existing methods.
  MethodComboBox->clear();
  for (int i = 0; ; ++i)
  {
    niftk::SurfaceReconstruction::Method  methodid;
    std::string                           name;

    bool  methodexists = niftk::SurfaceReconstruction::GetMethodDetails(i, &methodid, &name);
    if (!methodexists)
      break;

    MethodComboBox->addItem(QString::fromStdString(name));
  }
  // enable box only if there is any method.
  MethodComboBox->setEnabled(MethodComboBox->count() > 0);

#ifndef _USE_PCL
  m_GeneratePCLPointCloudRadioBox->setEnabled(false);
#endif
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  berry::IPreferences::Pointer prefs = prefService->GetSystemPreferences()->Node(VIEW_ID);
  assert(prefs);

  m_MaxTriangulationErrorThresholdSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultTriangulationErrorPrefsName, 0.1f));

  m_MinDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMinDepthRangePrefsName, 1.0f));
  m_MaxDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMaxDepthRangePrefsName, 1000.0f));
  m_BakeWorldTransformCheckBox->setChecked(prefs->GetBool(SurfaceReconViewPreferencePage::s_DefaultBakeCameraTransformPrefsName, true));

  bool  useUndistortDefaultPath = prefs->GetBool(SurfaceReconViewPreferencePage::s_UseUndistortionDefaultPathPrefsName, true);
  if (useUndistortDefaultPath)
  {
    // FIXME: hard-coded prefs node names, etc.
    //        how to access header files in another plugin?
    //        see https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/2505
    berry::IPreferences::Pointer undistortPrefs = prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.igiundistort");
    if (undistortPrefs.IsNotNull())
    {
      QString lastDirectory = undistortPrefs->Get("default calib file path", "");
      this->m_StereoCameraCalibrationSelectionWidget->SetLastDirectory(lastDirectory);
    }
  }
  else
  {
    QString lastDirectory = prefs->Get(SurfaceReconViewPreferencePage::s_DefaultCalibrationFilePathPrefsName, "");
    this->m_StereoCameraCalibrationSelectionWidget->SetLastDirectory(lastDirectory);
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnUpdate(const ctkEvent& event)
{
  // Optional. This gets called everytime the data sources are updated.
  // If the surface reconstruction was as fast as the GUI update, we could trigger it here.

  // we call this all the time to update the has-calib-property for the node comboboxes.
  m_StereoImageAndCameraSelectionWidget->UpdateNodeNameComboBox();

  if (m_AutomaticUpdateRadioButton->isChecked())
  {
    if (!m_BackgroundProcess.isRunning())
    {
      DoSurfaceReconstruction();
    }
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnFootSwitchPressed(const ctkEvent& event)
{
  if (!m_BackgroundProcess.isRunning())
  {
    DoSurfaceReconstruction();
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::WriteCurrentConfig(const QString& directory) const
{
  QFile   infoFile(directory + QDir::separator() + VIEW_ID + ".txt");
  bool opened = infoFile.open(QIODevice::ReadWrite | QIODevice::Text | QIODevice::Append);
  if (opened)
  {
    QTextStream   info(&infoFile);
    info.setCodec("UTF-8");
    info << "START: " << QDateTime::currentDateTime().toString() << "\n";

    mitk::DataNode::Pointer leftNode = m_StereoImageAndCameraSelectionWidget->GetLeftNode();
    mitk::DataNode::Pointer rightNode = m_StereoImageAndCameraSelectionWidget->GetRightNode();

    info << "leftnode=" << (leftNode.IsNotNull() ? QString::fromStdString("\"" + leftNode->GetName() + "\"") : "null") << "\n";
    info << "rightnode=" << (rightNode.IsNotNull() ? QString::fromStdString("\"" + rightNode->GetName() + "\"") : "null") << "\n";

    info << "leftcalibfile=" << m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName() << "\n";
    info << "rightcalibfile=" << m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName() << "\n";
    info << "stereorigfile=" << m_StereoCameraCalibrationSelectionWidget->GetLeftToRightTransformationFileName() << "\n";

    info << "outputnodeisvisible=" << (OutputNodeIsVisibleCheckBox->isChecked() ? "yes" : "no") << "\n";
    info << "outputnode=" << OutputNodeNameLineEdit->text() << "\n";
    info << "autoincoutputnodename=" << (AutoIncNodeNameCheckBox->isChecked() ? "yes" : "no") << "\n";
    info << "outputtype=" << (GenerateDisparityImageRadioBox->isChecked() ? "disparityimage"
                                : (m_GeneratePCLPointCloudRadioBox->isChecked() ? "pclpointcloud" : "mitkpointcloud")) << "\n";

    mitk::DataNode::Pointer camNode = m_StereoImageAndCameraSelectionWidget->GetCameraNode();
    info << "cameranode=" << (camNode.IsNotNull() ? QString::fromStdString("\"" + camNode->GetName() + "\"") : "null") << "\n";

    info << "methodname=" << MethodComboBox->currentText() << "\n";
    info << "maxtrierror=" << m_MaxTriangulationErrorThresholdSpinBox->value() << "\n";
    info << "mindepth=" << m_MinDepthRangeSpinBox->value() << "\n";
    info << "maxdepth=" << m_MaxDepthRangeSpinBox->value() << "\n";

    info << "bakingcameratoworldtransform=" << (m_BakeWorldTransformCheckBox->isChecked() ? "yes" : "no") << "\n";
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnRecordingStarted(const ctkEvent& event)
{
  QString   directory = event.getProperty("directory").toString();
  if (!directory.isEmpty())
  {
    try
    {
      WriteCurrentConfig(directory);
    }
    catch (...)
    {
      MITK_ERROR << "Caught exception while writing info file! Ignoring it and aborting info file.";
    }
  }
  else
  {
    MITK_WARN << "Received igi-recording-started event without directory information! Ignoring it.";
  }
}


//-----------------------------------------------------------------------------
std::string SurfaceReconView::IncrementNodeName(const std::string& name)
{
  // note: we do not trim white-space!
  // this is intentional: it allows the user to add a second dimension of numbers to it.
  // for example: 
  //   name="hello world"    --> "hello world 1"
  //   name="hello world 1"  --> "hello world 2"
  //   name "hello world 1 " --> "hello world 1 1"


  // scan from the back of the name until we find a character that is not a number.
  int    numberstartindex = name.length();
  for (; numberstartindex > 0; --numberstartindex)
  {
    if (!std::isdigit(name[numberstartindex - 1]))
      break;
  }
  assert(numberstartindex >= 0);

  // no number in the name
  if (numberstartindex >= name.length())
  {
    return name + "1";
  }

  std::string   numbersubstring = name.substr(numberstartindex);
  // if we get here we expect there to be a number!
  assert(!numbersubstring.empty());
  int           number = atoi(numbersubstring.c_str());

  // might be empty if there is nothing but number.
  std::string   basename = name.substr(0, numberstartindex);

  std::ostringstream  newname;
  // we preserved white-space so we dont need to add more.
  newname << basename << (number + 1);

  return newname.str();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::DoSurfaceReconstruction()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    mitk::Image::Pointer leftImage = m_StereoImageAndCameraSelectionWidget->GetLeftImage();
    mitk::Image::Pointer rightImage = m_StereoImageAndCameraSelectionWidget->GetRightImage();
    mitk::DataNode::Pointer leftNode = m_StereoImageAndCameraSelectionWidget->GetLeftNode();
    mitk::DataNode::Pointer rightNode = m_StereoImageAndCameraSelectionWidget->GetRightNode();

    mitk::Image::Pointer leftMask = m_StereoImageAndCameraSelectionWidget->GetLeftMask();
    mitk::Image::Pointer rightMask = m_StereoImageAndCameraSelectionWidget->GetRightMask();

    // we store these for background processing.
    // we keep names instead of pointers because data storage screws up if we add the output node
    // with parents that might have been deleted already.
    m_BackgroundLeftNodeName = "";
    m_BackgroundRightNodeName = "";
    if (leftNode.IsNotNull())
    {
      m_BackgroundLeftNodeName = leftNode->GetName();
    }
    if (rightNode.IsNotNull())
    {
      m_BackgroundRightNodeName = rightNode->GetName();
    }

    // mark output node invisible to avoid trashing the renderer with millions of points.
    m_BackgroundOutputNodeIsVisible = OutputNodeIsVisibleCheckBox->isChecked();

    if (leftNode.IsNotNull()
      && rightNode.IsNotNull()
      && leftImage.IsNotNull()
      && rightImage.IsNotNull()
      )
    {
      // store the name for use once processing has finished in OnBackgroundProcessFinished().
      m_BackgroundOutputNodeName = OutputNodeNameLineEdit->text().toStdString();
      if (AutoIncNodeNameCheckBox->isChecked())
      {
        m_BackgroundOutputNodeName = IncrementNodeName(m_BackgroundOutputNodeName);
        // update the gui so the user notices what is happening.
        OutputNodeNameLineEdit->setText(QString::fromStdString(m_BackgroundOutputNodeName));
      }

      bool    needToLoadLeftCalib  = niftk::Undistortion::NeedsToLoadIntrinsicCalib(m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName().toStdString(),  leftImage);
      bool    needToLoadRightCalib = niftk::Undistortion::NeedsToLoadIntrinsicCalib(m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName().toStdString(), rightImage);

      if (needToLoadLeftCalib)
      {
        niftk::Undistortion::LoadIntrinsicCalibration(m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName().toStdString(), leftImage);
      }
      if (needToLoadRightCalib)
      {
        niftk::Undistortion::LoadIntrinsicCalibration(m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName().toStdString(), rightImage);
      }
      niftk::Undistortion::LoadStereoRig(
          m_StereoCameraCalibrationSelectionWidget->GetLeftToRightTransformationFileName().toStdString(),
          rightImage);

      niftk::Undistortion::CopyImagePropsIfNecessary(leftNode,  leftImage);
      niftk::Undistortion::CopyImagePropsIfNecessary(rightNode, rightImage);

      niftk::SurfaceReconstruction::OutputType  outputtype = niftk::SurfaceReconstruction::PCL_POINT_CLOUD;
      if (GenerateDisparityImageRadioBox->isChecked())
      {
        assert(!m_GenerateMITKPointCloudRadioBox->isChecked());
        assert(!m_GeneratePCLPointCloudRadioBox->isChecked());
        outputtype = niftk::SurfaceReconstruction::DISPARITY_IMAGE;
      }
      if (m_GenerateMITKPointCloudRadioBox->isChecked())
      {
        assert(!GenerateDisparityImageRadioBox->isChecked());
        assert(!m_GeneratePCLPointCloudRadioBox->isChecked());
        outputtype = niftk::SurfaceReconstruction::MITK_POINT_CLOUD;
      }
#ifdef _USE_PCL
      if (m_GeneratePCLPointCloudRadioBox->isChecked())
      {
        assert(!GenerateDisparityImageRadioBox->isChecked());
        assert(!m_GenerateMITKPointCloudRadioBox->isChecked());
        outputtype = niftk::SurfaceReconstruction::PCL_POINT_CLOUD;
      }
#endif

      // where to place the point cloud in 3d space
      // is ok if node doesnt exist, SurfaceReconstruction will deal with that.
      mitk::DataNode::Pointer camNode = m_StereoImageAndCameraSelectionWidget->GetCameraNode();

      QString   methodname = MethodComboBox->currentText();
      niftk::SurfaceReconstruction::Method  method = niftk::SurfaceReconstruction::ParseMethodName(methodname.toStdString());

      float maxTriError = (float) m_MaxTriangulationErrorThresholdSpinBox->value();
      float minDepth    = (float) m_MinDepthRangeSpinBox->value();
      float maxDepth    = (float) m_MaxDepthRangeSpinBox->value();
      bool  bakeTransform = m_BakeWorldTransformCheckBox->isChecked();

      try
      {
        // dont allow clicking on it until we are done with the current one.
        DoItButton->setEnabled(false);

        // have a parameters packet because QtConcurrent::run() does not cater for functions with lots of parameters.
        niftk::SurfaceReconstruction::ParamPacket   params;
        params.image1 = leftImage;
        params.image2 = rightImage;
        params.mask1 = leftMask;
        params.mask2 = rightMask;
        params.method = method;
        params.outputtype = outputtype;
        params.camnode = camNode;
        params.maxTriangulationError = maxTriError;
        params.minDepth = minDepth;
        params.maxDepth = maxDepth;
        params.bakeCameraTransform = bakeTransform;

        // make sure to reset this before we start processing.
        m_BackgroundErrorMessage = "";

        m_BackgroundProcess = QtConcurrent::run(this, &SurfaceReconView::RunBackgroundReconstruction, params);
        m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
      }
      catch (const std::exception& e)
      {
        // i dont think this will ever catch here. it's a remnant of when surface-recon was called directly,
        // instead of bouncing it through a future.
        MITK_ERROR << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;

        QMessageBox msgBox;
        msgBox.setText("Starting surface reconstruction failed.");
        msgBox.setInformativeText(QString::fromStdString(e.what()));
        msgBox.setStandardButtons(QMessageBox::Ok);
        msgBox.setDefaultButton(QMessageBox::Ok);
        msgBox.exec();
      }
    }
  }
}


//-----------------------------------------------------------------------------
mitk::BaseData::Pointer SurfaceReconView::RunBackgroundReconstruction(niftk::SurfaceReconstruction::ParamPacket param)
{
  mitk::BaseData::Pointer   result;
  try
  {
    result = m_SurfaceReconstruction->Run(param);
  }
  catch (const std::exception& e)
  {
    MITK_ERROR << "Caught exception: " << e.what() << std::endl;
    m_BackgroundErrorMessage = e.what();
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception!" << std::endl;
    m_BackgroundErrorMessage = "unknown exception";
  }
  return result;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnBackgroundProcessFinished()
{
  if (!m_BackgroundErrorMessage.empty())
  {
    MITK_ERROR << "Background processing returned an error message: " << m_BackgroundErrorMessage;

    QMessageBox msgBox;
    msgBox.setText("Surface Reconstruction failed.");
    msgBox.setInformativeText(QString::fromStdString(m_BackgroundErrorMessage));
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();

    DoItButton->setEnabled(true);

    // dont think we should continue, trying to parse the output from our future.
    return;
  }

  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    mitk::BaseData::Pointer data = m_BackgroundProcessWatcher.result();

    // if our output node exists already then we recycle it, of course.
    // it may not be tagged as "derived" from the correct source nodes
    // but that shouldn't be a problem here.
    mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(m_BackgroundOutputNodeName);
    if (outputNode.IsNull())
    {
      outputNode = mitk::DataNode::New();
      outputNode->SetName(m_BackgroundOutputNodeName);

      mitk::DataStorage::SetOfObjects::Pointer   nodeParents = mitk::DataStorage::SetOfObjects::New();
      if (!m_BackgroundLeftNodeName.empty())
      {
        mitk::DataNode::Pointer   leftNode = storage->GetNamedNode(m_BackgroundLeftNodeName);
        if (leftNode.IsNotNull())
        {
          nodeParents->push_back(leftNode);
        }
      }
      if (!m_BackgroundRightNodeName.empty())
      {
        mitk::DataNode::Pointer   rightNode = storage->GetNamedNode(m_BackgroundRightNodeName);
        if (rightNode.IsNotNull())
        {
          nodeParents->push_back(rightNode);
        }
      }

      // we need to have data on the node before we add it to data storage!
      // otherwise listeners on it are not fired properly. (bug in mitk?)
      outputNode->SetData(data);
      storage->Add(outputNode, nodeParents);
    }
    else
    {
      outputNode->SetData(data);
    }

    outputNode->SetVisibility(m_BackgroundOutputNodeIsVisible);

    DoItButton->setEnabled(true);
  }
  else
  {
    // i think this else branch will only ever happen if we are half-way down the shutdown process.
    // but not sure...
    MITK_ERROR << "Warning: data storage gone while processing surface reconstruction!";
  }
}

