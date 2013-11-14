/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
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
#include <QFileDialog>
#include <mitkCoordinateAxesData.h>
#include "SurfaceReconViewPreferencePage.h"
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <QtConcurrentRun>
#include <boost/bind.hpp>


const std::string SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";


//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
  : m_BackgroundOutputNodeIsVisible(true)
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

  // wait for it to finish first and then disconnect?
  // or the other way around?
  // i'd say disconnect first then wait because at that time we no longer care about the result
  // and the finished-handler might access some half-destroyed objects.
  ok = disconnect(&m_BackgroundProcessWatcher, SIGNAL(finished()), this, SLOT(OnBackgroundProcessFinished()));
  assert(ok);
  m_BackgroundProcessWatcher.waitForFinished();
}


//-----------------------------------------------------------------------------
std::string SurfaceReconView::GetViewID() const
{
  return VIEW_ID;
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
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }

  this->RetrievePreferenceValues();

  m_StereoImageAndCameraSelectionWidget->SetDataStorage(this->GetDataStorage());
  m_StereoImageAndCameraSelectionWidget->UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService = berry::Platform::GetServiceRegistry().GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);
  berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(SurfaceReconViewPreferencePage::s_PrefsNodeName)).Cast<berry::IBerryPreferences>();
  assert(prefs);

  m_MaxTriangulationErrorThresholdSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultTriangulationErrorPrefsName, 0.1f));

  m_MinDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMinDepthRangePrefsName, 1.0f));
  m_MaxDepthRangeSpinBox->setValue(prefs->GetFloat(SurfaceReconViewPreferencePage::s_DefaultMaxDepthRangePrefsName, 1000.0f));

  bool  useUndistortDefaultPath = prefs->GetBool(SurfaceReconViewPreferencePage::s_UseUndistortionDefaultPathPrefsName, true);
  if (useUndistortDefaultPath)
  {
    // FIXME: hard-coded prefs node names, etc.
    //        how to access header files in another plugin?
    //        see https://cmicdev.cs.ucl.ac.uk/trac/ticket/2505
    berry::IBerryPreferences::Pointer undistortPrefs = (prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.igiundistort")).Cast<berry::IBerryPreferences>();
    if (undistortPrefs.IsNotNull())
    {
      QString lastDirectory = QString::fromStdString(undistortPrefs->Get("default calib file path", ""));
      this->m_StereoCameraCalibrationSelectionWidget->SetLastDirectory(lastDirectory);
    }
  }
  else
  {
    QString lastDirectory = QString::fromStdString(prefs->Get(SurfaceReconViewPreferencePage::s_DefaultCalibrationFilePathPrefsName, ""));
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
void SurfaceReconView::DoSurfaceReconstruction()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    mitk::Image::Pointer leftImage = m_StereoImageAndCameraSelectionWidget->GetLeftImage();
    mitk::Image::Pointer rightImage = m_StereoImageAndCameraSelectionWidget->GetRightImage();
    mitk::DataNode::Pointer leftNode = m_StereoImageAndCameraSelectionWidget->GetLeftNode();
    mitk::DataNode::Pointer rightNode = m_StereoImageAndCameraSelectionWidget->GetRightNode();

    // we store these background processing.
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

      niftk::SurfaceReconstruction::OutputType  outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
      if (GenerateDisparityImageRadioBox->isChecked())
      {
        assert(!GeneratePointCloudRadioBox->isChecked());
        outputtype = niftk::SurfaceReconstruction::DISPARITY_IMAGE;
      }
      if (GeneratePointCloudRadioBox->isChecked())
      {
        assert(!GenerateDisparityImageRadioBox->isChecked());
        outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
      }

      // where to place the point cloud in 3d space
      // is ok if node doesnt exist, SurfaceReconstruction will deal with that.
      mitk::DataNode::Pointer camNode = m_StereoImageAndCameraSelectionWidget->GetCameraNode();

      niftk::SurfaceReconstruction::Method  method = (niftk::SurfaceReconstruction::Method) MethodComboBox->currentIndex();
      float maxTriError = (float) m_MaxTriangulationErrorThresholdSpinBox->value();
      float minDepth    = (float) m_MinDepthRangeSpinBox->value();
      float maxDepth    = (float) m_MaxDepthRangeSpinBox->value();

      try
      {
        // dont allow clicking on it until we are done with the current one.
        DoItButton->setEnabled(false);

        // have a parameters packet because QtConcurrent::run() does not cater for functions with lots of parameters.
        niftk::SurfaceReconstruction::ParamPacket   params;
        params.image1 = leftImage;
        params.image2 = rightImage;
        params.method = method;
        params.outputtype = outputtype;
        params.camnode = camNode;
        params.maxTriangulationError = maxTriError;
        params.minDepth = minDepth;
        params.maxDepth = maxDepth;

        m_BackgroundProcess = QtConcurrent::run(this, &SurfaceReconView::RunBackgroundReconstruction, params);
        m_BackgroundProcessWatcher.setFuture(m_BackgroundProcess);
      }
      catch (const std::exception& e)
      {
        std::cerr << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;
        // FIXME: show an error message on the plugin panel somewhere?
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
    std::cerr << "Caught exception: " << e.what() << std::endl;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception!" << std::endl;
  }
  return result;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnBackgroundProcessFinished()
{
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
    std::cerr << "Warning: data storage gone while processing surface reconstruction!" << std::endl;
  }
}

