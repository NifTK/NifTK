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

const std::string SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";

//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
{
  m_SurfaceReconstruction = niftk::SurfaceReconstruction::New();
}


//-----------------------------------------------------------------------------
SurfaceReconView::~SurfaceReconView()
{
  bool ok = false;
  ok = disconnect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));
  assert(ok);
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

    if (leftNode.IsNotNull()
      && rightNode.IsNotNull()
      && leftImage.IsNotNull()
      && rightImage.IsNotNull()
      )
    {
      // if our output node exists already then we recycle it, of course.
      // it may not be tagged as "derived" from the correct source nodes
      // but that shouldn't be a problem here.

      std::string               outputName = OutputNodeNameLineEdit->text().toStdString();
      mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(outputName);
      if (outputNode.IsNull())
      {
        outputNode = mitk::DataNode::New();
        outputNode->SetName(outputName);

        mitk::DataStorage::SetOfObjects::Pointer   nodeParents = mitk::DataStorage::SetOfObjects::New();
        nodeParents->push_back(leftNode);
        nodeParents->push_back(rightNode);

        storage->Add(outputNode, nodeParents);
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
        // Then delagate everything to class outside of plugin, so we can unit test it.
        m_SurfaceReconstruction->Run(storage, outputNode, leftImage, rightImage, method, outputtype, camNode, maxTriError, minDepth, maxDepth);
      }
      catch (const std::exception& e)
      {
        std::cerr << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;
        // FIXME: show an error message on the plugin panel somewhere?
      }
    }
  }
}
