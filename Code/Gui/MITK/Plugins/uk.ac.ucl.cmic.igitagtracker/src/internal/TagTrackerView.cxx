/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TagTrackerView.h"
#include "TagTrackerViewActivator.h"
#include "TagTrackerViewPreferencePage.h"
#include <QFile>
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesData.h>
#include <mitkMonoTagExtractor.h>
#include <mitkStereoTagExtractor.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <Undistortion.h>
#include <SurfaceReconstruction.h>

const std::string TagTrackerView::VIEW_ID = "uk.ac.ucl.cmic.igitagtracker";
const std::string TagTrackerView::NODE_ID = "Tag Locations";

//-----------------------------------------------------------------------------
TagTrackerView::TagTrackerView()
: m_ListenToEventBusPulse(true)
, m_MonoLeftCameraOnly(false)
, m_MinSize(0.01)
, m_MaxSize(0.0125)
{
}


//-----------------------------------------------------------------------------
TagTrackerView::~TagTrackerView()
{
}


//-----------------------------------------------------------------------------
std::string TagTrackerView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void TagTrackerView::CreateQtPartControl( QWidget *parent )
{
  setupUi(parent);

  bool ok = false;
  ok = connect(m_UpdateButton, SIGNAL(pressed()), this, SLOT(OnManualUpdate()));
  assert(ok);

  ctkServiceReference ref = mitk::TagTrackerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::TagTrackerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }

  this->RetrievePreferenceValues();

  m_StereoImageAndCameraSelectionWidget->SetDataStorage(this->GetDataStorage());
  m_StereoImageAndCameraSelectionWidget->UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TagTrackerView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    m_ListenToEventBusPulse = prefs->GetBool(TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS_NAME, TagTrackerViewPreferencePage::LISTEN_TO_EVENT_BUS);
    m_MinSize = static_cast<float>(prefs->GetDouble(TagTrackerViewPreferencePage::MIN_SIZE_NAME, TagTrackerViewPreferencePage::MIN_SIZE));
    m_MaxSize = static_cast<float>(prefs->GetDouble(TagTrackerViewPreferencePage::MAX_SIZE_NAME, TagTrackerViewPreferencePage::MAX_SIZE));
  }

  if (m_ListenToEventBusPulse)
  {
    m_UpdateButton->setEnabled(false);
  }
  else
  {
    m_UpdateButton->setEnabled(true);
  }
  m_MonoLeftCameraOnly = prefs->GetBool(TagTrackerViewPreferencePage::DO_MONO_LEFT_CAMERA_NAME, TagTrackerViewPreferencePage::DO_MONO_LEFT_CAMERA);
  m_StereoImageAndCameraSelectionWidget->SetLeftChannelEnabled(true);
  m_StereoImageAndCameraSelectionWidget->SetRightChannelEnabled(!m_MonoLeftCameraOnly);
  m_StereoCameraCalibrationSelectionWidget->SetLeftChannelEnabled(!m_MonoLeftCameraOnly);
  m_StereoCameraCalibrationSelectionWidget->SetRightChannelEnabled(!m_MonoLeftCameraOnly);
}


//-----------------------------------------------------------------------------
void TagTrackerView::SetFocus()
{
  m_StereoImageAndCameraSelectionWidget->setFocus();
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnUpdate(const ctkEvent& event)
{
  if (m_ListenToEventBusPulse)
  {
    this->UpdateTags();
  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnManualUpdate()
{
  this->UpdateTags();
}


//-----------------------------------------------------------------------------
void TagTrackerView::UpdateTags()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  assert(dataStorage);

  mitk::Image::Pointer leftImage = m_StereoImageAndCameraSelectionWidget->GetLeftImage();
  mitk::Image::Pointer rightImage = m_StereoImageAndCameraSelectionWidget->GetRightImage();
  mitk::DataNode::Pointer leftNode = m_StereoImageAndCameraSelectionWidget->GetLeftNode();
  mitk::DataNode::Pointer rightNode = m_StereoImageAndCameraSelectionWidget->GetRightNode();
  QString leftIntrinsicFileName = m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName();
  QString rightIntrinsicFileName = m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName();
  QString leftToRightTransformationFileName = m_StereoCameraCalibrationSelectionWidget->GetLeftToRightTransformationFileName();

  if (leftNode.IsNotNull() || rightNode.IsNotNull())
  {
    bool needToLoadLeftCalib  = false;
    if (leftImage.IsNotNull())
    {
      needToLoadLeftCalib = niftk::Undistortion::NeedsToLoadCalib(leftIntrinsicFileName.toStdString(),  leftImage);
    }

    bool needToLoadRightCalib = false;
    bool needToLoadLeftToRightTransformation = false;
    if (rightImage.IsNotNull())
    {
      needToLoadRightCalib = niftk::Undistortion::NeedsToLoadCalib(rightIntrinsicFileName.toStdString(), rightImage);
      needToLoadLeftToRightTransformation = niftk::Undistortion::NeedsToLoadStereoRigExtrinsics(leftToRightTransformationFileName.toStdString(), rightImage);
    }

    if (needToLoadLeftCalib)
    {
      niftk::Undistortion::LoadCalibration(m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName().toStdString(), leftImage);
    }
    if (needToLoadRightCalib)
    {
      niftk::Undistortion::LoadCalibration(m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName().toStdString(), rightImage);
    }
    if (needToLoadLeftToRightTransformation)
    {
      niftk::Undistortion::LoadStereoRig(
          m_StereoCameraCalibrationSelectionWidget->GetLeftToRightTransformationFileName().toStdString(),
          niftk::Undistortion::s_StereoRigTransformationPropertyName,
          rightImage);
    }

    // Retrieve the point set node from data storage, or create it if it does not exist.
    mitk::PointSet::Pointer pointSet;
    mitk::DataNode::Pointer pointSetNode = dataStorage->GetNamedNode(NODE_ID);

    if (pointSetNode.IsNull())
    {
      pointSet = mitk::PointSet::New();
      pointSetNode = mitk::DataNode::New();
      pointSetNode->SetData( pointSet );
      pointSetNode->SetProperty( "name", mitk::StringProperty::New(NODE_ID));
      pointSetNode->SetProperty( "opacity", mitk::FloatProperty::New(1));
      pointSetNode->SetProperty( "point line width", mitk::IntProperty::New(1));
      pointSetNode->SetProperty( "point 2D size", mitk::IntProperty::New(5));
      pointSetNode->SetVisibility(true);
      pointSetNode->SetBoolProperty("helper object", false);
      pointSetNode->SetBoolProperty("show distant lines", false);
      pointSetNode->SetBoolProperty("show distant points", false);
      pointSetNode->SetBoolProperty("show distances", false);
      pointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
      pointSetNode->SetColor( 1.0, 0, 0 );
      dataStorage->Add(pointSetNode);
    }
    else
    {
      pointSet = dynamic_cast<mitk::PointSet*>(pointSetNode->GetData());
      if (pointSet.IsNull())
      {
        // Give up, as the node has the wrong data.
        MITK_ERROR << "TagTrackerView::OnUpdate node " << NODE_ID << " does not contain an mitk::PointSet" << std::endl;
        return;
      }
    }

    // Extract camera to world matrix to pass onto either mono or stereo.
    vtkSmartPointer<vtkMatrix4x4> cameraToWorldMatrix = vtkMatrix4x4::New();
    cameraToWorldMatrix->Identity();

    mitk::CoordinateAxesData::Pointer cameraToWorld = m_StereoImageAndCameraSelectionWidget->GetCameraTransform();
    if (cameraToWorld.IsNotNull())
    {
      cameraToWorld->GetVtkMatrix(*cameraToWorldMatrix);
    }

    // Now use the data to extract points, and update the point set.
    QString modeName;

    if ((leftNode.IsNotNull() && rightNode.IsNull())
        || (leftNode.IsNull() && rightNode.IsNotNull())
        || m_MonoLeftCameraOnly
        )
    {
      mitk::Image::Pointer image;

      if (leftNode.IsNotNull())
      {
        image = dynamic_cast<mitk::Image*>(leftNode->GetData());
        modeName = "left";
      }
      else
      {
        image = dynamic_cast<mitk::Image*>(rightNode->GetData());
        modeName = "right";
      }

      if (image.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, mono case, image node is NULL" << std::endl;
        return;
      }

      // Mono Case.
      mitk::MonoTagExtractor::Pointer extractor = mitk::MonoTagExtractor::New();
      extractor->ExtractPoints(
          image,
          m_MinSize,
          m_MaxSize,
          pointSet,
          cameraToWorldMatrix
          );
    }
    else
    {
      modeName = "stereo";

      if (leftImage.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, left image is NULL" << std::endl;
        return;
      }
      if (rightImage.IsNull())
      {
        MITK_ERROR << "TagTrackerView::OnUpdate, stereo case, right image is NULL" << std::endl;
        return;
      }

      // Stereo Case.
      mitk::StereoTagExtractor::Pointer extractor = mitk::StereoTagExtractor::New();
      extractor->ExtractPoints(
          leftImage,
          rightImage,
          m_MinSize,
          m_MaxSize,
          pointSet,
          cameraToWorldMatrix
          );
    } // end if mono/stereo

    int numberOfTrackedPoints = pointSet->GetSize();

    QString numberString;
    numberString.setNum(numberOfTrackedPoints);

    m_NumberOfTagsLabel->setText(modeName + QString(" tags ") + numberString);
    m_TagPositionDisplay->clear();

    if (numberOfTrackedPoints > 0)
    {
      mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet(0);
      mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
      mitk::PointSet::PointsIterator pIt;
      mitk::PointSet::PointIdentifier pointID;
      mitk::PointSet::PointType point;

      for (pIt = points->Begin(); pIt != points->End(); ++pIt)
      {
        pointID = pIt->Index();
        point = pIt->Value();

        QString pointIdString;
        pointIdString.setNum(pointID);
        QString xNum;
        xNum.setNum(point[0]);
        QString yNum;
        yNum.setNum(point[1]);
        QString zNum;
        zNum.setNum(point[2]);

        m_TagPositionDisplay->appendPlainText(QString("point [") + pointIdString + "]=(" + xNum + ", " + yNum + ", " + zNum + ")");
      }
    }
  } // end if we have at least one node specified
}
