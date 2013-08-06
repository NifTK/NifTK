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
#include <QMessageBox>
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <mitkSurface.h>
#include <mitkCoordinateAxesData.h>
#include <mitkMonoTagExtractor.h>
#include <mitkStereoTagExtractor.h>
#include <mitkNodePredicateDataType.h>
#include <mitkPointBasedRegistration.h>
#include <mitkPointsAndNormalsBasedRegistration.h>
#include <mitkCoordinateAxesData.h>
#include <mitkNodePredicateOr.h>
#include <Undistortion.h>
#include <SurfaceReconstruction.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

const std::string TagTrackerView::VIEW_ID = "uk.ac.ucl.cmic.igitagtracker";

//-----------------------------------------------------------------------------
TagTrackerView::TagTrackerView()
: m_ListenToEventBusPulse(true)
, m_MonoLeftCameraOnly(false)
, m_ShownStereoSameNameWarning(false)
{
  m_RangesOfRotationalParams[0] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[1] = std::numeric_limits<double>::min();
  m_RangesOfRotationalParams[2] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[3] = std::numeric_limits<double>::min();
  m_RangesOfRotationalParams[4] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[5] = std::numeric_limits<double>::min();
}


//-----------------------------------------------------------------------------
TagTrackerView::~TagTrackerView()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  mitk::DataNode::Pointer dataNode = dataStorage->GetNamedNode(mitk::TagTrackingRegistrationManager::POINTSET_NODE_ID);
  if (dataNode.IsNotNull())
  {
    dataStorage->Remove(dataNode);
  }

  dataNode = dataStorage->GetNamedNode(mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID);
  if (dataNode.IsNotNull())
  {
    dataStorage->Remove(dataNode);
  }

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

  m_BlockSizeSpinBox->setMinimum(3);
  m_BlockSizeSpinBox->setMaximum(50);
  m_BlockSizeSpinBox->setValue(20);
  m_OffsetSpinBox->setMinimum(-50);
  m_OffsetSpinBox->setMaximum(50);
  m_OffsetSpinBox->setValue(10);
  m_MinSizeSpinBox->setMinimum(0.001);
  m_MinSizeSpinBox->setMaximum(0.999);
  m_MinSizeSpinBox->setValue(0.005);
  m_MaxSizeSpinBox->setMinimum(0.001);
  m_MaxSizeSpinBox->setMaximum(0.999);
  m_MaxSizeSpinBox->setValue(0.125);

  bool ok = false;
  ok = connect(m_UpdateButton, SIGNAL(pressed()), this, SLOT(OnManualUpdate()));
  assert(ok);
  ok = connect(m_BlockSizeSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnSpinBoxPressed()));
  assert(ok);
  ok = connect(m_OffsetSpinBox, SIGNAL(valueChanged(int)), this, SLOT(OnSpinBoxPressed()));
  assert(ok);
  ok = connect(m_MinSizeSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnSpinBoxPressed()));
  assert(ok);
  ok = connect(m_MaxSizeSpinBox, SIGNAL(valueChanged(double)), this, SLOT(OnSpinBoxPressed()));
  assert(ok);
  ok = connect(m_RegistrationEnabledCheckbox, SIGNAL(toggled(bool)), this, SLOT(OnRegistrationEnabledChecked(bool)));
  assert(ok);

  ctkServiceReference ref = mitk::TagTrackerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::TagTrackerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }

  mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();
  mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
  mitk::NodePredicateOr::Pointer isPointSetOrIsSurface = mitk::NodePredicateOr::New(isPointSet, isSurface);

  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  assert(dataStorage);

  m_RegistrationModelComboBox->SetDataStorage(dataStorage);
  m_RegistrationModelComboBox->SetPredicate(isPointSetOrIsSurface);
  m_RegistrationModelComboBox->SetAutoSelectNewItems(false);
  m_RegistrationMatrix->setEditable(false);
  m_RegistrationMatrix->setRange(-1e4, 1e4);

  this->RetrievePreferenceValues();

  m_InformationGroupBox->setCollapsed(true);
  m_TrackingParametersGroupBox->setCollapsed(true);
  m_RegistrationGroupBox->setCollapsed(true);
  m_RegistrationEnabledCheckbox->setChecked(false);
  this->OnRegistrationEnabledChecked(false);
  m_RegistrationMethodPointsRadio->setChecked(true);

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
void TagTrackerView::OnRegistrationEnabledChecked(bool isChecked)
{
  m_RegistrationModelComboBox->setEnabled(isChecked);
  m_RegistrationMethodPointsRadio->setEnabled(isChecked);
  m_RegistrationMethodPointsNormalsRadio->setEnabled(isChecked);
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
void TagTrackerView::OnSpinBoxPressed()
{ 
  m_RangesOfRotationalParams[0] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[1] = std::numeric_limits<double>::min();
  m_RangesOfRotationalParams[2] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[3] = std::numeric_limits<double>::min();
  m_RangesOfRotationalParams[4] = std::numeric_limits<double>::max();
  m_RangesOfRotationalParams[5] = std::numeric_limits<double>::min();
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

  double minSize = m_MinSizeSpinBox->value();
  double maxSize = m_MaxSizeSpinBox->value();
  int blockSize = m_BlockSizeSpinBox->value();
  int offset = m_OffsetSpinBox->value();

  if (leftNode.IsNotNull() || rightNode.IsNotNull())
  {
    bool needToLoadLeftCalib  = false;
    if (leftImage.IsNotNull())
    {
      needToLoadLeftCalib = niftk::Undistortion::NeedsToLoadIntrinsicCalib(leftIntrinsicFileName.toStdString(),  leftNode);
    }

    bool needToLoadRightCalib = false;
    bool needToLoadLeftToRightTransformation = false;
    if (rightImage.IsNotNull())
    {
      needToLoadRightCalib = niftk::Undistortion::NeedsToLoadIntrinsicCalib(rightIntrinsicFileName.toStdString(), rightNode);
      needToLoadLeftToRightTransformation = niftk::Undistortion::NeedsToLoadStereoRigExtrinsics(leftToRightTransformationFileName.toStdString(), rightImage);
    }

    if (needToLoadLeftCalib)
    {
      niftk::Undistortion::LoadIntrinsicCalibration(m_StereoCameraCalibrationSelectionWidget->GetLeftIntrinsicFileName().toStdString(), leftNode);
    }
    if (needToLoadRightCalib)
    {
      niftk::Undistortion::LoadIntrinsicCalibration(m_StereoCameraCalibrationSelectionWidget->GetRightIntrinsicFileName().toStdString(), rightNode);
    }
    if (needToLoadLeftToRightTransformation)
    {
      niftk::Undistortion::LoadStereoRig(
          m_StereoCameraCalibrationSelectionWidget->GetLeftToRightTransformationFileName().toStdString(),
          rightImage);
    }

    // Retrieve the point set node from data storage, or create it if it does not exist.
    mitk::PointSet::Pointer tagNormals = mitk::PointSet::New();
    mitk::PointSet::Pointer tagPointSet = NULL;
    mitk::DataNode::Pointer tagPointSetNode = dataStorage->GetNamedNode(mitk::TagTrackingRegistrationManager::POINTSET_NODE_ID);

    if (tagPointSetNode.IsNull())
    {
      tagPointSet = mitk::PointSet::New();
      tagPointSetNode = mitk::DataNode::New();
      tagPointSetNode->SetData( tagPointSet );
      tagPointSetNode->SetProperty( "name", mitk::StringProperty::New(mitk::TagTrackingRegistrationManager::POINTSET_NODE_ID));
      tagPointSetNode->SetProperty( "opacity", mitk::FloatProperty::New(1));
      tagPointSetNode->SetProperty( "point line width", mitk::IntProperty::New(1));
      tagPointSetNode->SetProperty( "point 2D size", mitk::IntProperty::New(5));
      tagPointSetNode->SetProperty( "pointsize", mitk::FloatProperty::New(5));
      tagPointSetNode->SetBoolProperty("helper object", false);
      tagPointSetNode->SetBoolProperty("show distant lines", false);
      tagPointSetNode->SetBoolProperty("show distant points", false);
      tagPointSetNode->SetBoolProperty("show distances", false);
      tagPointSetNode->SetProperty("layer", mitk::IntProperty::New(99));
      tagPointSetNode->SetColor( 1.0, 1.0, 0 );
      tagPointSetNode->SetVisibility(false);
      dataStorage->Add(tagPointSetNode);
    }
    else
    {
      tagPointSet = dynamic_cast<mitk::PointSet*>(tagPointSetNode->GetData());
      if (tagPointSet.IsNull())
      {
        // Give up, as the node has the wrong data.
        MITK_ERROR << "TagTrackerView::OnUpdate node " << mitk::TagTrackingRegistrationManager::POINTSET_NODE_ID << " does not contain an mitk::PointSet" << std::endl;
        return;
      }
    }

    // Similarly, create the transform node.
    mitk::CoordinateAxesData::Pointer transformToUpdate = NULL;
    mitk::DataNode::Pointer transformNode = dataStorage->GetNamedNode(mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID);
    if(transformNode.IsNull())
    {
      transformToUpdate = mitk::CoordinateAxesData::New();

      transformNode = mitk::DataNode::New();
      transformNode->SetData(transformToUpdate);
      transformNode->SetProperty( "name", mitk::StringProperty::New(mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID));
      transformNode->SetVisibility(false);

      dataStorage->Add(transformNode);
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
          minSize,
          maxSize,
          blockSize,
          offset,
          cameraToWorldMatrix,
          tagPointSet
          );
      tagPointSetNode->Modified();
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
      if (leftNode->GetName() == rightNode->GetName())
      {
        if(!m_ShownStereoSameNameWarning)
		    {
          m_ShownStereoSameNameWarning = true;
          QMessageBox msgBox;
          msgBox.setText("The left and right image are the same!");
          msgBox.setInformativeText("They need to be different for stereo tracking.");
          msgBox.setStandardButtons(QMessageBox::Ok);
          msgBox.setDefaultButton(QMessageBox::Ok);
          msgBox.exec();
          return;
		    }
        return;
      }
	  
      // Stereo Case.
      mitk::StereoTagExtractor::Pointer extractor = mitk::StereoTagExtractor::New();
      extractor->ExtractPoints(
          leftImage,
          rightImage,
          minSize,
          maxSize,
          blockSize,
          offset,
          cameraToWorldMatrix,
          tagPointSet,
          tagNormals
          );
    } // end if mono/stereo

    m_TagPositionDisplay->clear();

    vtkSmartPointer<vtkMatrix4x4> registrationMatrix = vtkMatrix4x4::New();
    registrationMatrix->Identity();

    int numberOfTrackedPoints = tagPointSet->GetSize();

    QString numberString;
    numberString.setNum(numberOfTrackedPoints);

    QString labelText;
    labelText = modeName + QString(" tags ") + numberString;

    if (numberOfTrackedPoints > 0)
    {
      mitk::PointSet::DataType* itkPointSet = tagPointSet->GetPointSet(0);
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

      if (m_RegistrationEnabledCheckbox->isChecked())
      {
        mitk::DataNode::Pointer selectedNode = m_RegistrationModelComboBox->GetSelectedNode();
        double fiducialRegistrationError = std::numeric_limits<double>::max();

        mitk::TagTrackingRegistrationManager::Pointer manager = mitk::TagTrackingRegistrationManager::New();
        bool isSuccessful = manager->Update(
             dataStorage,
             tagPointSet,
             tagNormals,
             selectedNode,
             mitk::TagTrackingRegistrationManager::TRANSFORM_NODE_ID,
             m_RegistrationMethodPointsNormalsRadio->isChecked(),
             *registrationMatrix,
             fiducialRegistrationError
            );

        QString fiducialRegistrationErrorString;
        fiducialRegistrationErrorString.setNum(fiducialRegistrationError);

        if (isSuccessful)
        {
          double pi = 3.14159265359;
          double cosRx = registrationMatrix->GetElement(0,0);
          double rx = acos(cosRx) * 180.0 / pi;
          double cosRy = registrationMatrix->GetElement(1,1);
          double ry = acos(cosRy) * 180.0 / pi;
          double cosRz = registrationMatrix->GetElement(2,2);
          double rz = acos(cosRz) * 180.0 / pi;

          QString rxString;
          rxString.setNum(rx);

          QString ryString;
          ryString.setNum(ry);

          QString rzString;
          rzString.setNum(rz);

          if (rx < m_RangesOfRotationalParams[0])
          {
            m_RangesOfRotationalParams[0] = rx;
          }
          if (ry < m_RangesOfRotationalParams[2])
          {
            m_RangesOfRotationalParams[2] = ry;
          }
          if (rz < m_RangesOfRotationalParams[4])
          {
            m_RangesOfRotationalParams[4] = rz;
          }
          if (rx > m_RangesOfRotationalParams[1])
          {
            m_RangesOfRotationalParams[1] = rx;
          }
          if (ry > m_RangesOfRotationalParams[3])
          {
            m_RangesOfRotationalParams[3] = ry;
          }
          if (rz > m_RangesOfRotationalParams[5])
          {
            m_RangesOfRotationalParams[5] = rz;
          }

          labelText += (QString(", FRE=") + fiducialRegistrationErrorString + QString(", rx=") + rxString + QString(", ry=") + ryString + QString(", rz=") + rzString);

          MITK_INFO << "Tag Tracking range, rx=(" << m_RangesOfRotationalParams[0] << ", " << m_RangesOfRotationalParams[1] << "), ry=(" <<m_RangesOfRotationalParams[2] << ", " << m_RangesOfRotationalParams[3] << "), rz=(" << m_RangesOfRotationalParams[4] << ", " << m_RangesOfRotationalParams[5] << ")" << std::endl;
        }
        else
        {
          labelText += (QString(", registration FAILED"));
        }

      } // end if we are doing registration

    } // end if we had some tracked points

    // Always update registration matrix contained within the view - just for visual feedback.
    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        m_RegistrationMatrix->setValue(i, j, registrationMatrix->GetElement(i, j));
      }
    }

    m_NumberOfTagsLabel->setText(labelText);

  } // end if we have at least one node specified
}
