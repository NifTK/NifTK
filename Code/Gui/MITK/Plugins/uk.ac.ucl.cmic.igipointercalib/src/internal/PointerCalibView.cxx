/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "PointerCalibView.h"
#include "PointerCalibViewPreferencePage.h"
#include "PointerCalibViewActivator.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateOr.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesData.h>
#include <mitkFileIOUtils.h>
#include <mitkPointUtils.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <QMessageBox>
#include <QCoreApplication>
#include <QObject>
#include <QFileDialog>

const std::string PointerCalibView::VIEW_ID = "uk.ac.ucl.cmic.igipointercalib";

//-----------------------------------------------------------------------------
PointerCalibView::PointerCalibView()
: m_Controls(NULL)
, m_DataStorage(NULL)
, m_Interactor(NULL)
, m_ImagePointsAddObserverTag(0)
, m_ImagePointsRemoveObserverTag(0)
{
  m_Calibrator = mitk::UltrasoundPointerBasedCalibration::New();
  m_Calibrator->SetImagePoints(m_ImagePoints);
  m_Calibrator->SetSensorPoints(m_SensorPoints);

  m_ImagePoints = mitk::PointSet::New();
  m_ImagePointsNode = mitk::DataNode::New();
  m_ImagePointsNode->SetData(m_ImagePoints);
  m_ImagePointsNode->SetName("PointerCalibImagePoints");
  m_SensorPoints = mitk::PointSet::New();

  itk::SimpleMemberCommand<PointerCalibView>::Pointer pointAddedCommand = itk::SimpleMemberCommand<PointerCalibView>::New();
  pointAddedCommand->SetCallbackFunction(this, &PointerCalibView::OnPointAdded);
  m_ImagePointsAddObserverTag = m_ImagePoints->AddObserver( mitk::PointSetAddEvent(), pointAddedCommand);

  itk::SimpleMemberCommand<PointerCalibView>::Pointer pointRemovedCommand = itk::SimpleMemberCommand<PointerCalibView>::New();
  pointRemovedCommand->SetCallbackFunction(this, &PointerCalibView::OnPointRemoved);
  m_ImagePointsRemoveObserverTag = m_ImagePoints->AddObserver( mitk::PointSetRemoveEvent(), pointRemovedCommand);
}


//-----------------------------------------------------------------------------
PointerCalibView::~PointerCalibView()
{
  m_ImagePoints->RemoveObserver(m_ImagePointsRemoveObserverTag);
  m_ImagePoints->RemoveObserver(m_ImagePointsAddObserverTag);

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->Remove(m_ImagePointsNode);
  }
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string PointerCalibView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void PointerCalibView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::PointerCalibView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_DataStorage = dataStorage;
    m_DataStorage->Add(m_ImagePointsNode);
    m_Interactor = mitk::PointSetDataInteractor::New();
    m_Interactor->LoadStateMachine("PointSet.xml");
    m_Interactor->SetEventConfig("PointSetConfig.xml");
    m_Interactor->SetDataNode(m_ImagePointsNode);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();

    m_Controls->m_PointerToWorldNode->SetPredicate(isTransform);
    m_Controls->m_PointerToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_PointerToWorldNode->SetAutoSelectNewItems(false);

    m_Controls->m_SensorToWorldNode->SetPredicate(isTransform);
    m_Controls->m_SensorToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_SensorToWorldNode->SetAutoSelectNewItems(false);

    m_Controls->m_TipOriginSpinBoxes->setSingleStep(0.01);
    m_Controls->m_TipOriginSpinBoxes->setDecimals(3);
    m_Controls->m_TipOriginSpinBoxes->setMinimum(-100000);
    m_Controls->m_TipOriginSpinBoxes->setMaximum(100000);
    m_Controls->m_TipOriginSpinBoxes->setCoordinates(0,0,0);

    m_Controls->m_MapsToSpinBoxes->setSingleStep(0.01);
    m_Controls->m_MapsToSpinBoxes->setDecimals(3);
    m_Controls->m_MapsToSpinBoxes->setCoordinates(0,0,0);

    m_Controls->m_ScalingParametersLabel->setText("");
    m_Controls->m_FiducialRegistrationErrorLabel->setText("");

    RetrievePreferenceValues();

    ctkServiceReference ref = mitk::PointerCalibViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::PointerCalibViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }

    connect(m_Controls->m_SaveToFileButton, SIGNAL(pressed()), this, SLOT(OnSaveToFileButtonPressed()));
  }
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void PointerCalibView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
  }
}


//-----------------------------------------------------------------------------
void PointerCalibView::SetFocus()
{
  m_Controls->m_SensorToWorldNode->setFocus();
}


//-----------------------------------------------------------------------------
mitk::PointSet::PointIdentifier PointerCalibView::GetMissingPointId(const mitk::PointSet::Pointer& a,
                                                                    const mitk::PointSet::Pointer& b)
{
  if (a->GetSize() == 0 && b->GetSize() == 0)
  {
    mitkThrow() << "Both point sets are empty";
  }
  if (abs(a->GetSize() - b->GetSize()) != 1)
  {
    mitkThrow() << "Point sets do not differ by 1";
  }
  mitk::PointSet::Pointer smaller = a;
  mitk::PointSet::Pointer larger = b;
  if (b->GetSize() < a->GetSize())
  {
    smaller = b;
    larger = a;
  }

  mitk::PointSet::DataType* itkPointSet = larger->GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    if (!smaller->IndexExists(pointID))
    {
      return pointID;
    }
  }

  mitkThrow() << "Didn't find a missing point.";
}


//-----------------------------------------------------------------------------
void PointerCalibView::UpdateDisplayedPoints()
{
  mitk::PointSet::Pointer pointSet = m_ImagePoints;
  mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType imagePoint;
  mitk::PointSet::PointType sensorPoint;

  m_Controls->m_PointsTextBox->clear();

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    imagePoint = pIt->Value();
    sensorPoint = m_SensorPoints->GetPoint(pointID);

    m_Controls->m_PointsTextBox->appendPlainText(tr("%1:Image[%2, %3, %4]->Sensor[%5, %6, %7]")
      .arg(pointID)
      .arg(imagePoint[0]).arg(imagePoint[1]).arg(imagePoint[2])
      .arg(sensorPoint[0]).arg(sensorPoint[1]).arg(sensorPoint[2])
      );
  }
  m_Controls->m_PointsTextBox->appendPlainText(tr("size:%1").arg(pointSet->GetSize()));
}


//-----------------------------------------------------------------------------
void PointerCalibView::UpdateRegistration()
{
  double fre = 0;

  if (m_ImagePoints->GetSize() > 3 && m_SensorPoints->GetSize() > 3)
  {
    fre = m_Calibrator->DoPointerBasedCalibration();
    m_Controls->m_FiducialRegistrationErrorLabel->setText(tr("FRE: %1").arg(fre));

    vtkSmartPointer<vtkMatrix4x4> scaling = m_Calibrator->GetScalingMatrix();
    vtkSmartPointer<vtkMatrix4x4> rigid = m_Calibrator->GetRigidBodyMatrix();

    m_Controls->m_ScalingParametersLabel->setText(tr("scaling: %1, %2").arg(scaling->GetElement(0,0)).arg(scaling->GetElement(1,1)));
    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        m_Controls->m_RigidMatrix->setValue(i, j, rigid->GetElement(i, j));
      }
    }
  }
  else
  {
    vtkSmartPointer<vtkMatrix4x4> identity = vtkSmartPointer<vtkMatrix4x4>::New();
    identity->Identity();

    m_Controls->m_FiducialRegistrationErrorLabel->setText(tr("FRE: %1").arg(fre));
    m_Controls->m_ScalingParametersLabel->setText(tr("scaling: %1, %2").arg(identity->GetElement(0,0)).arg(identity->GetElement(1,1)));
    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        m_Controls->m_RigidMatrix->setValue(i, j, identity->GetElement(i, j));
      }
    }
  }
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnPointAdded()
{
  mitk::PointSet::PointIdentifier pointID = this->GetMissingPointId(m_ImagePoints, m_SensorPoints);
  mitk::Point3D point = this->GetPointerTipInSensorCoordinates();
  m_SensorPoints->InsertPoint(pointID, point);

  this->UpdateRegistration();
  this->UpdateDisplayedPoints();
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnPointRemoved()
{
  mitk::PointSet::PointIdentifier pointIDToRemove = this->GetMissingPointId(m_ImagePoints, m_SensorPoints);

  mitk::PointSet::Pointer temporaryPoints = mitk::PointSet::New();
  mitk::PointSet::DataType* itkPointSet = m_SensorPoints->GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    if (pointID != pointIDToRemove)
    {
      temporaryPoints->InsertPoint(pointID, pIt->Value());
    }
  }
  mitk::CopyPointSets(*temporaryPoints, *m_SensorPoints);

  this->UpdateRegistration();
  this->UpdateDisplayedPoints();
}


//-----------------------------------------------------------------------------
mitk::Point3D PointerCalibView::GetPointerTipInSensorCoordinates() const
{
  mitk::Point3D tip;
  tip.Fill(0);

  mitk::DataNode::Pointer pointerToWorldNode = m_Controls->m_PointerToWorldNode->GetSelectedNode();
  mitk::DataNode::Pointer sensorToWorldNode = m_Controls->m_SensorToWorldNode->GetSelectedNode();
  const double *tipCoordinateInModelCoordinates = m_Controls->m_TipOriginSpinBoxes->coordinates();

  if (   sensorToWorldNode.IsNotNull()
      && pointerToWorldNode.IsNotNull()
      && tipCoordinateInModelCoordinates != NULL)
  {

    mitk::CoordinateAxesData::Pointer sensorToWorldTransform =
        dynamic_cast<mitk::CoordinateAxesData*>(sensorToWorldNode->GetData());

    mitk::CoordinateAxesData::Pointer pointerToWorldTransform =
        dynamic_cast<mitk::CoordinateAxesData*>(pointerToWorldNode->GetData());

    if (sensorToWorldTransform.IsNotNull() && pointerToWorldTransform.IsNotNull())
    {
      vtkSmartPointer<vtkMatrix4x4> pointerToWorldMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      pointerToWorldTransform->GetVtkMatrix(*pointerToWorldMatrix);

      vtkSmartPointer<vtkMatrix4x4> worldToSensorMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
      sensorToWorldTransform->GetVtkMatrix(*worldToSensorMatrix);
      worldToSensorMatrix->Invert();

      double tipCoordinate[4];
      double tipCoordinateInWorldSpace[4];
      double tipCoordinateInSensorSpace[4];

      tipCoordinate[0] = tipCoordinateInModelCoordinates[0];
      tipCoordinate[1] = tipCoordinateInModelCoordinates[1];
      tipCoordinate[2] = tipCoordinateInModelCoordinates[2];
      tipCoordinate[3] = 1;

      pointerToWorldMatrix->MultiplyPoint(tipCoordinate, tipCoordinateInWorldSpace);
      worldToSensorMatrix->MultiplyPoint(tipCoordinateInWorldSpace, tipCoordinateInSensorSpace);

      tip[0] = tipCoordinateInSensorSpace[0];
      tip[1] = tipCoordinateInSensorSpace[1];
      tip[2] = tipCoordinateInSensorSpace[2];
    }
  }
  return tip;
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::Point3D tip = this->GetPointerTipInSensorCoordinates();
  m_Controls->m_MapsToSpinBoxes->setCoordinates(tip[0], tip[1], tip[2]);
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnSaveToFileButtonPressed()
{
  QString fileName = QFileDialog::getSaveFileName( NULL,
                                                   tr("Save Transform As ..."),
                                                   QDir::currentPath(),
                                                   "Matrix file (*.mat);;4x4 file (*.4x4);;Text file (*.txt);;All files (*.*)" );

  vtkSmartPointer<vtkMatrix4x4> matrix = m_Calibrator->GetRigidBodyMatrix();
  if (fileName.size() > 0)
  {
    mitk::SaveVtkMatrix4x4ToFile(fileName.toStdString(), *matrix);
  }
}
