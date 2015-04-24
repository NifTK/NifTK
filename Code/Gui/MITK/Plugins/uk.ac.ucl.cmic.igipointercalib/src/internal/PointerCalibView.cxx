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
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <QMessageBox>
#include <QCoreApplication>
#include <QObject>

const std::string PointerCalibView::VIEW_ID = "uk.ac.ucl.cmic.igipointercalib";

//-----------------------------------------------------------------------------
PointerCalibView::PointerCalibView()
: m_Controls(NULL)
, m_DataStorage(NULL)
, m_ImagePointsAddObserverTag(0)
, m_ImagePointsRemoveObserverTag(0)
{
  m_ImagePoints = mitk::PointSet::New();
  m_ImagePointsNode = mitk::DataNode::New();
  m_ImagePointsNode->SetData(m_ImagePoints);
  m_ImagePointsNode->SetName("PointerCalibImagePoints");
  m_SensorPoints = mitk::PointSet::New();
  m_TipCoordinate[0] = 0;
  m_TipCoordinate[1] = 0;
  m_TipCoordinate[2] = 0;

  m_Calibrator = mitk::UltrasoundPointerBasedCalibration::New();
  m_Calibrator->SetImagePoints(m_ImagePoints);
  m_Calibrator->SetSensorPoints(m_SensorPoints);

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

  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->Remove(m_ImagePointsNode);
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

    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);

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

    RetrievePreferenceValues();

    ctkServiceReference ref = mitk::PointerCalibViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::PointerCalibViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }
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
  if (m_ImagePoints->GetSize() > 3 && m_SensorPoints->GetSize())
  {
    double fre = m_Calibrator->DoPointerBasedCalibration();
  }
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnPointAdded()
{
  this->UpdateDisplayedPoints();
  this->UpdateRegistration();
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnPointRemoved()
{
  this->UpdateDisplayedPoints();
  this->UpdateRegistration();
}


//-----------------------------------------------------------------------------
void PointerCalibView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer sensorToWorldTransform = m_Controls->m_SensorToWorldNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();

  const double *currentCoordinateInModelCoordinates = m_Controls->m_TipOriginSpinBoxes->coordinates();

  if (   probeToWorldTransform.IsNotNull()
      && sensorToWorldTransform.IsNotNull()
      && currentCoordinateInModelCoordinates != NULL)
  {
    mitk::Point3D tipCoordinate;

    tipCoordinate[0] = currentCoordinateInModelCoordinates[0];
    tipCoordinate[1] = currentCoordinateInModelCoordinates[1];
    tipCoordinate[2] = currentCoordinateInModelCoordinates[2];

    m_Controls->m_MapsToSpinBoxes->setCoordinates(tipCoordinate[0], tipCoordinate[1], tipCoordinate[2]);
  }
}
