/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TrackedPointerView.h"
#include "TrackedPointerViewPreferencePage.h"
#include "TrackedPointerViewActivator.h"

#include <QCoreApplication>
#include <QMessageBox>
#include <QObject>

#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

#include <vtkMatrix4x4.h>

#include <mitkImage.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateOr.h>
#include <mitkPointSet.h>
#include <mitkSurface.h>
#include <mitkTrackedPointer.h>

#include <niftkCoordinateAxesData.h>
#include <niftkFileIOUtils.h>

const QString TrackedPointerView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedpointer";

//-----------------------------------------------------------------------------
TrackedPointerView::TrackedPointerView()
: m_Controls(NULL)
, m_UpdateViewCoordinate(false)
, m_NumberOfPointsToAverageOver(1)
, m_RemainingPointsCounter(0)
{
  m_TrackedPointer = mitk::TrackedPointer::New();
}


//-----------------------------------------------------------------------------
TrackedPointerView::~TrackedPointerView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::TrackedPointerView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_DataStorage = dataStorage;

    m_TrackedPointer->SetDataStorage(dataStorage);

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
    mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();
    mitk::NodePredicateOr::Pointer isSurfaceOrPointSet = mitk::NodePredicateOr::New(isSurface, isPointSet);

    m_Controls->m_ProbeSurfaceNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeSurfaceNode->SetPredicate(isSurfaceOrPointSet);
    m_Controls->m_ProbeSurfaceNode->SetDataStorage(dataStorage);

    mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<niftk::CoordinateAxesData>::New();
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->setCurrentIndex(0);

    m_Controls->m_TipOriginSpinBoxes->setSingleStep(0.01);
    m_Controls->m_TipOriginSpinBoxes->setDecimals(2);
    m_Controls->m_TipOriginSpinBoxes->setMinimum(-100000);
    m_Controls->m_TipOriginSpinBoxes->setMaximum(100000);
    m_Controls->m_TipOriginSpinBoxes->setCoordinates(0,0,0);

    m_Controls->m_MapsToSpinBoxes->setSingleStep(0.01);
    m_Controls->m_MapsToSpinBoxes->setDecimals(2);
    m_Controls->m_MapsToSpinBoxes->setCoordinates(0,0,0);

    RetrievePreferenceValues();

    ctkServiceReference ref = mitk::TrackedPointerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::TrackedPointerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }

    connect(this->m_Controls->m_GrabPointsButton, SIGNAL(pressed()), this, SLOT(OnStartGrabPoints()));
    connect(this->m_Controls->m_ClearPointsButton, SIGNAL(pressed()), this, SLOT(OnClearPoints()));
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {
    m_TipToProbeFileName = prefs->Get(TrackedPointerViewPreferencePage::CALIBRATION_FILE_NAME, "").toStdString();
    m_TipToProbeTransform = niftk::LoadVtkMatrix4x4FromFile(m_TipToProbeFileName);

    m_UpdateViewCoordinate = prefs->GetBool(TrackedPointerViewPreferencePage::UPDATE_VIEW_COORDINATE_NAME, mitk::TrackedPointer::UPDATE_VIEW_COORDINATE_DEFAULT);
    m_NumberOfPointsToAverageOver = prefs->GetInt(TrackedPointerViewPreferencePage::NUMBER_OF_SAMPLES_TO_AVERAGE, 1);
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerView::SetFocus()
{
  m_Controls->m_ProbeSurfaceNode->setFocus();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnStartGrabPoints()
{
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();
  if (probeToWorldTransform.IsNull())
  {
    QString message("Please select a pointer to world transformation!");
    QMessageBox::warning(NULL, tr("%1").arg(QCoreApplication::applicationName()),
                               tr("%1").arg(message),
                               QMessageBox::Ok);
    return;
  }

  m_RemainingPointsCounter = m_NumberOfPointsToAverageOver;
  m_TipCoordinate[0] = 0;
  m_TipCoordinate[1] = 0;
  m_TipCoordinate[2] = 0;
  QString text = QString("grab %1").arg(m_RemainingPointsCounter);
  m_Controls->m_GrabPointsButton->setText(text);
}


//-----------------------------------------------------------------------------
void TrackedPointerView::UpdateDisplayedPoints()
{
  mitk::PointSet::Pointer pointSet = m_TrackedPointer->RetrievePointSet();
  mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet();
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType point;

  m_Controls->m_PointsTextBox->clear();

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    pointID = pIt->Index();
    point = pIt->Value();

    m_Controls->m_PointsTextBox->appendPlainText(tr("%1:[%2, %3, %4]").arg(pointID).arg(point[0]).arg(point[1]).arg(point[2]));
  }
  m_Controls->m_PointsTextBox->appendPlainText(tr("size:%1").arg(pointSet->GetSize()));
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnClearPoints()
{
  m_TrackedPointer->OnClearPoints();
  this->UpdateDisplayedPoints();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer probeModel = m_Controls->m_ProbeSurfaceNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();
  const double *currentCoordinateInModelCoordinates = m_Controls->m_TipOriginSpinBoxes->coordinates();

  // dont move our own output pointset
  if (probeModel == GetDataStorage()->GetNamedNode(mitk::TrackedPointer::TRACKED_POINTER_POINTSET_NAME))
    probeModel = mitk::DataNode::Pointer();

  if (   probeToWorldTransform.IsNotNull()
      && currentCoordinateInModelCoordinates != NULL)
  {
    mitk::Point3D tipCoordinate;

    tipCoordinate[0] = currentCoordinateInModelCoordinates[0];
    tipCoordinate[1] = currentCoordinateInModelCoordinates[1];
    tipCoordinate[2] = currentCoordinateInModelCoordinates[2];

    m_TrackedPointer->Update(*m_TipToProbeTransform,
                                    probeToWorldTransform,
                                    probeModel,              // The Geometry on this gets updated, so we surface model moving
                                    tipCoordinate            // This gets updated.
                                   );


    if (m_RemainingPointsCounter > 0)
    {
      m_TipCoordinate[0] += tipCoordinate[0];
      m_TipCoordinate[1] += tipCoordinate[1];
      m_TipCoordinate[2] += tipCoordinate[2];
      m_RemainingPointsCounter--;

      QString text = QString("grab %1").arg(m_RemainingPointsCounter);
      m_Controls->m_GrabPointsButton->setText(text);

      if (m_RemainingPointsCounter == 0)
      {
        double divisor = static_cast<double>(m_NumberOfPointsToAverageOver);
        m_TipCoordinate[0] /= divisor;
        m_TipCoordinate[1] /= divisor;
        m_TipCoordinate[2] /= divisor;

        m_TrackedPointer->OnGrabPoint(m_TipCoordinate);
        m_Controls->m_GrabPointsButton->setText("grab");
      }
    }

    m_Controls->m_MapsToSpinBoxes->setCoordinates(tipCoordinate[0], tipCoordinate[1], tipCoordinate[2]);
    if (m_UpdateViewCoordinate)
    {
      this->SetViewToCoordinate(tipCoordinate);
    }
  }
  this->UpdateDisplayedPoints();
}
