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
#include <QMessageBox>
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkSurface.h>
#include <vtkMatrix4x4.h>
#include <mitkCoordinateAxesData.h>
#include <mitkTrackedPointerManager.h>
#include <mitkFileIOUtils.h>

const std::string TrackedPointerView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedpointer";

//-----------------------------------------------------------------------------
TrackedPointerView::TrackedPointerView()
: m_Controls(NULL)
, m_UpdateViewCoordinate(false)
{
  m_TrackedPointerManager = mitk::TrackedPointerManager::New();
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
std::string TrackedPointerView::GetViewID() const
{
  return VIEW_ID;
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

    m_TrackedPointerManager->SetDataStorage(dataStorage);

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
    m_Controls->m_ProbeSurfaceNode->SetPredicate(isSurface);
    m_Controls->m_ProbeSurfaceNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeSurfaceNode->SetAutoSelectNewItems(false);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);

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

    connect(this->m_Controls->m_GrabPointsButton, SIGNAL(pressed()), this, SLOT(OnGrabPoints()));
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
    m_TipToProbeFileName = prefs->Get(TrackedPointerViewPreferencePage::CALIBRATION_FILE_NAME, "").c_str();
    m_TipToProbeTransform = mitk::LoadVtkMatrix4x4FromFile(m_TipToProbeFileName);

    m_UpdateViewCoordinate = prefs->GetBool(TrackedPointerViewPreferencePage::UPDATE_VIEW_COORDINATE_NAME, mitk::TrackedPointerManager::UPDATE_VIEW_COORDINATE_DEFAULT);
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerView::SetFocus()
{
  m_Controls->m_ProbeSurfaceNode->setFocus();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnGrabPoints()
{
  const double *currentTipCoordinate = m_Controls->m_MapsToSpinBoxes->coordinates();
  mitk::Point3D tipCoordinate;

  tipCoordinate[0] = currentTipCoordinate[0];
  tipCoordinate[1] = currentTipCoordinate[1];
  tipCoordinate[2] = currentTipCoordinate[2];

  m_TrackedPointerManager->OnGrabPoint(tipCoordinate);
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnClearPoints()
{
  QMessageBox msgBox;
  msgBox.setText("This operation cannot be un-done.");
  msgBox.setInformativeText("Are you sure you want to erase the point-set?");
  msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  int returnValue = msgBox.exec();
  if (returnValue == QMessageBox::No)
  {
    return;
  }

  m_TrackedPointerManager->OnClearPoints();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer surfaceNode = m_Controls->m_ProbeSurfaceNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();
  const double *currentCoordinateInModelCoordinates = m_Controls->m_TipOriginSpinBoxes->coordinates();

  if (m_TipToProbeTransform != NULL
      && probeToWorldTransform.IsNotNull()
      && currentCoordinateInModelCoordinates != NULL)
  {
    mitk::Point3D tipCoordinate;

    tipCoordinate[0] = currentCoordinateInModelCoordinates[0];
    tipCoordinate[1] = currentCoordinateInModelCoordinates[1];
    tipCoordinate[2] = currentCoordinateInModelCoordinates[2];

    m_TrackedPointerManager->Update(m_TipToProbeTransform,
                                    probeToWorldTransform,
                                    surfaceNode,             // The Geometry on this gets updated.
                                    tipCoordinate            // This gets updated.
                                   );

    m_Controls->m_MapsToSpinBoxes->setCoordinates(tipCoordinate[0], tipCoordinate[1], tipCoordinate[2]);
    if (m_UpdateViewCoordinate)
    {
      this->SetViewToCoordinate(tipCoordinate);
    }
  }
}
