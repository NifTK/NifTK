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
#include <mitkTrackedPointerCommand.h>
#include "TrackedPointerViewActivator.h"
#include <QmitkFileIOUtils.h>

const std::string TrackedPointerView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedpointer";

//-----------------------------------------------------------------------------
TrackedPointerView::TrackedPointerView()
: m_Controls(NULL)
{
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

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
    m_Controls->m_ProbeSurfaceNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeSurfaceNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeSurfaceNode->SetPredicate(isSurface);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);

    m_Controls->m_TipOriginSpinBoxes->setSingleStep(0.01);
    m_Controls->m_TipOriginSpinBoxes->setDecimals(2);
    m_Controls->m_TipOriginSpinBoxes->setMinimum(-10000);
    m_Controls->m_TipOriginSpinBoxes->setMaximum(10000);
    m_Controls->m_TipOriginSpinBoxes->setCoordinates(0,0,0);

    connect(m_Controls->m_TipToProbeCalibrationFile, SIGNAL(currentPathChanged(QString)), this, SLOT(OnTipToProbeChanged()));

    RetrievePreferenceValues();

    ctkServiceReference ref = mitk::TrackedPointerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::TrackedPointerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }
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

  }
}


//-----------------------------------------------------------------------------
void TrackedPointerView::SetFocus()
{
  m_Controls->m_TipToProbeCalibrationFile->setFocus();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnTipToProbeChanged()
{
  m_TipToProbeTransform = Load4x4MatrixFromFile(m_Controls->m_TipToProbeCalibrationFile->currentPath());
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);

  mitk::DataNode::Pointer surfaceNode = m_Controls->m_ProbeSurfaceNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();

  mitk::Point3D tipCoordinate;
  const double *currentCoordinateInModelCoordinates = m_Controls->m_TipOriginSpinBoxes->coordinates();
  tipCoordinate[0] = currentCoordinateInModelCoordinates[0];
  tipCoordinate[1] = currentCoordinateInModelCoordinates[1];
  tipCoordinate[2] = currentCoordinateInModelCoordinates[2];

  mitk::TrackedPointerCommand::Pointer command = mitk::TrackedPointerCommand::New();
  command->Update(m_TipToProbeTransform,
                  probeToWorldTransform,
                  surfaceNode,             // The Geometry on this gets updated.
                  tipCoordinate            // This gets updated.
                  );

  this->SetViewToCoordinate(tipCoordinate);
}
