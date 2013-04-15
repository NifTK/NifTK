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
#include "mitkCoordinateAxesData.h"
#include "TrackedPointerViewActivator.h"
#include "mitkTrackedPointerCommand.h"

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

    mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
    m_Controls->m_ImageNode->SetDataStorage(dataStorage);
    m_Controls->m_ImageNode->SetAutoSelectNewItems(false);
    m_Controls->m_ImageNode->SetPredicate(isImage);

    mitk::TNodePredicateDataType<mitk::Surface>::Pointer isSurface = mitk::TNodePredicateDataType<mitk::Surface>::New();
    m_Controls->m_ProbeSurfaceNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeSurfaceNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeSurfaceNode->SetPredicate(isSurface);

    mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
    m_Controls->m_ProbeToWorldNode->SetDataStorage(dataStorage);
    m_Controls->m_ProbeToWorldNode->SetAutoSelectNewItems(false);
    m_Controls->m_ProbeToWorldNode->SetPredicate(isTransform);

    connect(m_Controls->m_ImageToProbeCalibrationFile, SIGNAL(currentPathChanged(QString)), this, SLOT(OnImageToProbeChanged()));

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
  m_Controls->m_ImageNode->setFocus();
}


//-----------------------------------------------------------------------------
void TrackedPointerView::LoadImageToProbeTransform(const QString& fileName)
{
  QFile matrixFile(fileName);
  if (!matrixFile.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    MITK_ERROR << "TrackedPointerView::LoadImageToProbeTransform, failed to open file:" << fileName.toStdString() << std::endl;
    return;
  }

  QTextStream matrixIn(&matrixFile);

  vtkMatrix4x4 *matrix = vtkMatrix4x4::New();

  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      double tmp;
      matrixIn >> tmp;
      matrix->SetElement(row, col, tmp);
    }
  }
  matrixFile.close();
  m_ImageToProbeTransform = matrix;
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnImageToProbeChanged()
{
  this->LoadImageToProbeTransform(m_Controls->m_ImageToProbeCalibrationFile->currentPath());
}


//-----------------------------------------------------------------------------
void TrackedPointerView::OnUpdate(const ctkEvent& event)
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  mitk::DataNode::Pointer imageNode = m_Controls->m_ImageNode->GetSelectedNode();
  mitk::DataNode::Pointer surfaceNode = m_Controls->m_ProbeSurfaceNode->GetSelectedNode();
  mitk::DataNode::Pointer probeToWorldTransform = m_Controls->m_ProbeToWorldNode->GetSelectedNode();

  mitk::TrackedPointerCommand::Pointer command = mitk::TrackedPointerCommand::New();
}
