/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKTrackerDataSourceService.h"
#include <niftkIGIMatrixPerFileBackend.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::MITKTrackerDataSourceService(
  QString name,
  QString factoryName,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage,
  niftk::NDITracker::Pointer tracker
)
: IGITrackerDataSourceService(name, factoryName, dataStorage)
, m_DataGrabbingThread(nullptr)
{
  if (tracker.IsNull())
  {
    mitkThrow() << "Tracker is NULL!";
  }

  // First assign to base class (protected) pointers.
  m_Tracker = tracker;
  m_BackEnd = niftk::IGIMatrixPerFileBackend::New(this->GetName(), this->GetDataStorage());
  m_BackEnd->SetExpectedFramesPerSecond(m_Tracker->GetExpectedFramesPerSecond());

  this->SetTimeStampTolerance((m_Tracker->GetExpectedFramesPerSecond()
                               / m_Tracker->GetExpectedNumberOfTools())
                              *1000000 // convert to nanoseconds
                              *5       // allow up to 5 frames late
                             );

  this->SetProperties(properties);
  this->SetShouldUpdate(true);

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);

  bool isOk = connect(m_DataGrabbingThread, SIGNAL(ErrorOccured(QString)), this, SLOT(OnErrorFromThread(QString)));
  assert(isOk);

  m_DataGrabbingThread->SetUseFastPolling(true);
  m_DataGrabbingThread->SetInterval(1000 / m_Tracker->GetExpectedFramesPerSecond());
  m_DataGrabbingThread->start();

  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetDescription(this->GetName());
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
MITKTrackerDataSourceService::~MITKTrackerDataSourceService()
{
  m_DataGrabbingThread->ForciblyStop();

  bool isOk = disconnect(m_DataGrabbingThread, SIGNAL(ErrorOccured(QString)), this, SLOT(OnErrorFromThread(QString)));
  assert(isOk);

  delete m_DataGrabbingThread;
}


//-----------------------------------------------------------------------------
void MITKTrackerDataSourceService::OnErrorFromThread(QString message)
{
  this->OnErrorOccurred(message);
}

} // end namespace
