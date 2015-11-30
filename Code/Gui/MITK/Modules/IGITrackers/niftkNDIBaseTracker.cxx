/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNDIBaseTracker.h"
#include <mitkException.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NDIBaseTracker::NDIBaseTracker()
: m_PreferredFramesPerSecond(60)
{
  // Derived class constructor MUST populate m_Tracker, and set m_PreferredFramesPerSecond.
}


//-----------------------------------------------------------------------------
NDIBaseTracker::~NDIBaseTracker()
{
  // Smart pointer deletes tracker.
}


//-----------------------------------------------------------------------------
void NDIBaseTracker::StartTracking()
{
  m_Tracker->StartTracking();
}


//-----------------------------------------------------------------------------
void NDIBaseTracker::StopTracking()
{
  m_Tracker->StopTracking();
}


//-----------------------------------------------------------------------------
void NDIBaseTracker::SetVisibilityOfTrackingVolume(bool isVisible)
{
  m_Tracker->SetVisibilityOfTrackingVolume(isVisible);
}


//-----------------------------------------------------------------------------
bool NDIBaseTracker::GetVisibilityOfTrackingVolume() const
{
  return m_Tracker->GetVisibilityOfTrackingVolume();
}


//-----------------------------------------------------------------------------
void NDIBaseTracker::Update()
{
  m_Tracker->Update();
}


//-----------------------------------------------------------------------------
std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > NDIBaseTracker::GetTrackingData()
{
  return m_Tracker->GetTrackingData();
}

} // end namespace
