/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSignalCollector.h"

#include <mitkSliceNavigationController.h>
#include <mitkFocusManager.h>

namespace mitk
{

//-----------------------------------------------------------------------------
SignalCollector::SignalCollector()
: itk::Command()
{
}


//-----------------------------------------------------------------------------
SignalCollector::~SignalCollector()
{
  this->Clear();
}


//-----------------------------------------------------------------------------
void SignalCollector::Execute(itk::Object* caller, const itk::EventObject& event)
{
  this->Execute((const itk::Object*) caller, event);
}


//-----------------------------------------------------------------------------
void SignalCollector::Execute(const itk::Object* object, const itk::EventObject& event)
{
  /// Create a copy of the event as a newly allocated object.
  m_Signals.push_back(Signal(object, event.MakeObject()));
}


//-----------------------------------------------------------------------------
const SignalCollector::Signals& SignalCollector::GetSignals() const
{
  return m_Signals;
}


//-----------------------------------------------------------------------------
void SignalCollector::Clear()
{
  /// Destruct the copies of the original events.
  Signals::iterator it = m_Signals.begin();
  Signals::iterator signalsEnd = m_Signals.end();
  for ( ; it != signalsEnd; ++it)
  {
    delete it->second;
  }

  /// Remove the elements.
  m_Signals.clear();
}


//-----------------------------------------------------------------------------
void SignalCollector::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Signals::const_iterator it = m_Signals.begin();
  Signals::const_iterator signalsEnd = m_Signals.end();
  for ( ; it != signalsEnd; ++it)
  {
    os << indent << ((void*) it->first) << ": " << it->second->GetEventName() << std::endl;
  }
}

}
