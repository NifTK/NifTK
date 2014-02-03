/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkItkSignalCollector.h"

#include <mitkSliceNavigationController.h>
#include <mitkFocusManager.h>

namespace mitk
{

//-----------------------------------------------------------------------------
ItkSignalCollector::ItkSignalCollector()
: itk::Command()
{
}


//-----------------------z\\------------------------------------------------------
ItkSignalCollector::~ItkSignalCollector()
{
  ObserverMap::iterator observerTagIt = m_ObserverTags.begin();
  ObserverMap::iterator observerTagEnd = m_ObserverTags.end();
  for ( ; observerTagIt != observerTagEnd; ++observerTagIt)
  {
    itk::Object* object = observerTagIt->first;
    unsigned long observerTag = observerTagIt->second;
    object->RemoveObserver(observerTag);
  }
  this->Clear();
}


//-----------------------------------------------------------------------------
void ItkSignalCollector::Connect(itk::Object* object, const itk::EventObject& event)
{
  unsigned long observerTag = object->AddObserver(event, this);
  m_ObserverTags.insert(ObserverMap::value_type(object, observerTag));
}


//-----------------------------------------------------------------------------
void ItkSignalCollector::Execute(itk::Object* caller, const itk::EventObject& event)
{
  this->ProcessEvent(caller, event);
}


//-----------------------------------------------------------------------------
void ItkSignalCollector::Execute(const itk::Object* caller, const itk::EventObject& event)
{
  this->ProcessEvent(caller, event);
}


//-----------------------------------------------------------------------------
void ItkSignalCollector::ProcessEvent(const itk::Object* caller, const itk::EventObject& event)
{
  /// Create a copy of the event as a newly allocated object.
  m_Signals.push_back(Signal(caller, event.MakeObject()));
}


//-----------------------------------------------------------------------------
const ItkSignalCollector::Signals& ItkSignalCollector::GetSignals() const
{
  return m_Signals;
}


//-----------------------------------------------------------------------------
void ItkSignalCollector::Clear()
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
void ItkSignalCollector::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Signals::const_iterator it = m_Signals.begin();
  Signals::const_iterator signalsEnd = m_Signals.end();
  int i = 0;
  for ( ; it != signalsEnd; ++it, ++i)
  {
    os << indent << i << ": " << ((void*) it->first) << ": " << it->second->GetEventName() << std::endl;
  }
}

}
