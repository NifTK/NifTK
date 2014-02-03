/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkQtSignalCollector.h"

#include <mitkSliceNavigationController.h>
#include <mitkFocusManager.h>

namespace mitk
{

//-----------------------------------------------------------------------------
QtSignalCollector::QtSignalCollector()
: QObject()
, itk::Object()
{
}


//-----------------------------------------------------------------------------
QtSignalCollector::~QtSignalCollector()
{
  this->Clear();
}


//-----------------------------------------------------------------------------
//void QtSignalCollector::OnEventOccurred(const QObject* object, const QEvent* event)
//{
//}


//-----------------------------------------------------------------------------
void QtSignalCollector::OnSignalEmitted(const QObject* object, const char* signal)
{
  /// Create a copy of the event as a newly allocated object.
  m_Signals.push_back(Signal(object, signal));
}


//-----------------------------------------------------------------------------
const QtSignalCollector::Signals& QtSignalCollector::GetSignals() const
{
  return m_Signals;
}


//-----------------------------------------------------------------------------
void QtSignalCollector::Clear()
{
  /// Destruct the copies of the original events.
  Signals::iterator it = m_Signals.begin();
  Signals::iterator signalsEnd = m_Signals.end();
  for ( ; it != signalsEnd; ++it)
  {
//    delete it->second;
  }

  /// Remove the elements.
  m_Signals.clear();
}


//-----------------------------------------------------------------------------
void QtSignalCollector::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Signals::const_iterator it = m_Signals.begin();
  Signals::const_iterator signalsEnd = m_Signals.end();
  int i = 0;
  for ( ; it != signalsEnd; ++it, ++i)
  {
    os << indent << i << ": " << ((void*) it->first) << ": " << it->second << std::endl;
  }
}

}
