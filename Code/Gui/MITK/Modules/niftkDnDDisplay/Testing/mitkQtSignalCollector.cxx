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
QtSignalNotifier::QtSignalNotifier(QtSignalListener* signalListener, const QObject* object, const char* signal)
{
  m_QtSignalListener = signalListener;
  m_Object = object;
  m_Signal = QMetaObject::normalizedSignature(signal);
  this->connect(object, signal, SLOT(OnQtSignalReceived));
}


//-----------------------------------------------------------------------------
QtSignalNotifier::~QtSignalNotifier()
{
  QObject::disconnect(m_Object, m_Signal, this, SLOT(OnQtSignalReceived));
}


//-----------------------------------------------------------------------------
void QtSignalNotifier::OnQtSignalReceived()
{
  m_QtSignalListener->OnQtSignalReceived(m_Object, m_Signal);
}


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

  std::vector<QtSignalNotifier*>::const_iterator it = m_SignalNotifiers.begin();
  std::vector<QtSignalNotifier*>::const_iterator signalNotifiersEnd = m_SignalNotifiers.end();
  for ( ; it != signalNotifiersEnd; ++it)
  {
    delete *it;
  }
}


//-----------------------------------------------------------------------------
void QtSignalCollector::Connect(const QObject* object, const char* signal)
{
  QtSignalNotifier* signalNotifier = new QtSignalNotifier(this, object, signal);
  m_SignalNotifiers.push_back(signalNotifier);
}


//-----------------------------------------------------------------------------
void QtSignalCollector::AddListener(QtSignalListener* listener)
{
  std::vector<QtSignalListener*>::iterator it = std::find(m_Listeners.begin(), m_Listeners.end(), listener);
  if (it == m_Listeners.end())
  {
    m_Listeners.push_back(listener);
  }
}


//-----------------------------------------------------------------------------
void QtSignalCollector::RemoveListener(QtSignalListener* listener)
{
  std::vector<QtSignalListener*>::iterator it = std::find(m_Listeners.begin(), m_Listeners.end(), listener);
  if (it != m_Listeners.end())
  {
    m_Listeners.erase(it);
  }
}


//-----------------------------------------------------------------------------
QtSignalCollector::Signals QtSignalCollector::GetSignals(const char* signal) const
{
  return this->GetSignals(0, signal);
}


//-----------------------------------------------------------------------------
QtSignalCollector::Signals QtSignalCollector::GetSignals(QObject* object, const char* signal) const
{
  Signals selectedSignals;
  Signals::const_iterator it = m_Signals.begin();
  Signals::const_iterator signalsEnd = m_Signals.end();
  for ( ; it != signalsEnd; ++it)
  {
    if ((it->first == 0 || it->first == object)
        && signal == 0 || QMetaObject::normalizedSignature(it->second) == QMetaObject::normalizedSignature(signal))
    {
      selectedSignals.push_back(*it);
    }
  }
  return selectedSignals;
}


//-----------------------------------------------------------------------------
void QtSignalCollector::OnQtSignalReceived(const QObject* object, const char* signal)
{
  /// Create a copy of the event as a newly allocated object.
  m_Signals.push_back(Signal(object, signal));

  std::vector<QtSignalListener*>::iterator it = m_Listeners.begin();
  std::vector<QtSignalListener*>::iterator listenersEnd = m_Listeners.end();
  for ( ; it != listenersEnd; ++it)
  {
    (*it)->OnQtSignalReceived(object, signal);
  }
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
