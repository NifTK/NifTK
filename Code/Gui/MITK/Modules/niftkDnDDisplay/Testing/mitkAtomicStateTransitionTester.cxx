/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkAtomicStateTransitionTester.h"

#include <QTest>

namespace mitk
{

//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
AtomicStateTransitionTester<TestObject, TestObjectState>::AtomicStateTransitionTester(TestObject testObject)
: Superclass()
, m_TestObject(testObject)
, m_InitialState(TestObjectState::New(testObject))
, m_NewState(0)
, m_ExpectedState(0)
, m_SignalNumberAtFirstTransition(-1)
{
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
AtomicStateTransitionTester<TestObject, TestObjectState>::~AtomicStateTransitionTester()
{
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Clear()
{
  m_InitialState = TestObjectState::New(m_TestObject);
  m_NewState = 0;
  m_ExpectedState = 0;
  m_SignalNumberAtFirstTransition = -1;
  Superclass::Clear();
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::ProcessEvent(const itk::Object* caller, const itk::EventObject& event)
{
  if (m_InitialState.IsNotNull())
  {
    typename TestObjectState::Pointer newState = TestObjectState::New(m_TestObject);

    if (m_NewState.IsNull())
    {
      if (newState != m_InitialState)
      {
        if (m_ExpectedState.IsNotNull() && newState != m_ExpectedState)
        {
          MITK_INFO << "Illegal state transition happened. Unexpected state.";
          MITK_INFO << "History of events:";
          MITK_INFO << this;
          MITK_INFO << "";
          MITK_INFO << "Current signal:";
          MITK_INFO << this->GetSignals().size() << ": " << ((void*) caller) << ": " << event.GetEventName() << std::endl;
          MITK_INFO << "";
          MITK_INFO << "State after the current (illegal) state transition:";
          MITK_INFO << newState;
          MITK_INFO << "";
          MITK_INFO << "Expected state:";
          MITK_INFO << m_ExpectedState;
          QFAIL("Illegal state transition happened. Unexpected state.");
        }
        m_NewState = newState;
        m_SignalNumberAtFirstTransition = this->GetSignals().size();
      }
    }
    else
    {
      if (newState != m_NewState)
      {
        MITK_INFO << "Illegal state transition happened. Unexpected state transition.";
        MITK_INFO << "History of events:";
        MITK_INFO << this;
        MITK_INFO << "";
        MITK_INFO << "Current signal:";
        MITK_INFO << this->GetSignals().size() << ": " << ((void*) caller) << ": " << event.GetEventName() << std::endl;
        MITK_INFO << "";
        MITK_INFO << "State after the current (illegal) state transition:";
        MITK_INFO << newState;
        QFAIL("Illegal state transition happened.");
      }
    }

    Superclass::ProcessEvent(caller, event);
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  os << indent << "Initial state: " << std::endl;
  os << indent << m_InitialState;

  const Signals& signals_ = this->GetSignals();

  Signals::const_iterator it = signals_.begin();
  Signals::const_iterator signalsEnd = signals_.end();
  int i = 0;
  for ( ; it != signalsEnd && i != m_SignalNumberAtFirstTransition; ++it, ++i)
  {
    os << indent << i << ": " << ((void*) it->first) << ": " << it->second->GetEventName() << std::endl;
  }

  if (it != signalsEnd)
  {
    os << indent << i << ": " << ((void*) it->first) << ": " << it->second->GetEventName() << std::endl;
    os << indent << "New state: " << std::endl;
    os << indent << m_NewState;

    for (++it, ++i; it != signalsEnd; ++it, ++i)
    {
      os << indent << i << ": " << ((void*) it->first) << ": " << it->second->GetEventName() << std::endl;
    }
  }

  os << indent << "Expected state: " << std::endl;
  os << indent << m_ExpectedState;
}

}
