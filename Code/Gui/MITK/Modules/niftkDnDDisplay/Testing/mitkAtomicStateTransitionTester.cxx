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
, m_NextState(0)
, m_ExpectedState(0)
{
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
AtomicStateTransitionTester<TestObject, TestObjectState>::~AtomicStateTransitionTester()
{
  ObserverMap::const_iterator it = m_ObserverTags.begin();
  ObserverMap::const_iterator observerTagsEnd = m_ObserverTags.end();
  for ( ; it != observerTagsEnd; ++it)
  {
    itk::Object::Pointer object = it->first;
    unsigned long observerTag = it->second;
    object->RemoveObserver(observerTag);
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Clear()
{
  m_InitialState = TestObjectState::New(m_TestObject);
  m_NextState = 0;
  m_ExpectedState = 0;
  Superclass::Clear();
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::ProcessEvent(const itk::Object* caller, const itk::EventObject& event)
{
  Superclass::ProcessEvent(caller, event);
  this->OnSignalReceived();
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::OnSignalReceived()
{
  typename TestObjectState::Pointer newState = TestObjectState::New(m_TestObject);

  if (m_NextState.IsNull())
  {
    if (newState == m_InitialState)
    {
      MITK_INFO << "ERROR: Illegal state. Signal received but the state of the object has not changed.";
      MITK_INFO << this;
      QFAIL("Illegal state. Signal received but the state of the object has not changed.");
    }
    else if (m_ExpectedState.IsNotNull() && newState != m_ExpectedState)
    {
      MITK_INFO << "ERROR: Illegal state. The new state of the object is not equal to the expected state.";
      MITK_INFO << this;
      MITK_INFO << "New, illegal state:";
      MITK_INFO << newState;
      QFAIL("Illegal state. The new state of the object is not equal to the expected state.");
    }
    m_NextState = newState;
  }
  else if (newState != m_NextState)
  {
    MITK_INFO << "ERROR: Illegal state. The state of the object has already changed once.";
    MITK_INFO << this;
    MITK_INFO << "New, illegal state:";
    MITK_INFO << newState;
    QFAIL("Illegal state. The state of the object has already changed once.");
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  os << indent << "Initial state: " << std::endl;
  os << indent << m_InitialState;

  if (m_ExpectedState.IsNotNull())
  {
    os << indent << "Expected state: " << std::endl;
    os << indent << m_ExpectedState;
  }

  if (m_NextState.IsNotNull())
  {
    os << indent << "Next state: " << std::endl;
    os << indent << m_NextState;
    os << indent << "Signals:" << std::endl;
    Superclass::PrintSelf(os, indent);
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(itk::Object::Pointer object, const itk::EventObject& event)
{
  unsigned long observerTag = object->AddObserver(event, this);
  m_ObserverTags.insert(ObserverMap::value_type(object, observerTag));
}

template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(const itk::EventObject& event)
{
  this->Connect(m_TestObject, event);
}

}
