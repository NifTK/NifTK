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
#include <QMetaObject>
#include <QMetaMethod>

/// Note that the is_pointer struct is part of the Cxx11 standard.

template<typename T>
struct is_pointer
{
  static const bool value = false;
};

template<typename T>
struct is_pointer<T*>
{
  static const bool value = true;
};


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
  /// If the tested object is a QObject then let us discover its public signals and connect this object to them.
  if (::is_pointer<TestObject>::value)
  {
    const QObject* qTestObject = dynamic_cast<const QObject*>(testObject);

    if (qTestObject)
    {
      const QMetaObject* metaObject = qTestObject->metaObject();
      int methodCount = metaObject->methodCount();
      for (int i = 0; i < methodCount; ++i)
      {
        QMetaMethod metaMethod = metaObject->method(i);
        if (metaMethod.methodType() == QMetaMethod::Signal
            && metaMethod.access() == QMetaMethod::Public)
        {
//          QMetaObject::connect(qTestObject, methodCount, this, callbackSlotIndex);
          this->Connect(qTestObject, metaMethod.signature());
        }
      }
    }
  }

  /// We collect the ITK signals using an ItkSignalCollector object.
  m_ItkSignalCollector = mitk::ItkSignalCollector::New();
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
AtomicStateTransitionTester<TestObject, TestObjectState>::~AtomicStateTransitionTester()
{
  std::vector<QtSignalNotifier*>::const_iterator qtSignalNotifiersIt = m_QtSignalNotifiers.begin();
  std::vector<QtSignalNotifier*>::const_iterator qtSignalNotifiersEnd = m_QtSignalNotifiers.end();
  for ( ; qtSignalNotifiersIt != qtSignalNotifiersEnd; ++qtSignalNotifiersIt)
  {
    delete *qtSignalNotifiersIt;
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Clear()
{
  m_InitialState = TestObjectState::New(m_TestObject);
  m_NextState = 0;
  m_ExpectedState = 0;

  m_ItkSignalCollector->Clear();
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::CheckState()
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
    os << indent << "ITK signals:" << std::endl;
    os << indent << m_ItkSignalCollector;
  }
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(itk::Object* object, const itk::EventObject& event)
{
  ItkSignalNotifier::Pointer itkSignalNotifier = ItkSignalNotifier::New(this, object, event);
  m_ItkSignalNotifiers.push_back(itkSignalNotifier);
  itkSignalNotifier->Register();
//  m_ItkSignalCollector->Connect(object, event);
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(const itk::EventObject& event)
{
  this->Connect(m_TestObject, event);
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(const QObject* object, const char* signal)
{
  QtSignalNotifier* qtSignalNotifier = new QtSignalNotifier(this, object, signal);
  m_QtSignalNotifiers.push_back(qtSignalNotifier);
}


//-----------------------------------------------------------------------------
template <class TestObject, class TestObjectState>
void AtomicStateTransitionTester<TestObject, TestObjectState>::Connect(const char* signal)
{
  if (::is_pointer<TestObject>::value)
  {
    const QObject* qTestObject = dynamic_cast<const QObject*>(m_TestObject);
    if (qTestObject)
    {
      this->Connect(qTestObject, signal);
    }
  }
}

}
