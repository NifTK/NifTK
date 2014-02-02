/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkAtomicStateTransitionTester_h
#define __mitkAtomicStateTransitionTester_h

#include <mitkSignalCollector.h>

#include <mitkCommon.h>

#include <map>

namespace mitk
{

/// \class AtomicStateTransitionTester
///
/// Test class to ensure the atomic transition from one object state to another.
///
/// The state of the tested object must change at most once during the execution
/// of a public function.
///
/// Pattern of use:
///
///   typedef mitk::AtomicStateTransitionTester<Viewer, ViewerState> ViewerStateTester;
///   ViewerStateTester::Pointer viewerStateTester = ViewerStateTester::New(viewer);
///   const mitk::SignalCollector::Signals& viewerSignals = viewerStateTester->GetSignals();
///
///   ... add viewerStateTester to the observers of ITK events sent out from this object.
///
///   viewer->SomePublicFunction(...);
///
///   ... check the contents of viewerSignals if needed.
///
///   viewerStateTester->Clear();
///   viewer->AnotherPublicFunction(...);
///   ...
///
///   ... remove viewerStateTester from the observers of ITK events sent out from this object.
///
template <class TestObject, class TestObjectState>
class AtomicStateTransitionTester : public mitk::SignalCollector
{
public:
  mitkClassMacro(AtomicStateTransitionTester, mitk::SignalCollector);
  mitkNewMacro1Param(AtomicStateTransitionTester, TestObject);

  /// \brief Gets the object whose state consistency is being tested.
  itkGetConstMacro(TestObject, TestObject);

  /// \brief Gets the initial state of the test object.
  itkGetConstMacro(InitialState, typename TestObjectState::Pointer);

  /// \brief Gets the next state of the test object.
  itkGetConstMacro(NextState, typename TestObjectState::Pointer);

  /// \brief Gets the expected state of the test object.
  itkGetConstMacro(ExpectedState, typename TestObjectState::Pointer);

  /// \brief Sets the expected state of the test object.
  itkSetMacro(ExpectedState, typename TestObjectState::Pointer);

  /// \brief Clears the collected signals and resets the states.
  virtual void Clear();

  void Connect(itk::Object::Pointer itkObject, const itk::EventObject& event);
  void Connect(const itk::EventObject& event);

public slots:

  /// \brief Called when a signal is received and checks if the state of the object is legal.
  /// The state is illegal in any of the following cases:
  ///
  ///   <li>The state is equal to the initial state. Signals should not be sent out when the
  ///       visible state of the object does not change.
  ///   <li>The new state is not equal to the expected state when the expected state is set.
  ///   <li>The state of the object has changed twice. Signals should be withhold until the
  ///       object has reached its final state, and should be sent out only at that point.
  void OnSignalReceived();

protected:

  /// \brief Constructs an AtomicStateTransitionTester object.
  AtomicStateTransitionTester(TestObject testObject);

  /// \brief Destructs an AtomicStateTransitionTester object.
  virtual ~AtomicStateTransitionTester();

  /// \brief Called when the event happens to the caller.
  virtual void ProcessEvent(const itk::Object* object, const itk::EventObject& event);

  /// \brief Prints the collected signals to the given stream or to the standard output if no stream is given.
  virtual void PrintSelf(std::ostream & os, itk::Indent indent) const;

private:

  /// \brief The test object whose state consistency is being checked.
  TestObject m_TestObject;

  /// \brief The initial state of the test object.
  typename TestObjectState::Pointer m_InitialState;

  /// \brief The next state of the test object.
  typename TestObjectState::Pointer m_NextState;

  /// \brief The expected state of the test object.
  typename TestObjectState::Pointer m_ExpectedState;

  typedef std::multimap<itk::Object::Pointer, unsigned long> ObserverMap;
  ObserverMap m_ObserverTags;

};

}

#endif
