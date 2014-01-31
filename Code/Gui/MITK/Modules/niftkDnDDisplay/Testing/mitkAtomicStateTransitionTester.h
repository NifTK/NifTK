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

  /// \brief Gets the new state of the test object.
  itkGetConstMacro(NewState, typename TestObjectState::Pointer);

  /// \brief Gets the expected state of the test object.
  itkGetConstMacro(ExpectedState, typename TestObjectState::Pointer);

  /// \brief Sets the expected state of the test object.
  itkSetMacro(ExpectedState, typename TestObjectState::Pointer);

  /// \brief Clears the collected signals and resets the states.
  virtual void Clear();

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

  /// \brief The new state of the test object.
  typename TestObjectState::Pointer m_NewState;

  /// \brief The expected state of the test object.
  typename TestObjectState::Pointer m_ExpectedState;

  /// \brief Number of collected signals when the state transition happened.
  Signals::size_type m_SignalNumberAtFirstTransition;

};

}

#endif
