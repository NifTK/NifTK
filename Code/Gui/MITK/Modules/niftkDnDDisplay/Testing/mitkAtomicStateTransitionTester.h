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

#include <mitkItkSignalCollector.h>
#include <mitkQtSignalCollector.h>

#include <QObject>

#include <mitkCommon.h>

#include <map>

namespace mitk
{


//-----------------------------------------------------------------------------
class QtSignalListener
{
public:
  virtual void OnQtSignalReceived(const QObject* object, const char* signal) = 0;
};


//-----------------------------------------------------------------------------
class QtSignalNotifier : public QObject
{
  Q_OBJECT

public:
  QtSignalNotifier(QtSignalListener* signalListener, const QObject* object, const char* signal)
  {
    m_QtSignalListener = signalListener;
    m_Object = object;
    m_Signal = signal;
    this->connect(object, signal, SLOT(OnQtSignalReceived));
  }

  virtual ~QtSignalNotifier()
  {
    QObject::disconnect(m_Object, m_Signal, this, SLOT(OnQtSignalReceived));
  }

private slots:
  virtual void OnQtSignalReceived()
  {
    m_QtSignalListener->OnQtSignalReceived(m_Object, m_Signal);
  }

private:
  QtSignalListener* m_QtSignalListener;
  const QObject* m_Object;
  const char* m_Signal;
};


//-----------------------------------------------------------------------------
class ItkSignalListener
{
public:
  virtual void OnItkSignalReceived(const itk::Object* object, const itk::EventObject& event) = 0;
};


//-----------------------------------------------------------------------------
class ItkSignalNotifier : public itk::Command
{
public:
  ItkSignalNotifier(ItkSignalListener* signalListener, itk::Object* object, const itk::EventObject& event)
  {
    m_ItkSignalListener = signalListener;
    m_Object = object;
//    m_Event = event.MakeObject();
    m_ObserverTag = object->AddObserver(event, this);
  }

  virtual ~ItkSignalNotifier()
  {
    m_Object->RemoveObserver(m_ObserverTag);
//    delete m_Event;
  }

private:

  /// \brief Called when the event happens to the caller.
  virtual void Execute(itk::Object* object, const itk::EventObject& event)
  {
    m_ItkSignalListener->OnItkSignalReceived(object, event);
  }

  /// \brief Called when the event happens to the caller.
  virtual void Execute(const itk::Object* object, const itk::EventObject& event)
  {
    m_ItkSignalListener->OnItkSignalReceived(object, event);
  }

private:
  ItkSignalListener* m_ItkSignalListener;
  itk::Object* m_Object;
//  itk::EventObject* m_Event;
  unsigned long m_ObserverTag;
};


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
class AtomicStateTransitionTester : public itk::Object, private mitk::ItkSignalListener, private mitk::QtSignalListener
{
public:

  mitkClassMacro(AtomicStateTransitionTester, itk::Object);
  mitkNewMacro1Param(AtomicStateTransitionTester, TestObject);

  typedef mitk::ItkSignalCollector::Signal ItkSignal;
  typedef mitk::ItkSignalCollector::Signals ItkSignals;

  typedef std::pair<QObject*, const char*> QtSignal;
  typedef std::vector<QtSignal> QtSignals;

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

  /// \brief Connects this object to the specified events of itkObject.
  /// The consistency of the test object will be checked after these ITK events.
  void Connect(itk::Object* itkObject, const itk::EventObject& event);

  /// \brief Connects this object to the specified events of the test object.
  /// The consistency of the test object will be checked after these ITK events.
  /// The function assumes that the test object is an itk::Object.
  void Connect(const itk::EventObject& event);

  void Connect(const QObject* qObject, const char* signal);

  /// \brief Connects this object to the specified signals of the test object.
  /// The consistency of the test object will be checked after these Qt signals.
  /// The function assumes that the test object is a QObject.
  void Connect(const char* signal);

  const ItkSignals& GetItkSignals() const
  {
    return m_ItkSignalCollector->GetSignals();
  }

  const QtSignals& GetQtSignals() const
  {
    return m_ItkSignalCollector->GetSignals();
  }

protected:

  /// \brief Constructs an AtomicStateTransitionTester object.
  AtomicStateTransitionTester(TestObject testObject);

  /// \brief Destructs an AtomicStateTransitionTester object.
  virtual ~AtomicStateTransitionTester();

  /// \brief Handler for the ITK signals. Checks the consistency of the test object.
  virtual void OnItkSignalReceived(const itk::Object* object, const itk::EventObject& event)
  {
    m_ItkSignalCollector->ProcessEvent(object, event);
    this->CheckState();
  }

  /// \brief Handler for the Qt signals. Checks the consistency of the test object.
  virtual void OnQtSignalReceived(const QObject* object, const char* signal)
  {
    m_QtSignalCollector->OnSignalEmitted(object, signal);
    this->CheckState();
  }

  /// \brief Prints the collected signals to the given stream or to the standard output if no stream is given.
  virtual void PrintSelf(std::ostream & os, itk::Indent indent) const;

private:

  /// \brief Called when a signal is received and checks if the state of the object is legal.
  /// The state is illegal in any of the following cases:
  ///
  ///   <li>The state is equal to the initial state. Signals should not be sent out when the
  ///       visible state of the object does not change.
  ///   <li>The new state is not equal to the expected state when the expected state is set.
  ///   <li>The state of the object has changed twice. Signals should be withhold until the
  ///       object has reached its final state, and should be sent out only at that point.
  void CheckState();

  /// \brief The test object whose state consistency is being checked.
  TestObject m_TestObject;

  /// \brief The initial state of the test object.
  typename TestObjectState::Pointer m_InitialState;

  /// \brief The next state of the test object.
  typename TestObjectState::Pointer m_NextState;

  /// \brief The expected state of the test object.
  typename TestObjectState::Pointer m_ExpectedState;

  std::vector<ItkSignalNotifier*> m_ItkSignalNotifiers;

  std::vector<QtSignalNotifier*> m_QtSignalNotifiers;

  mitk::ItkSignalCollector::Pointer m_ItkSignalCollector;

  mitk::QtSignalCollector::Pointer m_QtSignalCollector;

};

}

#endif
