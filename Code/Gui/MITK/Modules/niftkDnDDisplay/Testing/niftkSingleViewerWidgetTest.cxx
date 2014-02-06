/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerWidgetTest.h"

#include <QApplication>
#include <QSignalSpy>
#include <QTest>
#include <QTextStream>

#include <mitkGlobalInteraction.h>
#include <mitkIOUtil.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkTestingMacros.h>

#include <QmitkRenderingManagerFactory.h>
#include <QmitkApplicationCursor.h>

#include <mitkMIDASOrientationUtils.h>
#include <mitkNifTKCoreObjectFactory.h>
#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>

#include <mitkItkSignalCollector.cxx>
#include <mitkQtSignalCollector.cxx>
#include <niftkSingleViewerWidgetState.h>


class niftkSingleViewerWidgetTestClassPrivate
{
public:
  std::string FileName;
  mitk::DataStorage::Pointer DataStorage;
  mitk::RenderingManager::Pointer RenderingManager;

  mitk::DataNode::Pointer ImageNode;
  mitk::Image* Image;
  mitk::Vector3D ExtentsInVxInWorldCoordinateOrder;
  mitk::Vector3D SpacingInWorldCoordinateOrder;

  niftkSingleViewerWidget* Viewer;
  niftkMultiViewerVisibilityManager* VisibilityManager;

  niftkSingleViewerWidgetTestClass::ViewerStateTester::Pointer StateTester;

  bool InteractiveMode;
};


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::niftkSingleViewerWidgetTestClass()
: QObject()
, d_ptr(new niftkSingleViewerWidgetTestClassPrivate())
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->ImageNode = 0;
  d->Image = 0;
  d->ExtentsInVxInWorldCoordinateOrder.Fill(0.0);
  d->SpacingInWorldCoordinateOrder.Fill(1.0);
  d->Viewer = 0;
  d->VisibilityManager = 0;
  d->InteractiveMode = false;
}


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::~niftkSingleViewerWidgetTestClass()
{
}


// --------------------------------------------------------------------------
std::string niftkSingleViewerWidgetTestClass::GetFileName() const
{
  Q_D(const niftkSingleViewerWidgetTestClass);
  return d->FileName;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::SetFileName(const std::string& fileName)
{
  Q_D(niftkSingleViewerWidgetTestClass);
  d->FileName = fileName;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::GetInteractiveMode() const
{
  Q_D(const niftkSingleViewerWidgetTestClass);
  return d->InteractiveMode;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::SetInteractiveMode(bool interactiveMode)
{
  Q_D(niftkSingleViewerWidgetTestClass);
  d->InteractiveMode = interactiveMode;
}


// --------------------------------------------------------------------------
QPoint niftkSingleViewerWidgetTestClass::GetPointAtCursorPosition(QmitkRenderWindow *renderWindow, const mitk::Vector2D& cursorPosition)
{
  QRect rect = renderWindow->rect();
  double x = cursorPosition[0] * rect.width();
  double y = (1.0 - cursorPosition[1]) * rect.height();
  return QPoint(x, y);
}


// --------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidgetTestClass::GetCursorPositionAtPoint(QmitkRenderWindow *renderWindow, const QPoint& point)
{
  QRect rect = renderWindow->rect();
  mitk::Vector2D cursorPosition;
  cursorPosition[0] = double(point.x()) / rect.width();
  cursorPosition[1] = 1.0 - double(point.y()) / rect.height();
  return cursorPosition;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::Equals(const mitk::Point3D& selectedPosition1, const mitk::Point3D& selectedPosition2)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  for (int i = 0; i < 3; ++i)
  {
    double tolerance = d->SpacingInWorldCoordinateOrder[i] / 2.0;
    if (std::abs(selectedPosition1[i] - selectedPosition2[i]) > tolerance)
    {
      return false;
    }
  }

  return true;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::Equals(const mitk::Vector2D& cursorPosition1, const mitk::Vector2D& cursorPosition2, double tolerance)
{
  return std::abs(cursorPosition1[0] - cursorPosition2[0]) <= tolerance && std::abs(cursorPosition1[1] - cursorPosition2[1]) <= tolerance;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::Equals(const std::vector<mitk::Vector2D>& cursorPositions1, const std::vector<mitk::Vector2D>& cursorPositions2, double tolerance)
{
  return cursorPositions1.size() == size_t(3)
      && cursorPositions2.size() == size_t(3)
      && Self::Equals(cursorPositions1[0], cursorPositions2[0], tolerance)
      && Self::Equals(cursorPositions1[1], cursorPositions2[1], tolerance)
      && Self::Equals(cursorPositions1[2], cursorPositions2[2], tolerance);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::initTestCase()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  // Need to load images, specifically using MIDAS/DRC object factory.
  ::RegisterNifTKCoreObjectFactory();

  mitk::GlobalInteraction* globalInteraction =  mitk::GlobalInteraction::GetInstance();
  globalInteraction->Initialize("global");
  globalInteraction->GetStateMachineFactory()->LoadBehaviorString(mitk::DnDDisplayStateMachine::STATE_MACHINE_XML);

  /// Create and register RenderingManagerFactory for this platform.
  static QmitkRenderingManagerFactory qmitkRenderingManagerFactory;
  Q_UNUSED(qmitkRenderingManagerFactory);

  /// Create one instance
  static QmitkApplicationCursor globalQmitkApplicationCursor;
  Q_UNUSED(globalQmitkApplicationCursor);

  d->DataStorage = mitk::StandaloneDataStorage::New();

  d->RenderingManager = mitk::RenderingManager::GetInstance();
  d->RenderingManager->SetDataStorage(d->DataStorage);

  /// Disable VTK warnings. For some reason, they appear using these tests, but
  /// not with the real application. We simply suppress them here.
  vtkObject::GlobalWarningDisplayOff();

  std::vector<std::string> files;
  files.push_back(d->FileName);

  mitk::IOUtil::LoadFiles(files, *(d->DataStorage.GetPointer()));
  mitk::DataStorage::SetOfObjects::ConstPointer allImages = d->DataStorage->GetAll();
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(allImages->size(), 1), ".. Test image loaded.");

  d->ImageNode = (*allImages)[0];

  d->VisibilityManager = new niftkMultiViewerVisibilityManager(d->DataStorage);
  d->VisibilityManager->SetInterpolationType(DNDDISPLAY_CUBIC_INTERPOLATION);
  d->VisibilityManager->SetDefaultWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

  d->Image = dynamic_cast<mitk::Image*>(d->ImageNode->GetData());
  mitk::GetExtentsInVxInWorldCoordinateOrder(d->Image, d->ExtentsInVxInWorldCoordinateOrder);
  mitk::GetSpacingInWorldCoordinateOrder(d->Image, d->SpacingInWorldCoordinateOrder);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::cleanupTestCase()
{
  Q_D(niftkSingleViewerWidgetTestClass);
  delete d->VisibilityManager;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::init()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer = new niftkSingleViewerWidget(0, d->RenderingManager);
  d->Viewer->SetDataStorage(d->DataStorage);
  d->Viewer->setObjectName(tr("niftkSingleViewerWidget"));

//  QColor backgroundColour("black");
  QColor backgroundColour("#fffaf0");
  d->Viewer->SetDirectionAnnotationsVisible(true);
  d->Viewer->SetBackgroundColor(backgroundColour);
  d->Viewer->SetShow3DWindowIn2x2WindowLayout(true);
  d->Viewer->SetRememberSettingsPerWindowLayout(true);
  d->Viewer->SetDisplayInteractionsEnabled(true);
  d->Viewer->SetCursorPositionBinding(true);
  d->Viewer->SetScaleFactorBinding(true);
  d->Viewer->SetDefaultSingleWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->Viewer->SetDefaultMultiWindowLayout(WINDOW_LAYOUT_ORTHO);

  d->VisibilityManager->connect(d->Viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

  d->VisibilityManager->RegisterViewer(d->Viewer);
  d->VisibilityManager->SetAllNodeVisibilityForViewer(0, false);

  d->Viewer->resize(1024, 1024);
  d->Viewer->show();

  QTest::qWaitForWindowShown(d->Viewer);

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = d->ImageNode;

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  this->dropNodes(axialWindow, nodes);

  d->Viewer->SetCursorVisible(true);

  /// Create a state tester that works for all of the test functions.

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  d->StateTester = ViewerStateTester::New(d->Viewer);

  d->StateTester->Connect(focusManager, mitk::FocusEvent());
  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  d->StateTester->Connect(axialSnc, geometrySliceEvent);
  d->StateTester->Connect(sagittalSnc, geometrySliceEvent);
  d->StateTester->Connect(coronalSnc, geometrySliceEvent);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::cleanup()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Release the pointer so that the desctructor is be called.
  d->StateTester = 0;

  if (d->InteractiveMode)
  {
    QEventLoop loop;
    loop.connect(d->Viewer, SIGNAL(destroyed()), SLOT(quit()));
    loop.exec();
  }

  d->VisibilityManager->DeRegisterAllViewers();

  delete d->Viewer;
  d->Viewer = 0;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::dropNodes(QWidget* window, const std::vector<mitk::DataNode*>& nodes)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QMimeData* mimeData = new QMimeData;
  QString dataNodeAddresses("");
  for (int i = 0; i < nodes.size(); ++i)
  {
    long dataNodeAddress = reinterpret_cast<long>(nodes[i]);
    QTextStream(&dataNodeAddresses) << dataNodeAddress;

    if (i != nodes.size() - 1)
    {
      QTextStream(&dataNodeAddresses) << ",";
    }
  }
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toAscii()));
  QStringList types;
  types << "application/x-mitk-datanodes";
  QDropEvent dropEvent(window->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  dropEvent.acceptProposedAction();
  QApplication::instance()->sendEvent(window, &dropEvent);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testViewer()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Tests if the viewer has been successfully created.
  QVERIFY(d->Viewer);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetOrientation()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  ViewerState::Pointer state = ViewerState::New(d->Viewer);
  d->StateTester->SetExpectedState(state);

  MIDASOrientation orientation = d->Viewer->GetOrientation();

  /// The default window layout was set to coronal in the init() function.
  QCOMPARE(orientation, MIDAS_ORIENTATION_CORONAL);
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals().size(), size_t(0));
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetSelectedPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Make sure that the state does not change and no signal is sent out.
  ViewerState::Pointer state = ViewerState::New(d->Viewer);
  d->StateTester->SetExpectedState(state);

  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals().size(), size_t(0));

  mitk::Point3D centre = d->Image->GetGeometry()->GetCenter();

  for (int i = 0; i < 3; ++i)
  {
    double distanceFromCentre = std::abs(selectedPosition[i] - centre[i]);
    if (static_cast<int>(d->ExtentsInVxInWorldCoordinateOrder[i]) % 2 == 0)
    {
      /// If the number of slices is an even number then the selected position
      /// must be a half voxel far from the centre, either way.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(std::abs(distanceFromCentre - d->SpacingInWorldCoordinateOrder[i] / 2.0) < 0.001);
    }
    else
    {
      /// If the number of slices is an odd number then the selected position
      /// must be exactly at the centre position.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(distanceFromCentre < 0.001);
    }
  }
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  /// Register to listen to SliceNavigators, slice changed events.
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  mitk::Point3D initialPosition = d->Viewer->GetSelectedPosition();
  mitk::Point3D newPosition = initialPosition;
  newPosition[0] += 2 * d->SpacingInWorldCoordinateOrder[0];

  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(1));

  d->StateTester->Clear();

  newPosition[1] += 2 * d->SpacingInWorldCoordinateOrder[1];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(1));

  d->StateTester->Clear();

  newPosition[2] += 2 * d->SpacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(1));

  d->StateTester->Clear();

  newPosition[0] -= 3 * d->SpacingInWorldCoordinateOrder[0];
  newPosition[1] -= 3 * d->SpacingInWorldCoordinateOrder[1];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(2));

  d->StateTester->Clear();

  newPosition[0] -= 4 * d->SpacingInWorldCoordinateOrder[0];
  newPosition[2] -= 4 * d->SpacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(2));

  d->StateTester->Clear();

  newPosition[1] += 5 * d->SpacingInWorldCoordinateOrder[1];
  newPosition[2] += 5 * d->SpacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), size_t(2));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetCursorPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Note that the cursor positions of a render window are first initialised
  /// when the render window gets visible.

  mitk::Vector2D centrePosition;
  centrePosition.Fill(0.5);

  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centrePosition, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();

  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centrePosition, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  d->StateTester->Clear();

  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centrePosition, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetCursorPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  const char* cursorPositionChangedSignal = SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, MIDASOrientation, const mitk::Vector2D&));

  mitk::Vector2D axialCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);
  mitk::Vector2D sagittalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
  mitk::Vector2D coronalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  coronalCursorPosition[0] = 0.4;
  coronalCursorPosition[1] = 0.6;

  d->Viewer->SetCursorPosition(MIDAS_ORIENTATION_CORONAL, coronalCursorPosition);

  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), axialCursorPosition, 0.01));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), sagittalCursorPosition, 0.01));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), coronalCursorPosition, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals(cursorPositionChangedSignal).size() == 1);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetCursorPositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Note that the cursor positions of a render window are first initialised
  /// when the render window gets visible.

  mitk::Vector2D centrePosition;
  centrePosition.Fill(0.5);
  std::vector<mitk::Vector2D> centrePositions(3);
  std::fill(centrePositions.begin(), centrePositions.end(), centrePosition);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  d->StateTester->Clear();

  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPositions(), centrePositions, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetCursorPositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);
  /// TODO
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// The default window layout was set to coronal in the init() function.
  QCOMPARE(d->Viewer->GetWindowLayout(), WINDOW_LAYOUT_CORONAL);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetSelectedRenderWindow()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  /// The default window layout was set to coronal in the init() function.
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), coronalWindow);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedRenderWindow()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  QVERIFY(coronalWindow->hasFocus());
  QCOMPARE(focusedRenderer, coronalWindow->GetRenderer());

  QCOMPARE(d->Viewer->IsSelected(), true);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testFocusedRenderer()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  QVERIFY(coronalWindow->hasFocus());
  QCOMPARE(focusedRenderer, coronalWindow->GetRenderer());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  /// Disabled because the cursor state of the sagittal window will be different,
  /// since it will be initialised just now.
//  niftkSingleViewerWidgetState::Pointer expectedState = niftkSingleViewerWidgetState::New(d->Viewer);
//  expectedState->SetOrientation(MIDAS_ORIENTATION_SAGITTAL);
//  expectedState->SetSelectedRenderWindow(sagittalWindow);
//  expectedState->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
//  d->StateTester->SetExpectedState(expectedState);

  /// The default layout was set to coronal in the init() function.
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  QVERIFY(sagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), sagittalWindow);
  QVERIFY(focusManager->GetFocused() == sagittalWindow->GetRenderer());

  mitk::FocusEvent focusEvent;
  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, focusEvent).size(), size_t(1));
  QVERIFY(d->StateTester->GetItkSignals(axialSnc, geometrySliceEvent).size() <= 1);
  QVERIFY(d->StateTester->GetItkSignals(sagittalSnc, geometrySliceEvent).size() <= 1);
  QVERIFY(d->StateTester->GetItkSignals(coronalSnc, geometrySliceEvent).size() <= 1);

//  d->StateTester->Clear();

//  QRect rect = sagittalWindow->rect();
//  QPoint centre = rect.center();
//  QPoint bottomLeftCorner = rect.bottomLeft();
//  QPoint aPosition((bottomLeftCorner.x() + centre.x()) / 2, (bottomLeftCorner.y() + centre.y()) / 2);
//  QTest::mouseClick(sagittalWindow, Qt::LeftButton, Qt::NoModifier, aPosition);

//  QVERIFY(sncAndFocusSignals.size() <= 3);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testRememberSelectedPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  mitk::Point3D selectedPosition;
  selectedPosition[0] = 100.0;
  selectedPosition[1] = -50.0;
  selectedPosition[2] = -100.0;

  d->Viewer->SetSelectedPosition(selectedPosition);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  mitk::Point3D newPosition = d->Viewer->GetSelectedPosition();

  QVERIFY(this->Equals(newPosition, selectedPosition));

  mitk::Vector2D coronalCursorPosition;
  coronalCursorPosition[0] = 0.4;
  coronalCursorPosition[1] = 0.6;
  QPoint pointAtCoronalCursorPosition = this->GetPointAtCursorPosition(coronalWindow, coronalCursorPosition);

  d->StateTester->Clear();
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, pointAtCoronalCursorPosition);

  mitk::Point3D newPosition2 = d->Viewer->GetSelectedPosition();
  mitk::Vector2D newCoronalCursorPosition2 = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QVERIFY(this->Equals(coronalCursorPosition, newCoronalCursorPosition2));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  mitk::Point3D newPosition3 = d->Viewer->GetSelectedPosition();
  QVERIFY(this->Equals(newPosition3, newPosition2));
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectPositionByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  QPoint centre = coronalWindow->rect().center();
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, centre);

  d->StateTester->Clear();

  mitk::Point3D lastPosition = d->Viewer->GetSelectedPosition();

  QPoint newPoint = centre;
  newPoint.rx() += 30;
  std::vector<mitk::Vector2D> expectedCursorPositions = d->Viewer->GetCursorPositions();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL] = Self::GetCursorPositionAtPoint(coronalWindow, newPoint);

  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  mitk::Point3D newPosition = d->Viewer->GetSelectedPosition();
  std::vector<mitk::Vector2D> newCursorPositions = d->Viewer->GetCursorPositions();
  QVERIFY(newPosition[0] != lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QCOMPARE(newPosition[2], lastPosition[2]);
  QVERIFY(Self::Equals(newCursorPositions, expectedCursorPositions));
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));

  d->StateTester->Clear();

  lastPosition = newPosition;

  newPoint.ry() += 20;
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  newPosition = d->Viewer->GetSelectedPosition();
  QCOMPARE(newPosition[0], lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QVERIFY(newPosition[2] != lastPosition[1]);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));

  d->StateTester->Clear();

  lastPosition = d->Viewer->GetSelectedPosition();

  newPoint.rx() -= 40;
  newPoint.ry() += 50;
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  newPosition = d->Viewer->GetSelectedPosition();
  QVERIFY(newPosition[0] != lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QVERIFY(newPosition[2] != lastPosition[2]);
  QCOMPARE(d->StateTester->GetItkSignals(axialSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(sagittalSnc).size(), size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(coronalSnc).size(), size_t(0));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectRenderWindowByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();

  QCOMPARE(d->Viewer->IsSelected(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), coronalWindow);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  QCOMPARE(d->Viewer->IsSelected(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), coronalWindow);

  d->StateTester->Clear();

  QPoint centre = sagittalWindow->rect().center();
  QTest::mouseClick(sagittalWindow, Qt::LeftButton, Qt::NoModifier, centre);

  QCOMPARE(d->Viewer->IsSelected(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), sagittalWindow);
}


// --------------------------------------------------------------------------
static void ShiftArgs(int& argc, char* argv[], int steps = 1)
{
  /// We exploit that there must be a NULL pointer after the arguments.
  /// (Guaranteed by the standard.)
  int i = 1;
  do
  {
    argv[i] = argv[i + steps];
    ++i;
  }
  while (argv[i - 1]);
  argc -= steps;
}


// --------------------------------------------------------------------------
int niftkSingleViewerWidgetTest(int argc, char* argv[])
{
  QApplication app(argc, argv);
  Q_UNUSED(app);

  niftkSingleViewerWidgetTestClass test;

  std::string interactiveModeOption("-i");
  for (int i = 1; i < argc; ++i)
  {
    if (std::string(argv[i]) == interactiveModeOption)
    {
      test.SetInteractiveMode(true);
      ::ShiftArgs(argc, argv);
      break;
    }
  }

  if (argc < 2)
  {
    MITK_INFO << "Missing argument. No image file given.";
    return 1;
  }

  test.SetFileName(argv[1]);
  ::ShiftArgs(argc, argv);

  /// We used the arguments to initialise the test. No arguments is passed
  /// to the Qt test, so that all the test functions are executed.
//  argc = 1;
//  argv[1] = NULL;
  return QTest::qExec(&test, argc, argv);
}
