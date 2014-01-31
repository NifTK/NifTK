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

#include <mitkSignalCollector.cxx>

class niftkSingleViewerWidgetTestClassPrivate
{
public:
  std::string FileName;
  mitk::DataStorage::Pointer DataStorage;
  mitk::RenderingManager::Pointer RenderingManager;

  mitk::DataNode::Pointer ImageNode;

  niftkSingleViewerWidget* Viewer;
  niftkMultiViewerVisibilityManager* VisibilityManager;

  bool InteractiveMode;
};


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::niftkSingleViewerWidgetTestClass()
: QObject()
, d_ptr(new niftkSingleViewerWidgetTestClassPrivate())
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->ImageNode = 0;
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

  d->Viewer = new niftkSingleViewerWidget(tr("QmitkRenderWindow"),
    -5, 20, 0, d->RenderingManager, d->DataStorage);
  d->Viewer->setObjectName(tr("niftkSingleViewerWidget"));

//  QColor backgroundColour("black");
  QColor backgroundColour("#fffaf0");
  d->Viewer->SetDirectionAnnotationsVisible(true);
  d->Viewer->SetBackgroundColor(backgroundColour);
  d->Viewer->SetShow3DWindowIn2x2WindowLayout(true);
  d->Viewer->SetRememberSettingsPerWindowLayout(true);
  d->Viewer->SetDisplayInteractionsEnabled(true);
  d->Viewer->SetCursorPositionsBound(true);
  d->Viewer->SetScaleFactorsBound(true);
  d->Viewer->SetDefaultSingleWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->Viewer->SetDefaultMultiWindowLayout(WINDOW_LAYOUT_ORTHO);

  d->VisibilityManager->connect(d->Viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

  d->VisibilityManager->RegisterViewer(d->Viewer);
  d->VisibilityManager->SetAllNodeVisibilityForViewer(0, false);

  d->Viewer->resize(1024, 768);
  d->Viewer->show();

  QTest::qWaitForWindowShown(d->Viewer);

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = d->ImageNode;

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  this->dropNodes(axialWindow, nodes);

  d->Viewer->SetCursorVisible(true);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::cleanup()
{
  Q_D(niftkSingleViewerWidgetTestClass);

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

  /// The default window layout was set to coronal in the init() function.
  QCOMPARE(d->Viewer->GetOrientation(), MIDAS_ORIENTATION_CORONAL);
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
void niftkSingleViewerWidgetTestClass::testFocusedRenderer()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

  /// The default window layout was set to coronal in the init() function.
  /// TODO The focus should be on the selected window.
//  QCOMPARE(focusedRenderer, coronalWindow->GetRenderer());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetSelectedPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  mitk::Image* image = dynamic_cast<mitk::Image*>(d->ImageNode->GetData());
  mitk::Geometry3D::Pointer geometry = image->GetGeometry();
  mitk::Point3D centre = geometry->GetCenter();

  mitk::Vector3D extentsInWorldCoordinateOrder;
  mitk::GetExtentsInVxInWorldCoordinateOrder(image, extentsInWorldCoordinateOrder);

  mitk::Vector3D spacingInWorldCoordinateOrder;
  mitk::GetSpacingInWorldCoordinateOrder(image, spacingInWorldCoordinateOrder);

  for (int i = 0; i < 3; ++i)
  {
    double distanceFromCentre = std::abs(selectedPosition[i] - centre[i]);
    if (static_cast<int>(extentsInWorldCoordinateOrder[i]) % 2 == 0)
    {
      /// If the number of slices is an even number then the selected position
      /// must be a half voxel far from the centre, either way.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(std::abs(distanceFromCentre - spacingInWorldCoordinateOrder[i] / 2.0) < 0.001);
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

  mitk::Image* image = dynamic_cast<mitk::Image*>(d->ImageNode->GetData());

  mitk::Vector3D spacingInWorldCoordinateOrder;
  mitk::GetSpacingInWorldCoordinateOrder(image, spacingInWorldCoordinateOrder);

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  /// Register to listen to SliceNavigators, slice changed events.
  mitk::SignalCollector::Pointer axialSncSignalCollector = mitk::SignalCollector::New();
  mitk::SignalCollector::Pointer sagittalSncSignalCollector = mitk::SignalCollector::New();
  mitk::SignalCollector::Pointer coronalSncSignalCollector = mitk::SignalCollector::New();

  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  unsigned long axialSncObserverTag = axialSnc->AddObserver(geometrySliceEvent, axialSncSignalCollector);
  unsigned long sagittalSncObserverTag = sagittalSnc->AddObserver(geometrySliceEvent, sagittalSncSignalCollector);
  unsigned long coronalSncObserverTag = coronalSnc->AddObserver(geometrySliceEvent, coronalSncSignalCollector);

  /// Note that we store a reference to these objects so that we do not need to get them
  /// repeatedly after setting the selected position.
  const mitk::SignalCollector::Signals& axialSncSignals = axialSncSignalCollector->GetSignals();
  const mitk::SignalCollector::Signals& sagittalSncSignals = sagittalSncSignalCollector->GetSignals();
  const mitk::SignalCollector::Signals& coronalSncSignals = coronalSncSignalCollector->GetSignals();

  mitk::Point3D initialPosition = d->Viewer->GetSelectedPosition();
  mitk::Point3D newPosition = initialPosition;
  newPosition[0] += 2 * spacingInWorldCoordinateOrder[0];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 0ul);
  QCOMPARE(sagittalSncSignals.size(), 1ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  sagittalSncSignalCollector->Clear();

  newPosition[1] += 2 * spacingInWorldCoordinateOrder[1];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 0ul);
  QCOMPARE(sagittalSncSignals.size(), 0ul);
  QCOMPARE(coronalSncSignals.size(), 1ul);

  coronalSncSignalCollector->Clear();

  newPosition[2] += 2 * spacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 1ul);
  QCOMPARE(sagittalSncSignals.size(), 0ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  axialSncSignalCollector->Clear();

  newPosition[0] -= 3 * spacingInWorldCoordinateOrder[0];
  newPosition[1] -= 3 * spacingInWorldCoordinateOrder[1];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 0ul);
  QCOMPARE(sagittalSncSignals.size(), 1ul);
  QCOMPARE(coronalSncSignals.size(), 1ul);

  sagittalSncSignalCollector->Clear();
  coronalSncSignalCollector->Clear();

  newPosition[0] -= 4 * spacingInWorldCoordinateOrder[0];
  newPosition[2] -= 4 * spacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 1ul);
  QCOMPARE(sagittalSncSignals.size(), 1ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  axialSncSignalCollector->Clear();
  sagittalSncSignalCollector->Clear();

  newPosition[1] += 5 * spacingInWorldCoordinateOrder[1];
  newPosition[2] += 5 * spacingInWorldCoordinateOrder[2];
  d->Viewer->SetSelectedPosition(newPosition);

  QCOMPARE(d->Viewer->GetSelectedPosition(), newPosition);
  QCOMPARE(axialSncSignals.size(), 1ul);
  QCOMPARE(sagittalSncSignals.size(), 0ul);
  QCOMPARE(coronalSncSignals.size(), 1ul);

  axialSncSignalCollector->Clear();
  coronalSncSignalCollector->Clear();

  axialSnc->RemoveObserver(axialSncObserverTag);
  sagittalSnc->RemoveObserver(sagittalSncObserverTag);
  coronalSnc->RemoveObserver(coronalSncObserverTag);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectPositionByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::Image* image = dynamic_cast<mitk::Image*>(d->ImageNode->GetData());

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  /// Register to listen to SliceNavigators, slice changed events.
  mitk::SignalCollector::Pointer axialSncSignalCollector = mitk::SignalCollector::New();
  mitk::SignalCollector::Pointer sagittalSncSignalCollector = mitk::SignalCollector::New();
  mitk::SignalCollector::Pointer coronalSncSignalCollector = mitk::SignalCollector::New();

  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  unsigned long axialSncObserverTag = axialSnc->AddObserver(geometrySliceEvent, axialSncSignalCollector);
  unsigned long sagittalSncObserverTag = sagittalSnc->AddObserver(geometrySliceEvent, sagittalSncSignalCollector);
  unsigned long coronalSncObserverTag = coronalSnc->AddObserver(geometrySliceEvent, coronalSncSignalCollector);

  /// Note that we store a reference to these objects so that we do not need to get them
  /// repeatedly after setting the selected position.
  const mitk::SignalCollector::Signals& axialSncSignals = axialSncSignalCollector->GetSignals();
  const mitk::SignalCollector::Signals& sagittalSncSignals = sagittalSncSignalCollector->GetSignals();
  const mitk::SignalCollector::Signals& coronalSncSignals = coronalSncSignalCollector->GetSignals();

  QPoint centre = coronalWindow->rect().center();
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, centre);

  axialSncSignalCollector->Clear();
  sagittalSncSignalCollector->Clear();
  coronalSncSignalCollector->Clear();

  mitk::Point3D lastPosition = d->Viewer->GetSelectedPosition();

  QPoint newPoint = centre;
  newPoint.rx() += 30;
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  mitk::Point3D newPosition = d->Viewer->GetSelectedPosition();
  QVERIFY(newPosition[0] != lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QCOMPARE(newPosition[2], lastPosition[2]);
  QCOMPARE(axialSncSignals.size(), 0ul);
  QCOMPARE(sagittalSncSignals.size(), 1ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  sagittalSncSignalCollector->Clear();

  lastPosition = newPosition;

  newPoint.ry() += 20;
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  newPosition = d->Viewer->GetSelectedPosition();
  QCOMPARE(newPosition[0], lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QVERIFY(newPosition[2] != lastPosition[1]);
  QCOMPARE(axialSncSignals.size(), 1ul);
  QCOMPARE(sagittalSncSignals.size(), 0ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  axialSncSignalCollector->Clear();

  lastPosition = d->Viewer->GetSelectedPosition();

  newPoint.rx() -= 40;
  newPoint.ry() += 50;
  QTest::mouseClick(coronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  newPosition = d->Viewer->GetSelectedPosition();
  QVERIFY(newPosition[0] != lastPosition[0]);
  QCOMPARE(newPosition[1], lastPosition[1]);
  QVERIFY(newPosition[2] != lastPosition[2]);
  QCOMPARE(axialSncSignals.size(), 1ul);
  QCOMPARE(sagittalSncSignals.size(), 1ul);
  QCOMPARE(coronalSncSignals.size(), 0ul);

  axialSncSignalCollector->Clear();
  sagittalSncSignalCollector->Clear();

  axialSnc->RemoveObserver(axialSncObserverTag);
  sagittalSnc->RemoveObserver(sagittalSncObserverTag);
  coronalSnc->RemoveObserver(coronalSncObserverTag);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QmitkRenderWindow* axialWindow = d->Viewer->GetAxialWindow();
  mitk::SliceNavigationController* axialSnc = axialWindow->GetSliceNavigationController();
  QmitkRenderWindow* sagittalWindow = d->Viewer->GetSagittalWindow();
  mitk::SliceNavigationController* sagittalSnc = sagittalWindow->GetSliceNavigationController();
  QmitkRenderWindow* coronalWindow = d->Viewer->GetCoronalWindow();
  mitk::SliceNavigationController* coronalSnc = coronalWindow->GetSliceNavigationController();

  /// Register to listen to SliceNavigators, slice changed events.
  mitk::SignalCollector::Pointer signalCollector = mitk::SignalCollector::New();

  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  unsigned long axialSncObserverTag = axialSnc->AddObserver(geometrySliceEvent, signalCollector);
  unsigned long sagittalSncObserverTag = sagittalSnc->AddObserver(geometrySliceEvent, signalCollector);
  unsigned long coronalSncObserverTag = coronalSnc->AddObserver(geometrySliceEvent, signalCollector);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

  /// TODO The focus should be on the coronal window already.
  focusManager->SetFocused(coronalWindow->GetRenderer());

  unsigned long focusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), signalCollector);

  /// The default layout was set to coronal in the init() function.
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  const mitk::SignalCollector::Signals& sncAndFocusSignals = signalCollector->GetSignals();
  QVERIFY(sncAndFocusSignals.size() < 4);

  mitk::SignalCollector::Signals::const_iterator it = sncAndFocusSignals.begin();
  mitk::SignalCollector::Signals::const_iterator signalsEnd = sncAndFocusSignals.end();
  for ( ; it != signalsEnd; ++it)
  {
    const itk::EventObject* event = it->second;
    const mitk::SliceNavigationController::GeometrySliceEvent* geometrySliceEvent =
        dynamic_cast<const mitk::SliceNavigationController::GeometrySliceEvent*>(event);
    if (geometrySliceEvent)
    {
      MITK_INFO << "geometry slice event";
      continue;
    }

    const mitk::FocusEvent* focusEvent =
        dynamic_cast<const mitk::FocusEvent*>(event);
    if (focusEvent)
    {
      MITK_INFO << "focus event";
      continue;
    }
  }

  signalCollector->Clear();

  QRect rect = sagittalWindow->rect();
  QPoint centre = rect.center();
  QPoint bottomLeftCorner = rect.bottomLeft();
  QPoint aPosition((bottomLeftCorner.x() + centre.x()) / 2, (bottomLeftCorner.y() + centre.y()) / 2);
  QTest::mouseClick(sagittalWindow, Qt::LeftButton, Qt::NoModifier, aPosition);

  MITK_INFO << signalCollector;

  it = sncAndFocusSignals.begin();
  signalsEnd = sncAndFocusSignals.end();
  for ( ; it != signalsEnd; ++it)
  {
    const itk::EventObject* event = it->second;
    const mitk::SliceNavigationController::GeometrySliceEvent* geometrySliceEvent =
        dynamic_cast<const mitk::SliceNavigationController::GeometrySliceEvent*>(event);
    if (geometrySliceEvent)
    {
      MITK_INFO << "geometry slice event";
      continue;
    }

    const mitk::FocusEvent* focusEvent =
        dynamic_cast<const mitk::FocusEvent*>(event);
    if (focusEvent)
    {
      MITK_INFO << "focus event";
      continue;
    }
  }

  MITK_INFO << signalCollector;
  QVERIFY(sncAndFocusSignals.size() <= 3);

  axialSnc->RemoveObserver(axialSncObserverTag);
  sagittalSnc->RemoveObserver(sagittalSncObserverTag);
  coronalSnc->RemoveObserver(coronalSncObserverTag);

  focusManager->RemoveObserver(focusManagerObserverTag);
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

  /// We used the arguments to initialise the test. No arguments is passed
  /// to the Qt test, so that all the test functions are executed.
  argc = 1;
  argv[1] = NULL;
  return QTest::qExec(&test, argc, argv);
}
