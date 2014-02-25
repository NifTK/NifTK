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

#include <QmitkRegisterClasses.h>

#include <mitkMIDASOrientationUtils.h>
#include <mitkNifTKCoreObjectFactory.h>
#include <niftkSingleViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>

#include <mitkItkSignalCollector.cxx>
#include <mitkQtSignalCollector.cxx>
#include <niftkSingleViewerWidgetState.h>


class niftkSingleViewerWidgetTestClassPrivate
{
public:

  niftkSingleViewerWidgetTestClassPrivate()
  : GeometrySendEvent(NULL, 0)
  , GeometryUpdateEvent(NULL, 0)
  , GeometryTimeEvent(NULL, 0)
  , GeometrySliceEvent(NULL, 0)
  {
  }

  std::string FileName;
  mitk::DataStorage::Pointer DataStorage;
  mitk::RenderingManager::Pointer RenderingManager;

  mitk::DataNode::Pointer ImageNode;
  mitk::Image* Image;
  mitk::Vector3D WorldExtents;
  mitk::Vector3D WorldSpacings;
  int WorldAxisFlipped[3];

  niftkSingleViewerWidget* Viewer;
  niftkMultiViewerVisibilityManager* VisibilityManager;

  QmitkRenderWindow* AxialWindow;
  QmitkRenderWindow* SagittalWindow;
  QmitkRenderWindow* CoronalWindow;
  QmitkRenderWindow* _3DWindow;

  mitk::SliceNavigationController* AxialSnc;
  mitk::SliceNavigationController* SagittalSnc;
  mitk::SliceNavigationController* CoronalSnc;

  niftkSingleViewerWidgetTestClass::ViewerStateTester::Pointer StateTester;

  bool InteractiveMode;

  mitk::FocusEvent FocusEvent;
  mitk::SliceNavigationController::GeometrySliceEvent GeometrySendEvent;
  mitk::SliceNavigationController::GeometrySliceEvent GeometryUpdateEvent;
  mitk::SliceNavigationController::GeometrySliceEvent GeometryTimeEvent;
  mitk::SliceNavigationController::GeometrySliceEvent GeometrySliceEvent;

  const char* GeometryChanged;
  const char* WindowLayoutChanged;
  const char* SelectedRenderWindowChanged;
  const char* SelectedTimeStepChanged;
  const char* SelectedPositionChanged;
  const char* CursorPositionChanged;
  const char* ScaleFactorChanged;
  const char* CursorPositionBindingChanged;
  const char* ScaleFactorBindingChanged;
  const char* CursorVisibilityChanged;
};


// --------------------------------------------------------------------------
niftkSingleViewerWidgetTestClass::niftkSingleViewerWidgetTestClass()
: QObject()
, d_ptr(new niftkSingleViewerWidgetTestClassPrivate())
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->ImageNode = 0;
  d->Image = 0;
  d->WorldExtents.Fill(0.0);
  d->WorldSpacings.Fill(1.0);
  d->WorldAxisFlipped[0] = 0;
  d->WorldAxisFlipped[1] = 0;
  d->WorldAxisFlipped[2] = 0;
  d->Viewer = 0;
  d->VisibilityManager = 0;
  d->InteractiveMode = false;

  d->SelectedRenderWindowChanged = SIGNAL(SelectedRenderWindowChanged(MIDASOrientation));
  d->SelectedPositionChanged = SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&));
  d->SelectedTimeStepChanged = SIGNAL(SelectedTimeStepChanged(niftkSingleViewerWidget*, int));
  d->CursorPositionChanged = SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, MIDASOrientation, const mitk::Vector2D&));
  d->ScaleFactorChanged = SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double));
  d->CursorPositionBindingChanged = SIGNAL(CursorPositionBindingChanged(niftkSingleViewerWidget*, bool));
  d->ScaleFactorBindingChanged = SIGNAL(ScaleFactorBindingChanged(niftkSingleViewerWidget*, bool));
  d->WindowLayoutChanged = SIGNAL(WindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout));
  d->GeometryChanged = SIGNAL(GeometryChanged(niftkSingleViewerWidget*, mitk::TimeGeometry*));
  d->CursorVisibilityChanged = SIGNAL(CursorVisibilityChanged(niftkSingleViewerWidget*, bool));
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
mitk::Vector2D niftkSingleViewerWidgetTestClass::GetDisplayPositionAtPoint(QmitkRenderWindow *renderWindow, const QPoint& point)
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
    double tolerance = d->WorldSpacings[i] / 2.0;
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
  return cursorPositions1.size() == std::size_t(3)
      && cursorPositions2.size() == std::size_t(3)
      && Self::Equals(cursorPositions1[0], cursorPositions2[0], tolerance)
      && Self::Equals(cursorPositions1[1], cursorPositions2[1], tolerance)
      && Self::Equals(cursorPositions1[2], cursorPositions2[2], tolerance);
}


// --------------------------------------------------------------------------
mitk::Point3D niftkSingleViewerWidgetTestClass::GetRandomWorldPosition() const
{
  Q_D(const niftkSingleViewerWidgetTestClass);

  mitk::Geometry3D* geometry = d->Image->GetGeometry();

  return geometry->GetOrigin()
      + geometry->GetAxisVector(0) * ((double) std::rand() / RAND_MAX)
      + geometry->GetAxisVector(1) * ((double) std::rand() / RAND_MAX)
      + geometry->GetAxisVector(2) * ((double) std::rand() / RAND_MAX);
}


// --------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidgetTestClass::GetRandomDisplayPosition()
{
  mitk::Vector2D randomDisplayPosition;
  randomDisplayPosition[0] = (double) std::rand() / RAND_MAX;
  randomDisplayPosition[1] = (double) std::rand() / RAND_MAX;
  return randomDisplayPosition;
}


// --------------------------------------------------------------------------
std::vector<mitk::Vector2D> niftkSingleViewerWidgetTestClass::GetRandomDisplayPositions(std::size_t size)
{
  std::vector<mitk::Vector2D> randomDisplayPositions(size);
  for (std::size_t i = 0; i < size; ++i)
  {
    randomDisplayPositions[i] = Self::GetRandomDisplayPosition();
  }
  return randomDisplayPositions;
}


// --------------------------------------------------------------------------
double niftkSingleViewerWidgetTestClass::GetRandomScaleFactor()
{
  return 2.0 * std::rand() / RAND_MAX;
}


// --------------------------------------------------------------------------
std::vector<double> niftkSingleViewerWidgetTestClass::GetRandomScaleFactors(std::size_t size)
{
  std::vector<double> randomScaleFactors(size);
  for (std::size_t i = 0; i < size; ++i)
  {
    randomScaleFactors[i] = Self::GetRandomScaleFactor();
  }
  return randomScaleFactors;
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::SetRandomPositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->StateTester->Clear();
  d->Viewer->SetSelectedPosition(this->GetRandomWorldPosition());
  d->StateTester->Clear();
  d->Viewer->SetCursorPositions(Self::GetRandomDisplayPositions());
  d->StateTester->Clear();
  d->Viewer->SetScaleFactors(Self::GetRandomScaleFactors());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::initTestCase()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  // Need to load images, specifically using MIDAS/DRC object factory.
  ::RegisterNifTKCoreObjectFactory();

  QmitkRegisterClasses();

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

  mitk::GetExtentsInVxInWorldCoordinateOrder(d->Image, d->WorldExtents);
  mitk::GetSpacingInWorldCoordinateOrder(d->Image, d->WorldSpacings);

  /// This is fixed and does not depend on the image geometry.
  d->WorldAxisFlipped[SagittalAxis] = +1;
  d->WorldAxisFlipped[CoronalAxis] = -1;
  d->WorldAxisFlipped[AxialAxis] = -1;
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
  d->Viewer->SetRememberSettingsPerWindowLayout(false);
  d->Viewer->SetDisplayInteractionsEnabled(true);
  d->Viewer->SetCursorPositionBinding(false);
  d->Viewer->SetScaleFactorBinding(false);
  d->Viewer->SetDefaultSingleWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->Viewer->SetDefaultMultiWindowLayout(WINDOW_LAYOUT_ORTHO);

//  d->VisibilityManager->connect(d->Viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, QmitkRenderWindow*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

  d->VisibilityManager->RegisterViewer(d->Viewer);
  d->VisibilityManager->SetAllNodeVisibilityForViewer(0, false);

  d->Viewer->resize(1024, 1024);
  d->Viewer->show();

  QTest::qWaitForWindowShown(d->Viewer);

  std::vector<mitk::DataNode*> nodes(1);
  nodes[0] = d->ImageNode;

  d->AxialWindow = d->Viewer->GetAxialWindow();
  d->SagittalWindow = d->Viewer->GetSagittalWindow();
  d->CoronalWindow = d->Viewer->GetCoronalWindow();
  d->_3DWindow = d->Viewer->Get3DWindow();

  this->DropNodes(d->CoronalWindow, nodes);

  d->Viewer->SetCursorVisible(true);

  /// Create a state tester that works for all of the test functions.

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  d->AxialSnc = d->AxialWindow->GetSliceNavigationController();
  d->SagittalSnc = d->SagittalWindow->GetSliceNavigationController();
  d->CoronalSnc = d->CoronalWindow->GetSliceNavigationController();

  d->StateTester = ViewerStateTester::New(d->Viewer);

  d->StateTester->Connect(focusManager, mitk::FocusEvent());
  mitk::SliceNavigationController::GeometrySliceEvent geometrySliceEvent(NULL, 0);
  d->StateTester->Connect(d->AxialSnc, geometrySliceEvent);
  d->StateTester->Connect(d->SagittalSnc, geometrySliceEvent);
  d->StateTester->Connect(d->CoronalSnc, geometrySliceEvent);
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
void niftkSingleViewerWidgetTestClass::DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes)
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
//  QStringList types;
//  types << "application/x-mitk-datanodes";
  QDragEnterEvent dragEnterEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  QDropEvent dropEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  dropEvent.acceptProposedAction();
  if (!qApp->notify(renderWindow, &dragEnterEvent))
  {
    QTest::qWarn("Drag enter event not accepted by receiving widget.");
  }
  if (!qApp->notify(renderWindow, &dropEvent))
  {
    QTest::qWarn("Drop event not accepted by receiving widget.");
  }

  d->VisibilityManager->OnNodesDropped(d->Viewer, renderWindow, nodes);
//  d->Viewer->OnNodesDropped(renderWindow, nodes);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::MouseWheel(QWidget* widget,
                            Qt::MouseButtons buttons,
                            Qt::KeyboardModifiers modifiers,
                            QPoint position, int delta,
                            Qt::Orientation orientation)
{
  QTEST_ASSERT(modifiers == 0 || modifiers & Qt::KeyboardModifierMask);

  modifiers &= static_cast<unsigned int>(Qt::KeyboardModifierMask);
  QWheelEvent wheelEvent(position, widget->mapToGlobal(position), delta, buttons, modifiers, orientation);

  QSpontaneKeyEvent::setSpontaneous(&wheelEvent); // hmmmm
  if (!qApp->notify(widget, &wheelEvent))
  {
    QTest::qWarn("Wheel event not accepted by receiving widget.");
  }
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
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetSelectedPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Make sure that the state does not change and no signal is sent out.
  ViewerState::Pointer state = ViewerState::New(d->Viewer);
  d->StateTester->SetExpectedState(state);

  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());

  mitk::Point3D centre = d->Image->GetGeometry()->GetCenter();

  for (int i = 0; i < 3; ++i)
  {
    double distanceFromCentre = std::abs(selectedPosition[i] - centre[i]);
    if (static_cast<int>(d->WorldExtents[i]) % 2 == 0)
    {
      /// If the number of slices is an even number then the selected position
      /// must be a half voxel far from the centre, either way.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(std::abs(distanceFromCentre - d->WorldSpacings[i] / 2.0) < 0.001);
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

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);
  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  selectedPosition[SagittalAxis] += 20 * d->WorldSpacings[SagittalAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedSagittalSlice += d->WorldAxisFlipped[SagittalAxis] * 20;
  std::vector<mitk::Vector2D> cursorPositions = expectedState->GetCursorPositions();
  std::vector<double> scaleFactors = expectedState->GetScaleFactors();
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 20 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  selectedPosition[CoronalAxis] += 20 * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * 20;
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  selectedPosition[AxialAxis] += 20 * d->WorldSpacings[AxialAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedAxialSlice += d->WorldAxisFlipped[AxialAxis] * 20;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 20 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  selectedPosition[SagittalAxis] -= 30 * d->WorldSpacings[SagittalAxis];
  selectedPosition[CoronalAxis] -= 30 * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedSagittalSlice -= d->WorldAxisFlipped[SagittalAxis] * 30;
  expectedCoronalSlice -= d->WorldAxisFlipped[CoronalAxis] * 30;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 30 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  selectedPosition[SagittalAxis] -= 40 * d->WorldSpacings[SagittalAxis];
  selectedPosition[AxialAxis] -= 40 * d->WorldSpacings[AxialAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedSagittalSlice -= d->WorldAxisFlipped[SagittalAxis] * 40;
  expectedAxialSlice -= d->WorldAxisFlipped[AxialAxis] * 40;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 40 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 40 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  selectedPosition = d->Viewer->GetSelectedPosition();
  selectedPosition[CoronalAxis] += 50 * d->WorldSpacings[CoronalAxis];
  selectedPosition[AxialAxis] += 50 * d->WorldSpacings[AxialAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * 50;
  expectedAxialSlice += d->WorldAxisFlipped[AxialAxis] * 50;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 50 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  /// TODO:
  /// This test fails and is temporarily disabled.
//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->Viewer->GetSelectedPosition(), selectedPosition);
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testGetSelectedSlice()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  selectedPosition[CoronalAxis] += 20 * d->WorldSpacings[CoronalAxis];
  unsigned expectedCoronalSliceInSnc = expectedCoronalSlice + 20;
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * 20;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int coronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
  unsigned coronalSliceInSnc = d->CoronalSnc->GetSlice()->GetPos();
  QCOMPARE(coronalSlice, expectedCoronalSlice);
  QCOMPARE(coronalSliceInSnc, expectedCoronalSliceInSnc);

  selectedPosition[AxialAxis] += 30 * d->WorldSpacings[AxialAxis];
  unsigned expectedAxialSliceInSnc = expectedAxialSlice + 30;
  expectedAxialSlice += d->WorldAxisFlipped[AxialAxis] * 30;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int axialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  unsigned axialSliceInSnc = d->AxialSnc->GetSlice()->GetPos();
  QCOMPARE(axialSlice, expectedAxialSlice);
  QCOMPARE(axialSliceInSnc, expectedAxialSliceInSnc);

  selectedPosition[SagittalAxis] += 40 * d->WorldSpacings[SagittalAxis];
  unsigned expectedSagittalSliceInSnc = expectedSagittalSlice + 40;
  expectedSagittalSlice += d->WorldAxisFlipped[SagittalAxis] * 40;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int sagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  unsigned sagittalSliceInSnc = d->SagittalSnc->GetSlice()->GetPos();
  QCOMPARE(sagittalSlice, expectedSagittalSlice);
  QCOMPARE(sagittalSliceInSnc, expectedSagittalSliceInSnc);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedSlice()
{
  Q_D(niftkSingleViewerWidgetTestClass);

//  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

//  int axialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  int sagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int coronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
  unsigned expectedCoronalSncPos = d->CoronalSnc->GetSlice()->GetSteps() - 1 - coronalSlice;

  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();

  int delta;

  delta = +20;
  coronalSlice += delta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * delta * d->WorldSpacings[CoronalAxis];
  expectedCoronalSncPos += d->WorldAxisFlipped[CoronalAxis] * delta;

  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalSlice);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), coronalSlice);
  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSncPos);

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

  mitk::Vector2D cursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QVERIFY(::EqualsWithTolerance(cursorPosition, centrePosition));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(ViewerState::New(d->Viewer));

  cursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);

  QVERIFY(::EqualsWithTolerance(cursorPosition, centrePosition));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(ViewerState::New(d->Viewer));

  cursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);

  QVERIFY(::EqualsWithTolerance(cursorPosition, centrePosition));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetCursorPosition()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  std::vector<mitk::Vector2D> cursorPositions = d->Viewer->GetCursorPositions();
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.4;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.6;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPosition(MIDAS_ORIENTATION_CORONAL, cursorPositions[MIDAS_ORIENTATION_CORONAL]);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals(d->CursorPositionChanged).size() == 1);
  QVERIFY(d->StateTester->GetQtSignals().size() == 1);

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.45;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.65;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPosition(MIDAS_ORIENTATION_AXIAL, cursorPositions[MIDAS_ORIENTATION_AXIAL]);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), cursorPositions[MIDAS_ORIENTATION_AXIAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.35;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.65;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPosition(MIDAS_ORIENTATION_SAGITTAL, cursorPositions[MIDAS_ORIENTATION_SAGITTAL]);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), cursorPositions[MIDAS_ORIENTATION_SAGITTAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QVERIFY(d->StateTester->GetQtSignals().empty());
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

  QVERIFY(ViewerState::New(d->Viewer)->EqualsWithTolerance(d->Viewer->GetCursorPositions(), centrePositions));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetCursorPositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  std::vector<mitk::Vector2D> cursorPositions = d->Viewer->GetCursorPositions();
  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.41;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.61;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), cursorPositions[MIDAS_ORIENTATION_AXIAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.52;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.72;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), cursorPositions[MIDAS_ORIENTATION_SAGITTAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.33;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.23;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.44;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.74;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.64;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.84;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QVERIFY(d->StateTester->GetQtSignals().empty());

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.25;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.35;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.75;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.95;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.16;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.56;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.46;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.86;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.27;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.37;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.47;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.57;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.67;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.77;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  /// Note that the CursorPositionChanged and ScaleFactorChanged signals are emitted only for the visible windows.
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));
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

  /// The default window layout was set to coronal in the init() function.
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedRenderWindow()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

  /// Note:
  /// If the display is redirected (like during the overnight builds) then the application
  /// will not have key focus. Therefore, here we check if the focus is on the right window
  /// if and only if the application has key focus at all.
  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QCOMPARE(d->Viewer->IsSelected(), true);

  d->StateTester->Clear();

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);
  expectedState->SetSelectedRenderWindow(d->AxialWindow);
  expectedState->SetOrientation(MIDAS_ORIENTATION_AXIAL);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedRenderWindow(d->AxialWindow);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QCOMPARE(d->Viewer->IsSelected(), true);

  QCOMPARE(d->StateTester->GetItkSignals(d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

  mitk::Point3D centreWorldPosition = d->Viewer->GetGeometry()->GetCenterInWorld();

  mitk::Vector2D centreDisplayPosition;
  centreDisplayPosition.Fill(0.5);

  std::size_t scaleFactorChanges;
  std::size_t cursorPositionChanges;

  /// Note:
  /// If the display is redirected (like during the overnight builds) then the application
  /// will not have key focus. Therefore, here we check if the focus is on the right window
  /// if and only if the application has key focus at all.
  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), centreWorldPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));

  ViewerState::Pointer stateCoronal = ViewerState::New(d->Viewer);

  /// We cannot test the full state because the cursor position and scale factor
  /// of the sagittal and the axial windows have not been initialised yet.
  /// They will be initialised when we first switch to those windows.

  /// The default layout was set to coronal in the init() function.
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(!d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), centreWorldPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->ScaleFactorChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  ViewerState::Pointer stateSagittal = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(!d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), centreWorldPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->ScaleFactorChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  ViewerState::Pointer stateAxial = ViewerState::New(d->Viewer);

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  expectedState->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  expectedState->SetOrientation(MIDAS_ORIENTATION_CORONAL);
  expectedState->SetSelectedRenderWindow(d->CoronalWindow);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->ScaleFactorChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();
  expectedState->SetWindowLayout(WINDOW_LAYOUT_3D);
  expectedState->SetOrientation(MIDAS_ORIENTATION_UNKNOWN);
  expectedState->SetSelectedRenderWindow(d->_3DWindow);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);

  QVERIFY(!qApp->focusWidget() || d->_3DWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->_3DWindow);
  QCOMPARE(focusManager->GetFocused(), d->_3DWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(!d->CoronalWindow->isVisible());
  QVERIFY(d->_3DWindow->isVisible());
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  ViewerState::Pointer state3D = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges == std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges == std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer state3H = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer state3V = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateCorAxH = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateCorAxV = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateCorSagH = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateCorSagV = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(!d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateSagAxH = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(!d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateSagAxV = ViewerState::New(d->Viewer);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(focusManager->GetFocused(), d->SagittalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(0));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 0 + cursorPositionChanges + scaleFactorChanges);

  ViewerState::Pointer stateOrtho = ViewerState::New(d->Viewer);

  ///
  /// Check if we get the same state when returning to a previously selected orientation.
  ///

  this->SetRandomPositions();

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateAxial);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateAxial);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagittal);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagittal);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCoronal);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCoronal);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3D);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3D);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3H);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3H);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3V);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(state3V);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorAxH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorAxH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorAxV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorAxV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorSagH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorSagH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorSagV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateCorSagV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagAxH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagAxH);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagAxV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateSagAxV);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateOrtho);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  this->SetRandomPositions();
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(stateOrtho);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testRememberPositionsPerWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer->SetRememberSettingsPerWindowLayout(true);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  this->SetRandomPositions();
  ViewerState::Pointer axialState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  this->SetRandomPositions();
  ViewerState::Pointer sagittalState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  this->SetRandomPositions();
  ViewerState::Pointer coronalState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);
  this->SetRandomPositions();
  ViewerState::Pointer _3DState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);
  this->SetRandomPositions();
  ViewerState::Pointer _3HState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);
  this->SetRandomPositions();
  ViewerState::Pointer _3VState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);
  this->SetRandomPositions();
  ViewerState::Pointer corAxHState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);
  this->SetRandomPositions();
  ViewerState::Pointer corAxVState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);
  this->SetRandomPositions();
  ViewerState::Pointer corSagHState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);
  this->SetRandomPositions();
  ViewerState::Pointer corSagVState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);
  this->SetRandomPositions();
  ViewerState::Pointer sagAxHState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);
  this->SetRandomPositions();
  ViewerState::Pointer sagAxVState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  this->SetRandomPositions();
  ViewerState::Pointer orthoState = ViewerState::New(d->Viewer);

  ///
  /// Switch back to each layout and check if the states are correctly restored.
  ///

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(axialState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(sagittalState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(coronalState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(_3DState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(_3HState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(_3VState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(corAxHState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(corAxVState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(corSagHState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(corSagVState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(sagAxHState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(sagAxVState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);

  d->StateTester->Clear();
  d->StateTester->SetExpectedState(orthoState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  //// -------------------------------------------------------------------------

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  mitk::Vector2D cursorPosition;
  cursorPosition[0] = 0.4;
  cursorPosition[1] = 0.6;
  QPoint pointCursorPosition = this->GetPointAtCursorPosition(d->CoronalWindow, cursorPosition);

  d->StateTester->Clear();
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, pointCursorPosition);

  QVERIFY(this->Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), cursorPosition));

  coronalState = ViewerState::New(d->Viewer);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();
  d->StateTester->SetExpectedState(coronalState);
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectPositionByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QPoint newPoint;
  ViewerState::Pointer expectedState;
  mitk::Point3D expectedSelectedPosition;
  std::vector<mitk::Vector2D> expectedCursorPositions;

  double scaleFactor;

  /// ---------------------------------------------------------------------------
  /// Coronal window
  /// ---------------------------------------------------------------------------

  newPoint = d->CoronalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_CORONAL);

  /// Note that the origin of a QPoint is the upper left corner and the y coordinate
  /// is increasing downwards, in contrast with both the world coordinates and the
  /// display coordinates, where the origin is in the bottom left corner (and in the
  /// front for the world position) and the y coordinate is increasing upwards.

  newPoint.rx() += 120;
  expectedSelectedPosition[SagittalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 120.0 / d->CoronalWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 60.0 / d->CoronalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[AxialAxis] += 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 30.0 / d->CoronalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 40.0 / d->CoronalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// Axial window
  /// ---------------------------------------------------------------------------

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();

  newPoint = d->AxialWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_AXIAL);

  /// Note that the origin of a QPoint is the upper left corner and the y coordinate
  /// is increasing downwards, in contrast with both the world coordinates and the
  /// display coordinates, where the origin is in the bottom left corner (and in the
  /// front for the world position) and the y coordinate is increasing upwards.

  newPoint.rx() += 120;
  expectedSelectedPosition[SagittalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] += 120.0 / d->AxialWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[CoronalAxis] += 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] -= 60.0 / d->AxialWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[CoronalAxis] -= 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] -= 30.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] += 40.0 / d->AxialWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// Sagittal window
  /// ---------------------------------------------------------------------------

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  d->StateTester->Clear();

  newPoint = d->SagittalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_SAGITTAL);

  newPoint.rx() += 120;
  expectedSelectedPosition[CoronalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] += 120.0 / d->SagittalWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] -= 60.0 / d->SagittalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[CoronalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[AxialAxis] += 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] -= 30.0 / d->SagittalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] += 40.0 / d->SagittalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// 2x2 (orthogonal) window layout
  /// ---------------------------------------------------------------------------

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// 2x2 (orthogonal) window layout, coronal window
  /// ---------------------------------------------------------------------------

  newPoint = d->CoronalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_CORONAL);

  /// Note that the origin of a QPoint is the upper left corner and the y coordinate
  /// is increasing downwards, in contrast with both the world coordinates and the
  /// display coordinates, where the origin is in the bottom left corner (and in the
  /// front for the world position) and the y coordinate is increasing upwards.

  newPoint.rx() += 120;
  expectedSelectedPosition[SagittalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 120.0 / d->CoronalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] += 120.0 / d->AxialWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 60.0 / d->CoronalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] -= 60.0 / d->SagittalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[AxialAxis] += 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 30.0 / d->CoronalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 40.0 / d->CoronalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] -= 30.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] += 40.0 / d->SagittalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(4));

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// 2x2 (orthogonal) window layout, axial window
  /// ---------------------------------------------------------------------------

  newPoint = d->AxialWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  /// Note:
  /// If the display is redirected (like during the overnight builds) then the application
  /// will not have key focus. Therefore, here we check if the focus is on the right window
  /// if and only if the application has key focus at all.
  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(mitk::GlobalInteraction::GetInstance()->GetFocus(), d->AxialWindow->GetRenderer());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->AxialWindow);
  QCOMPARE(d->Viewer->GetOrientation(), MIDAS_ORIENTATION_AXIAL);
  QCOMPARE(d->StateTester->GetItkSignals(d->FocusEvent).size(), std::size_t(1));
  /// Disabled, see the TODO comment above. None of these events should be emitted.
//  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_AXIAL);

  /// Note that the origin of a QPoint is the upper left corner and the y coordinate
  /// is increasing downwards, in contrast with both the world coordinates and the
  /// display coordinates, where the origin is in the bottom left corner (and in the
  /// front for the world position) and the y coordinate is increasing upwards.

  newPoint.rx() += 120;
  expectedSelectedPosition[SagittalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] += 120.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 120.0 / d->AxialWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[CoronalAxis] += 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] -= 60.0 / d->AxialWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] += 60.0 / d->SagittalWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[CoronalAxis] -= 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] -= 30.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] += 40.0 / d->AxialWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 30.0 / d->CoronalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] -= 40.0 / d->SagittalWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(4));

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// 2x2 (orthogonal) window layout, sagittal window
  /// ---------------------------------------------------------------------------

  newPoint = d->SagittalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QVERIFY(!qApp->focusWidget() || d->SagittalWindow->hasFocus());
  QCOMPARE(mitk::GlobalInteraction::GetInstance()->GetFocus(), d->SagittalWindow->GetRenderer());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
  QCOMPARE(d->Viewer->GetOrientation(), MIDAS_ORIENTATION_SAGITTAL);
  QCOMPARE(d->StateTester->GetItkSignals(d->FocusEvent).size(), std::size_t(1));
  /// Disabled, see the TODO comment above. None of these events should be emitted.
//  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedRenderWindowChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_SAGITTAL);

  newPoint.rx() += 120;
  expectedSelectedPosition[CoronalAxis] += 120.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] += 120.0 / d->SagittalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] -= 120.0 / d->AxialWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= 60.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] -= 60.0 / d->SagittalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 60.0 / d->CoronalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO
  /// Two CursorPositionChanged signals should be emitted, but three is.
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[CoronalAxis] -= 30.0 * scaleFactor;
  expectedSelectedPosition[AxialAxis] += 40.0 * scaleFactor;
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] -= 30.0 / d->SagittalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] += 40.0 / d->SagittalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] += 30.0 / d->AxialWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 40.0 / d->CoronalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(4));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testChangeSliceByMouseInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QPoint centre;
  ViewerState::Pointer expectedState;
  mitk::Point3D expectedSelectedPosition;
  std::vector<mitk::Vector2D> expectedCursorPositions;
  int expectedAxialSlice;
  int expectedSagittalSlice;
  int expectedCoronalSlice;

  int delta;

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// Coronal window
  /// ---------------------------------------------------------------------------

  centre = d->CoronalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, centre);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();
  expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

  delta = +1;
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  /// TODO The selected position and the cursor position changes in an unexpected way.
//  d->StateTester->SetExpectedState(expectedState);

  Self::MouseWheel(d->CoronalWindow, Qt::NoButton, Qt::NoModifier, centre, delta);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSlice);
  QCOMPARE((int)d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  delta = -1;
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  /// TODO The selected position and the cursor position changes in an unexpected way.
//  d->StateTester->SetExpectedState(expectedState);

  Self::MouseWheel(d->CoronalWindow, Qt::NoButton, Qt::NoModifier, centre, delta);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSlice);
  QCOMPARE((int)d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testChangeSliceByKeyInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QPoint centre;
  ViewerState::Pointer expectedState;
  mitk::Point3D expectedSelectedPosition;
  std::vector<mitk::Vector2D> expectedCursorPositions;
  int expectedAxialSlice;
  int expectedSagittalSlice;
  int expectedCoronalSlice;

  int delta;

  d->StateTester->Clear();

  /// ---------------------------------------------------------------------------
  /// Coronal window
  /// ---------------------------------------------------------------------------

  centre = d->CoronalWindow->rect().center();

  /// TODO
  /// The position should already be in the centre, but there seem to be
  /// a half spacing difference from the expected position. After clicking
  /// into the middle of the render window, the selected position moves
  /// with about half a spacing.
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, centre);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();
  expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

  delta = +1;
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  /// TODO The selected position and the cursor position changes in an unexpected way.
//  d->StateTester->SetExpectedState(expectedState);

  QTest::keyClick(d->CoronalWindow, Qt::Key_A, Qt::NoModifier);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSlice);
  QCOMPARE((int)d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  delta = -1;
  expectedCoronalSlice += d->WorldAxisFlipped[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  /// TODO The selected position and the cursor position changes in an unexpected way.
//  d->StateTester->SetExpectedState(expectedState);

  QTest::keyClick(d->CoronalWindow, Qt::Key_Z, Qt::NoModifier);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSlice);
  QCOMPARE((int)d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSlice);
  /// TODO Inconsistent state. The viewer->GetSelectedSlice() and the SNC gives different slices.
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectSliceThroughSliceNavigationController()
{
  Q_D(niftkSingleViewerWidgetTestClass);

//  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();
  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

  unsigned axialSncPos = d->AxialSnc->GetSlice()->GetPos();
  unsigned sagittalSncPos = d->SagittalSnc->GetSlice()->GetPos();
  unsigned coronalSncPos = d->CoronalSnc->GetSlice()->GetPos();

  int axialSliceDelta;
  int sagittalSliceDelta;
  int coronalSliceDelta;

  axialSliceDelta = +2;
  axialSncPos -= axialSliceDelta;
  expectedAxialSlice += axialSliceDelta;
  expectedSelectedPosition[AxialAxis] += d->WorldAxisFlipped[AxialAxis] * axialSliceDelta * d->WorldSpacings[AxialAxis];

  d->AxialSnc->GetSlice()->SetPos(axialSncPos);

  // TODO There is 1 difference to the expected value.
//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

  d->StateTester->Clear();

  sagittalSliceDelta = -3;
  sagittalSncPos += sagittalSliceDelta;
  expectedSagittalSlice += sagittalSliceDelta;
  expectedSelectedPosition[SagittalAxis] += d->WorldAxisFlipped[SagittalAxis] * sagittalSliceDelta * d->WorldSpacings[SagittalAxis];

  d->SagittalSnc->GetSlice()->SetPos(sagittalSncPos);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

  d->StateTester->Clear();

  coronalSliceDelta = +5;
  coronalSncPos -= coronalSliceDelta;
  expectedCoronalSlice += coronalSliceDelta;
  expectedSelectedPosition[CoronalAxis] += d->WorldAxisFlipped[CoronalAxis] * coronalSliceDelta * d->WorldSpacings[CoronalAxis];

  d->CoronalSnc->GetSlice()->SetPos(coronalSncPos);

  // TODO There is 1 difference to the expected value.
//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectPositionThroughSliceNavigationController()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::TimeGeometry* timeGeometry = d->Viewer->GetGeometry();
  mitk::Geometry3D* worldGeometry = timeGeometry->GetGeometryForTimeStep(0);

  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();
  mitk::Point3D randomWorldPosition = this->GetRandomWorldPosition();

  expectedSelectedPosition[AxialAxis] = randomWorldPosition[AxialAxis];

  mitk::Index3D expectedSelectedIndex;
  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

  int expectedAxialSlice = expectedSelectedIndex[1];
  int expectedSagittalSlice = expectedSelectedIndex[0];
  int expectedCoronalSlice = expectedSelectedIndex[2];

  unsigned expectedAxialSncPos = d->AxialSnc->GetSlice()->GetSteps() - 1 - expectedAxialSlice;
  unsigned expectedSagittalSncPos = expectedSagittalSlice;
  unsigned expectedCoronalSncPos = d->CoronalSnc->GetSlice()->GetSteps() - 1 - expectedCoronalSlice;

  d->AxialSnc->SelectSliceByPoint(expectedSelectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSncPos);

  d->StateTester->Clear();

  expectedSelectedPosition[SagittalAxis] = randomWorldPosition[SagittalAxis];

  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

  expectedSagittalSlice = expectedSelectedIndex[0];
  expectedSagittalSncPos = expectedSagittalSlice;

  d->SagittalSnc->SelectSliceByPoint(expectedSelectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSncPos);

  d->StateTester->Clear();

  expectedSelectedPosition[CoronalAxis] = randomWorldPosition[CoronalAxis];

  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

  expectedCoronalSlice = expectedSelectedIndex[2];
  expectedCoronalSncPos = d->CoronalSnc->GetSlice()->GetSteps() - 1 - expectedCoronalSlice;

  d->CoronalSnc->SelectSliceByPoint(expectedSelectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSncPos);

  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectRenderWindowByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

//  QCOMPARE(d->Viewer->IsSelected(), true);
//  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);

//  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

//  QCOMPARE(d->Viewer->IsSelected(), true);
//  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);

//  d->StateTester->Clear();

//  QPoint centre = d->SagittalWindow->rect().center();
//  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, centre);

//  QCOMPARE(d->Viewer->IsSelected(), true);
//  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
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

  std::srand((unsigned) std::time(0));

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
