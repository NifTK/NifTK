/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingleViewerWidgetTest.h"

#include <climits>

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

  /// The world origin, that is the centre of the bottom-left-back voxel of the image.
  mitk::Point3D WorldOrigin;

  /// The extents (number of slices) in world coordinate order: sagittal, coronal, axial.
  mitk::Vector3D WorldExtents;

  /// The spacings (distance of slices) in world coordinate order: sagittal, coronal, axial.
  mitk::Vector3D WorldSpacings;

  /// A up directions of the axes in world coordinate order: sagittal, coronal, axial.
  /// If the updirection is +1 then higher voxel index means higher mm position in world,
  /// i.e. moving towards the top, right or front.
  /// If the updirection is -1 then higher voxel index means lower mm position in world,
  /// i.e. moving towards the bottom, left or back.
  mitk::Vector3D WorldUpDirections;

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
  d->WorldOrigin.Fill(0.0);
  d->WorldExtents.Fill(0.0);
  d->WorldSpacings.Fill(1.0);
  d->WorldUpDirections[0] = 0;
  d->WorldUpDirections[1] = 0;
  d->WorldUpDirections[2] = 0;
  d->Viewer = 0;
  d->VisibilityManager = 0;
  d->InteractiveMode = false;

  d->SelectedPositionChanged = SIGNAL(SelectedPositionChanged(niftkSingleViewerWidget*, const mitk::Point3D&));
  d->SelectedTimeStepChanged = SIGNAL(SelectedTimeStepChanged(niftkSingleViewerWidget*, int));
  d->CursorPositionChanged = SIGNAL(CursorPositionChanged(niftkSingleViewerWidget*, MIDASOrientation, const mitk::Vector2D&));
  d->ScaleFactorChanged = SIGNAL(ScaleFactorChanged(niftkSingleViewerWidget*, MIDASOrientation, double));
  d->CursorPositionBindingChanged = SIGNAL(CursorPositionBindingChanged(niftkSingleViewerWidget*, bool));
  d->ScaleFactorBindingChanged = SIGNAL(ScaleFactorBindingChanged(niftkSingleViewerWidget*, bool));
  d->WindowLayoutChanged = SIGNAL(WindowLayoutChanged(niftkSingleViewerWidget*, WindowLayout));
  d->GeometryChanged = SIGNAL(GeometryChanged(niftkSingleViewerWidget*, const mitk::TimeGeometry*));
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
mitk::Point3D niftkSingleViewerWidgetTestClass::GetWorldOrigin(const mitk::Geometry3D* geometry)
{
  const mitk::AffineTransform3D* affineTransform = geometry->GetIndexToWorldTransform();
  itk::Matrix<double, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
  affineTransformMatrix.GetVnlMatrix().normalize_columns();
  mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseTransformMatrix = affineTransformMatrix.GetInverse();

  int dominantAxisRL = itk::Function::Max3(inverseTransformMatrix[0][0], inverseTransformMatrix[1][0], inverseTransformMatrix[2][0]);
  int signRL = itk::Function::Sign(inverseTransformMatrix[dominantAxisRL][0]);
  int dominantAxisAP = itk::Function::Max3(inverseTransformMatrix[0][1], inverseTransformMatrix[1][1], inverseTransformMatrix[2][1]);
  int signAP = itk::Function::Sign(inverseTransformMatrix[dominantAxisAP][1]);
  int dominantAxisSI = itk::Function::Max3(inverseTransformMatrix[0][2], inverseTransformMatrix[1][2], inverseTransformMatrix[2][2]);
  int signSI = itk::Function::Sign(inverseTransformMatrix[dominantAxisSI][2]);

  int permutedAxes[3] = {dominantAxisRL, dominantAxisAP, dominantAxisSI};
  int flippedAxes[3] = {signRL, signAP, signSI};
  const mitk::Vector3D& spacings = geometry->GetSpacing();
  double permutedSpacing[3] = {spacings[permutedAxes[0]], spacings[permutedAxes[1]], spacings[permutedAxes[2]]};

  mitk::Point3D originInVx;
  for (int i = 0; i < 3; ++i)
  {
    originInVx[permutedAxes[i]] = flippedAxes[i] > 0 ? 0 : geometry->GetExtent(permutedAxes[i]) - 1;
  }

  mitk::Point3D originInMm;
  geometry->IndexToWorld(originInVx, originInMm);

  return originInMm;
}


// --------------------------------------------------------------------------
mitk::Vector3D niftkSingleViewerWidgetTestClass::GetWorldUpDirections(const mitk::Geometry3D* geometry)
{
  const mitk::AffineTransform3D* affineTransform = geometry->GetIndexToWorldTransform();
  itk::Matrix<double, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
  affineTransformMatrix.GetVnlMatrix().normalize_columns();
  mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseTransformMatrix = affineTransformMatrix.GetInverse();

  int dominantAxisRL = itk::Function::Max3(inverseTransformMatrix[0][0], inverseTransformMatrix[1][0], inverseTransformMatrix[2][0]);
  int signRL = itk::Function::Sign(inverseTransformMatrix[dominantAxisRL][0]);
  int dominantAxisAP = itk::Function::Max3(inverseTransformMatrix[0][1], inverseTransformMatrix[1][1], inverseTransformMatrix[2][1]);
  int signAP = itk::Function::Sign(inverseTransformMatrix[dominantAxisAP][1]);
  int dominantAxisSI = itk::Function::Max3(inverseTransformMatrix[0][2], inverseTransformMatrix[1][2], inverseTransformMatrix[2][2]);
  int signSI = itk::Function::Sign(inverseTransformMatrix[dominantAxisSI][2]);

  mitk::Vector3D worldUpDirections;
  worldUpDirections[0] = signRL;
  worldUpDirections[1] = signAP;
  worldUpDirections[2] = signSI;

  return worldUpDirections;
}


// --------------------------------------------------------------------------
mitk::Point3D niftkSingleViewerWidgetTestClass::GetWorldBottomLeftBackCorner(const mitk::Geometry3D* geometry)
{
  const mitk::AffineTransform3D* affineTransform = geometry->GetIndexToWorldTransform();
  itk::Matrix<double, 3, 3> affineTransformMatrix = affineTransform->GetMatrix();
  affineTransformMatrix.GetVnlMatrix().normalize_columns();
  mitk::AffineTransform3D::MatrixType::InternalMatrixType inverseTransformMatrix = affineTransformMatrix.GetInverse();

  int dominantAxisRL = itk::Function::Max3(inverseTransformMatrix[0][0], inverseTransformMatrix[1][0], inverseTransformMatrix[2][0]);
  int signRL = itk::Function::Sign(inverseTransformMatrix[dominantAxisRL][0]);
  int dominantAxisAP = itk::Function::Max3(inverseTransformMatrix[0][1], inverseTransformMatrix[1][1], inverseTransformMatrix[2][1]);
  int signAP = itk::Function::Sign(inverseTransformMatrix[dominantAxisAP][1]);
  int dominantAxisSI = itk::Function::Max3(inverseTransformMatrix[0][2], inverseTransformMatrix[1][2], inverseTransformMatrix[2][2]);
  int signSI = itk::Function::Sign(inverseTransformMatrix[dominantAxisSI][2]);

  int permutedAxes[3] = {dominantAxisRL, dominantAxisAP, dominantAxisSI};
  int upDirections[3] = {signRL, signAP, signSI};
  const mitk::Vector3D& spacings = geometry->GetSpacing();
  double permutedSpacing[3] = {spacings[permutedAxes[0]], spacings[permutedAxes[1]], spacings[permutedAxes[2]]};

  mitk::Point3D originInVx;
  for (int i = 0; i < 3; ++i)
  {
    originInVx[permutedAxes[i]] = upDirections[i] > 0 ? 0 : geometry->GetExtent(permutedAxes[i]) - 1;
  }

  mitk::Point3D bottomLeftBackCorner;
  geometry->IndexToWorld(originInVx, bottomLeftBackCorner);

  if (geometry->GetImageGeometry())
  {
    bottomLeftBackCorner[0] -= 0.5 * permutedSpacing[0];
    bottomLeftBackCorner[1] -= 0.5 * permutedSpacing[1];
    bottomLeftBackCorner[2] -= 0.5 * permutedSpacing[2];
  }
  else
  {
    if (permutedAxes[0] == 0 && permutedAxes[1] == 1 && permutedAxes[2] == 2) // Axial
    {
      /// TODO !!! This line should not be needed. !!!
      bottomLeftBackCorner[1] -= permutedSpacing[1];
    }
    else if (permutedAxes[0] == 2 && permutedAxes[1] == 0 && permutedAxes[2] == 1) // Sagittal
    {
    }
    else if (permutedAxes[0] == 0 && permutedAxes[1] == 2 && permutedAxes[2] == 1) // Coronal
    {
    }
    else
    {
      assert(false);
    }
  }

  return bottomLeftBackCorner;
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
mitk::Vector2D niftkSingleViewerWidgetTestClass::GetCentrePosition(int windowIndex)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::BaseRenderer* renderer = d->Viewer->GetRenderWindows()[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  const mitk::Geometry2D* worldGeometry2D = renderer->GetCurrentWorldGeometry2D();

  mitk::Point3D centreInMm = worldGeometry2D->GetCenter();
  mitk::Point2D centreInMm2D;
  displayGeometry->Map(centreInMm, centreInMm2D);
  mitk::Point2D centreInPx2D;
  displayGeometry->WorldToDisplay(centreInMm2D, centreInPx2D);

  mitk::Vector2D centrePosition;
  centrePosition[0] = centreInPx2D[0] / renderer->GetSizeX();
  centrePosition[1] = centreInPx2D[1] / renderer->GetSizeY();

  return centrePosition;
}


// --------------------------------------------------------------------------
std::vector<mitk::Vector2D> niftkSingleViewerWidgetTestClass::GetCentrePositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  const std::vector<QmitkRenderWindow*>& renderWindows = d->Viewer->GetRenderWindows();

  std::vector<mitk::Vector2D> centrePositions(3);
  for (std::size_t windowIndex = 0; windowIndex < 3; ++windowIndex)
  {
    if (renderWindows[windowIndex]->isVisible())
    {
      centrePositions[windowIndex] = this->GetCentrePosition(windowIndex);
    }
    else
    {
      /// This is not necessary but makes the printed results easier to read.
      centrePositions[windowIndex][0] = std::numeric_limits<double>::quiet_NaN();
      centrePositions[windowIndex][1] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  return centrePositions;
}


// --------------------------------------------------------------------------
mitk::Point3D niftkSingleViewerWidgetTestClass::GetWorldPositionAtDisplayPosition(int windowIndex, const mitk::Vector2D& displayPosition)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::BaseRenderer* renderer = d->Viewer->GetRenderWindows()[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();

  mitk::Point2D displayPositionInPx;
  displayPositionInPx[0] = displayPosition[0] * renderer->GetSizeX();
  displayPositionInPx[1] = displayPosition[1] * renderer->GetSizeY();

  mitk::Point2D worldPosition2D;
  displayGeometry->DisplayToWorld(displayPositionInPx, worldPosition2D);

  mitk::Point3D worldPosition;
  displayGeometry->Map(worldPosition2D, worldPosition);

  return worldPosition;
}


// --------------------------------------------------------------------------
mitk::Vector2D niftkSingleViewerWidgetTestClass::GetDisplayPositionAtWorldPosition(int windowIndex, const mitk::Point3D& worldPosition)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::BaseRenderer* renderer = d->Viewer->GetRenderWindows()[windowIndex]->GetRenderer();
  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();

  mitk::Point2D worldPosition2D;
  displayGeometry->Map(worldPosition, worldPosition2D);

  mitk::Point2D displayPositionInPx;
  displayGeometry->WorldToDisplay(worldPosition2D, displayPositionInPx);

  mitk::Vector2D displayPosition;
  displayPosition[0] = displayPositionInPx[0] / renderer->GetSizeX();
  displayPosition[1] = displayPositionInPx[1] / renderer->GetSizeY();

  return displayPosition;
}


// --------------------------------------------------------------------------
double niftkSingleViewerWidgetTestClass::GetVoxelCentreCoordinate(int axis, double position)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  return std::floor((position + 0.5 * d->WorldSpacings[axis]) / d->WorldSpacings[axis]) * d->WorldSpacings[axis];
}


// --------------------------------------------------------------------------
mitk::Point3D niftkSingleViewerWidgetTestClass::GetVoxelCentrePosition(const mitk::Point3D& position)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::Point3D voxelCentrePosition;
  for (int axis = 0; axis < 3; ++axis)
  {
    voxelCentrePosition[axis] = this->GetVoxelCentreCoordinate(axis, position[axis]);
  }

  return voxelCentrePosition;
}


// --------------------------------------------------------------------------
bool niftkSingleViewerWidgetTestClass::Equals(const mitk::Point3D& selectedPosition1, const mitk::Point3D& selectedPosition2, double tolerance)
{
  Q_D(niftkSingleViewerWidgetTestClass);

  for (int i = 0; i < 3; ++i)
  {
    double epsilon = tolerance >= 0 ? tolerance : d->WorldSpacings[i] / 2.0;

    if (std::abs(selectedPosition1[i] - selectedPosition2[i]) > epsilon)
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
  Q_D(const niftkSingleViewerWidgetTestClass);

  const std::vector<QmitkRenderWindow*>& renderWindows = d->Viewer->GetRenderWindows();

  return cursorPositions1.size() == std::size_t(3)
      && cursorPositions2.size() == std::size_t(3)
      && (!renderWindows[0]->isVisible() || Self::Equals(cursorPositions1[0], cursorPositions2[0], tolerance))
      && (!renderWindows[1]->isVisible() || Self::Equals(cursorPositions1[1], cursorPositions2[1], tolerance))
      && (!renderWindows[2]->isVisible() || Self::Equals(cursorPositions1[2], cursorPositions2[2], tolerance));
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

  /// Note:
  /// If the file is a DICOM file then all the DICOM images from the same directory
  /// will be opened. Therefore, we check if number of loaded images is positive.
  MITK_TEST_CONDITION_REQUIRED(allImages->size() > 0, ".. Test image loaded.");

  d->ImageNode = (*allImages)[0];

  d->VisibilityManager = new niftkMultiViewerVisibilityManager(d->DataStorage);
  d->VisibilityManager->SetInterpolationType(DNDDISPLAY_CUBIC_INTERPOLATION);
  d->VisibilityManager->SetDefaultWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->VisibilityManager->SetDropType(DNDDISPLAY_DROP_SINGLE);

  d->Image = dynamic_cast<mitk::Image*>(d->ImageNode->GetData());

  d->WorldOrigin = Self::GetWorldOrigin(d->Image->GetGeometry());
  mitk::GetExtentsInVxInWorldCoordinateOrder(d->Image, d->WorldExtents);
  mitk::GetSpacingInWorldCoordinateOrder(d->Image, d->WorldSpacings);

//  MITK_INFO << "Image origin: " << d->Image->GetGeometry()->GetOrigin();
//  MITK_INFO << "World origin: " << d->WorldOrigin;
//  MITK_INFO << "World extents: " << d->WorldExtents;
//  MITK_INFO << "World spacings: " << d->WorldSpacings;
//  MITK_INFO << "World up directions: " << d->WorldUpDirections;
//  mitk::Point3D bottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(d->Image->GetGeometry());
//  MITK_INFO << "World bottom left back corner: " << bottomLeftBackCorner;
//  MITK_INFO << "World centre: " << d->Image->GetGeometry()->GetCenter();
//  for (int i = 0; i < 8; ++i)
//  {
//    MITK_INFO << "corner point " << i << ": " << d->Image->GetGeometry()->GetCornerPoint(i & 4, i & 2, i & 1);
//  }

  d->WorldUpDirections = Self::GetWorldUpDirections(d->Image->GetGeometry());
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
  d->Viewer->SetBackgroundColour(backgroundColour);
  d->Viewer->SetShow3DWindowIn2x2WindowLayout(true);
  d->Viewer->SetRememberSettingsPerWindowLayout(false);
  d->Viewer->SetDisplayInteractionsEnabled(true);
  d->Viewer->SetCursorPositionBinding(false);
  d->Viewer->SetScaleFactorBinding(false);
  d->Viewer->SetDefaultSingleWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->Viewer->SetDefaultMultiWindowLayout(WINDOW_LAYOUT_ORTHO);

//  d->VisibilityManager->connect(d->Viewer, SIGNAL(NodesDropped(niftkSingleViewerWidget*, std::vector<mitk::DataNode*>)), SLOT(OnNodesDropped(niftkSingleViewerWidget*, std::vector<mitk::DataNode*>)), Qt::DirectConnection);

  d->VisibilityManager->RegisterViewer(d->Viewer);

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

  d->VisibilityManager->DeregisterViewers();

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

  d->VisibilityManager->OnNodesDropped(d->Viewer, nodes);
//  d->Viewer->OnNodesDropped(0, nodes);
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
void niftkSingleViewerWidgetTestClass::testGetTimeGeometry()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  const mitk::TimeGeometry* imageTimeGeometry = d->Image->GetTimeGeometry();
  const mitk::Geometry3D* imageGeometry = imageTimeGeometry->GetGeometryForTimePoint(0);

  const mitk::TimeGeometry* viewerTimeGeometry = d->Viewer->GetTimeGeometry();
  const mitk::Geometry3D* viewerGeometry = viewerTimeGeometry->GetGeometryForTimePoint(0);

  QVERIFY(imageTimeGeometry == viewerTimeGeometry);
  QVERIFY(imageGeometry == viewerGeometry);

  mitk::BaseRenderer* axialRenderer = d->AxialWindow->GetRenderer();
  mitk::BaseRenderer* sagittalRenderer = d->SagittalWindow->GetRenderer();
  mitk::BaseRenderer* coronalRenderer = d->CoronalWindow->GetRenderer();

  const mitk::Geometry3D* axialGeometry = axialRenderer->GetWorldGeometry();
  const mitk::Geometry3D* sagittalGeometry = sagittalRenderer->GetWorldGeometry();
  const mitk::Geometry3D* coronalGeometry = coronalRenderer->GetWorldGeometry();

  QVERIFY(axialGeometry);
  QVERIFY(sagittalGeometry);
  QVERIFY(coronalGeometry);

  mitk::Point3D axialOrigin = axialGeometry->GetOrigin();
  mitk::Point3D sagittalOrigin = sagittalGeometry->GetOrigin();
  mitk::Point3D coronalOrigin = coronalGeometry->GetOrigin();
  mitk::Point3D axialCentre = axialGeometry->GetCenter();
  mitk::Point3D sagittalCentre = sagittalGeometry->GetCenter();
  mitk::Point3D coronalCentre = coronalGeometry->GetCenter();
  mitk::Point3D axialBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(axialGeometry);
  mitk::Point3D sagittalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(sagittalGeometry);
  mitk::Point3D coronalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(coronalGeometry);

//  MITK_INFO << "axial origin: " << axialOrigin;
//  MITK_INFO << "sagittal origin: " << sagittalOrigin;
//  MITK_INFO << "coronal origin: " << coronalOrigin;
//  MITK_INFO << "axial centre: " << axialCentre;
//  MITK_INFO << "sagittal centre: " << sagittalCentre;
//  MITK_INFO << "coronal centre: " << coronalCentre;
//  MITK_INFO << "axial bottom left back corner: " << axialBottomLeftBackCorner;
//  MITK_INFO << "sagittal bottom left back corner: " << sagittalBottomLeftBackCorner;
//  MITK_INFO << "coronal bottom left back corner: " << coronalBottomLeftBackCorner;

  const mitk::SlicedGeometry3D* axialSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(axialGeometry);
  const mitk::SlicedGeometry3D* sagittalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(sagittalGeometry);
  const mitk::SlicedGeometry3D* coronalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(coronalGeometry);

  QVERIFY(axialSlicedGeometry);
  QVERIFY(sagittalSlicedGeometry);
  QVERIFY(coronalSlicedGeometry);

  const mitk::PlaneGeometry* axialFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* sagittalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* coronalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* axialSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(1));
  const mitk::PlaneGeometry* sagittalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(1));
  const mitk::PlaneGeometry* coronalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(1));

  QVERIFY(axialFirstPlaneGeometry);
  QVERIFY(sagittalFirstPlaneGeometry);
  QVERIFY(coronalFirstPlaneGeometry);
  QVERIFY(axialSecondPlaneGeometry);
  QVERIFY(sagittalSecondPlaneGeometry);
  QVERIFY(coronalSecondPlaneGeometry);

  mitk::Point3D axialFirstPlaneOrigin = axialFirstPlaneGeometry->GetOrigin();
  mitk::Point3D sagittalFirstPlaneOrigin = sagittalFirstPlaneGeometry->GetOrigin();
  mitk::Point3D coronalFirstPlaneOrigin = coronalFirstPlaneGeometry->GetOrigin();
  mitk::Point3D axialFirstPlaneCentre = axialFirstPlaneGeometry->GetCenter();
  mitk::Point3D sagittalFirstPlaneCentre = sagittalFirstPlaneGeometry->GetCenter();
  mitk::Point3D coronalFirstPlaneCentre = coronalFirstPlaneGeometry->GetCenter();
  mitk::Point3D axialSecondPlaneOrigin = axialSecondPlaneGeometry->GetOrigin();
  mitk::Point3D sagittalSecondPlaneOrigin = sagittalSecondPlaneGeometry->GetOrigin();
  mitk::Point3D coronalSecondPlaneOrigin = coronalSecondPlaneGeometry->GetOrigin();
  mitk::Point3D axialSecondPlaneCentre = axialSecondPlaneGeometry->GetCenter();
  mitk::Point3D sagittalSecondPlaneCentre = sagittalSecondPlaneGeometry->GetCenter();
  mitk::Point3D coronalSecondPlaneCentre = coronalSecondPlaneGeometry->GetCenter();

//  MITK_INFO << "axial first plane origin: " << axialFirstPlaneOrigin;
//  MITK_INFO << "sagittal first plane origin: " << sagittalFirstPlaneOrigin;
//  MITK_INFO << "coronal first plane origin: " << coronalFirstPlaneOrigin;
//  MITK_INFO << "axial first plane centre: " << axialFirstPlaneCentre;
//  MITK_INFO << "sagittal first plane centre: " << sagittalFirstPlaneCentre;
//  MITK_INFO << "coronal first plane centre: " << coronalFirstPlaneCentre;
//  MITK_INFO << "axial second plane origin: " << axialSecondPlaneOrigin;
//  MITK_INFO << "sagittal second plane origin: " << sagittalSecondPlaneOrigin;
//  MITK_INFO << "coronal second plane origin: " << coronalSecondPlaneOrigin;
//  MITK_INFO << "axial second plane centre: " << axialSecondPlaneCentre;
//  MITK_INFO << "sagittal second plane centre: " << sagittalSecondPlaneCentre;
//  MITK_INFO << "coronal second plane centre: " << coronalSecondPlaneCentre;

  /// Note:
  /// According to the MITK documentation, the origin of a world geometry is
  /// always at the bottom-left-back voxel, and the mm coordinates increase
  /// from left to right (sagittal), from bottom to top (axial), and
  /// from back to front (coronal).

  mitk::Point3D worldBottomLeftBackCorner = d->WorldOrigin;
  worldBottomLeftBackCorner[0] -= d->WorldSpacings[0] / 2.0;
  worldBottomLeftBackCorner[1] -= d->WorldSpacings[1] / 2.0;
  worldBottomLeftBackCorner[2] -= d->WorldSpacings[2] / 2.0;

  mitk::Point3D worldCentre = worldBottomLeftBackCorner;
  worldCentre[0] += d->WorldExtents[0] * d->WorldSpacings[0] / 2.0;
  worldCentre[1] += d->WorldExtents[1] * d->WorldSpacings[1] / 2.0;
  worldCentre[2] += d->WorldExtents[2] * d->WorldSpacings[2] / 2.0;

//  MITK_INFO << "world bottom left back corner: " << worldBottomLeftBackCorner;
//  MITK_INFO << "world centre: " << worldCentre;

  /// -------------------------------------------------------------------------
  /// The viewer is now initialised with the world geometry from an image
  /// -------------------------------------------------------------------------

  /// Note:
  /// The renderer geometries are half voxel shifted along the renderer axis.


  mitk::Point3D expectedAxialOrigin = worldBottomLeftBackCorner;
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialOrigin[1] += d->WorldExtents[1] * d->WorldSpacings[1];
  mitk::Point3D expectedSagittalOrigin = worldBottomLeftBackCorner;
  mitk::Point3D expectedCoronalOrigin = worldBottomLeftBackCorner;

  mitk::Point3D expectedAxialCentre = worldCentre;
  mitk::Point3D expectedSagittalCentre = worldCentre;
  mitk::Point3D expectedCoronalCentre = worldCentre;

  mitk::Point3D expectedAxialFirstPlaneOrigin = worldBottomLeftBackCorner;
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialFirstPlaneOrigin[1] += d->WorldExtents[1] * d->WorldSpacings[1];
  expectedAxialFirstPlaneOrigin[2] += 0.5 * d->WorldSpacings[2];

  mitk::Point3D expectedSagittalFirstPlaneOrigin = worldBottomLeftBackCorner;
  expectedSagittalFirstPlaneOrigin[0] += 0.5 * d->WorldSpacings[0];

  mitk::Point3D expectedCoronalFirstPlaneOrigin = worldBottomLeftBackCorner;
  expectedCoronalFirstPlaneOrigin[1] += 0.5 * d->WorldSpacings[1];

  mitk::Point3D expectedAxialFirstPlaneCentre = expectedAxialFirstPlaneOrigin;
  expectedAxialFirstPlaneCentre[0] += 0.5 * d->WorldExtents[0] * d->WorldSpacings[0];
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialFirstPlaneCentre[1] -= 0.5 * d->WorldExtents[1] * d->WorldSpacings[1];
  expectedAxialFirstPlaneCentre[2] -= 0.5 * d->WorldSpacings[2];

  mitk::Point3D expectedSagittalFirstPlaneCentre = expectedSagittalFirstPlaneOrigin;
  expectedSagittalFirstPlaneCentre[0] += 0.5 * d->WorldSpacings[0];
  expectedSagittalFirstPlaneCentre[1] += 0.5 * d->WorldExtents[1] * d->WorldSpacings[1];
  expectedSagittalFirstPlaneCentre[2] += 0.5 * d->WorldExtents[2] * d->WorldSpacings[2];

  mitk::Point3D expectedCoronalFirstPlaneCentre = expectedCoronalFirstPlaneOrigin;
  expectedCoronalFirstPlaneCentre[0] += 0.5 * d->WorldExtents[0] * d->WorldSpacings[0];
  /// Why is this minus and not plus?
  expectedCoronalFirstPlaneCentre[1] -= 0.5 * d->WorldSpacings[1];
  expectedCoronalFirstPlaneCentre[2] += 0.5 * d->WorldExtents[2] * d->WorldSpacings[2];

  mitk::Point3D expectedAxialSecondPlaneOrigin = expectedAxialFirstPlaneOrigin;
  expectedAxialSecondPlaneOrigin[2] += d->WorldSpacings[2];

  mitk::Point3D expectedSagittalSecondPlaneOrigin = expectedSagittalFirstPlaneOrigin;
  expectedSagittalSecondPlaneOrigin[0] += d->WorldSpacings[0];

  mitk::Point3D expectedCoronalSecondPlaneOrigin = expectedCoronalFirstPlaneOrigin;
  expectedCoronalSecondPlaneOrigin[1] += d->WorldSpacings[1];

  mitk::Point3D expectedAxialSecondPlaneCentre = expectedAxialFirstPlaneCentre;
  expectedAxialSecondPlaneCentre[2] += d->WorldSpacings[2];

  mitk::Point3D expectedSagittalSecondPlaneCentre = expectedSagittalFirstPlaneCentre;
  expectedSagittalSecondPlaneCentre[0] += d->WorldSpacings[0];

  mitk::Point3D expectedCoronalSecondPlaneCentre = expectedCoronalFirstPlaneCentre;
  expectedCoronalSecondPlaneCentre[1] += d->WorldSpacings[1];

//  MITK_INFO << "expected axial origin: " << expectedAxialOrigin;
//  MITK_INFO << "expected sagittal origin: " << expectedSagittalOrigin;
//  MITK_INFO << "expected coronal origin: " << expectedCoronalOrigin;
//  MITK_INFO << "expected axial centre: " << expectedAxialCentre;
//  MITK_INFO << "expected sagittal centre: " << expectedSagittalCentre;
//  MITK_INFO << "expected coronal centre: " << expectedCoronalCentre;
//  MITK_INFO << "expected axial first plane origin: " << expectedAxialFirstPlaneOrigin;
//  MITK_INFO << "expected sagittal first plane origin: " << expectedSagittalFirstPlaneOrigin;
//  MITK_INFO << "expected coronal first plane origin: " << expectedCoronalFirstPlaneOrigin;
//  MITK_INFO << "expected axial first plane centre: " << expectedAxialFirstPlaneCentre;
//  MITK_INFO << "expected sagittal first plane centre: " << expectedSagittalFirstPlaneCentre;
//  MITK_INFO << "expected coronal first plane centre: " << expectedCoronalFirstPlaneCentre;
//  MITK_INFO << "expected axial second plane origin: " << expectedAxialSecondPlaneOrigin;
//  MITK_INFO << "expected sagittal second plane origin: " << expectedSagittalSecondPlaneOrigin;
//  MITK_INFO << "expected coronal second plane origin: " << expectedCoronalSecondPlaneOrigin;
//  MITK_INFO << "expected axial second plane centre: " << expectedAxialSecondPlaneCentre;
//  MITK_INFO << "expected sagittal second plane centre: " << expectedSagittalSecondPlaneCentre;
//  MITK_INFO << "expected coronal second plane centre: " << expectedCoronalSecondPlaneCentre;

//  QVERIFY(Self::Equals(axialOrigin, expectedAxialOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalOrigin, expectedSagittalOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalOrigin, expectedCoronalOrigin, 0.001));
//  QVERIFY(Self::Equals(axialCentre, expectedAxialCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalCentre, expectedSagittalCentre, 0.001));
//  QVERIFY(Self::Equals(coronalCentre, expectedCoronalCentre, 0.001));
//  QVERIFY(Self::Equals(axialBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(sagittalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(coronalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneOrigin, expectedAxialFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneOrigin, expectedSagittalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneOrigin, expectedCoronalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneCentre, expectedAxialFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneCentre, expectedSagittalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneCentre, expectedCoronalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneOrigin, expectedAxialSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneOrigin, expectedSagittalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneOrigin, expectedCoronalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneCentre, expectedAxialSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneCentre, expectedSagittalSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneCentre, expectedCoronalSecondPlaneCentre, 0.001));
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedSlice2()
{
//  Q_D(niftkSingleViewerWidgetTestClass);

//  const mitk::TimeGeometry* imageTimeGeometry = d->Image->GetTimeGeometry();
//  const mitk::Geometry3D* imageGeometry = imageTimeGeometry->GetGeometryForTimePoint(0);

//  const mitk::TimeGeometry* viewerTimeGeometry = d->Viewer->GetTimeGeometry();
//  const mitk::Geometry3D* viewerGeometry = viewerTimeGeometry->GetGeometryForTimePoint(0);

//  QVERIFY(imageTimeGeometry == viewerTimeGeometry);
//  QVERIFY(imageGeometry == viewerGeometry);

//  mitk::BaseRenderer* axialRenderer = d->AxialWindow->GetRenderer();
//  mitk::BaseRenderer* sagittalRenderer = d->SagittalWindow->GetRenderer();
//  mitk::BaseRenderer* coronalRenderer = d->CoronalWindow->GetRenderer();

//  const mitk::TimeGeometry::Pointer axialTimeGeometry = axialRenderer->GetTimeWorldGeometry()->Clone();
//  const mitk::TimeGeometry::Pointer sagittalTimeGeometry = sagittalRenderer->GetTimeWorldGeometry()->Clone();
//  const mitk::TimeGeometry::Pointer coronalTimeGeometry = coronalRenderer->GetTimeWorldGeometry()->Clone();

//  const mitk::Geometry3D* axialGeometry = axialRenderer->GetWorldGeometry();
//  const mitk::Geometry3D* sagittalGeometry = sagittalRenderer->GetWorldGeometry();
//  const mitk::Geometry3D* coronalGeometry = coronalRenderer->GetWorldGeometry();

//  QVERIFY(axialGeometry);
//  QVERIFY(sagittalGeometry);
//  QVERIFY(coronalGeometry);

//  int sagittalSlices = d->Viewer->GetMaxSlice(MIDAS_ORIENTATION_SAGITTAL);
//  int coronalSlices = d->Viewer->GetMaxSlice(MIDAS_ORIENTATION_CORONAL);
//  int axialSlices = d->Viewer->GetMaxSlice(MIDAS_ORIENTATION_AXIAL);

//  /// -------------------------------------------------------------------------
//  /// Viewer initialised with the world geometry from an image
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry from an image.";

//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);

//  for (int sagittalIndex = 0; sagittalIndex <= sagittalSlices; sagittalIndex += 32)
//  {
//    d->StateTester->Clear();
////    MITK_INFO << "Setting index: " << sagittalIndex << " 0 0";
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, sagittalIndex);

//    int actualSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//    QVERIFY(actualSagittalSlice == sagittalIndex);

//    for (int coronalIndex = 0; coronalIndex <= coronalSlices; coronalIndex += 32)
//    {
//      d->StateTester->Clear();
////      MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " 0";
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalIndex);

//      int actualCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//      QVERIFY(actualCoronalSlice == coronalIndex);

//      for (int axialIndex = 0; axialIndex <= axialSlices; axialIndex += 32)
//      {
//        d->StateTester->Clear();
////        MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " " << axialIndex;
//        d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, axialIndex);

//        int actualAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//        QVERIFY(actualAxialSlice == axialIndex);
//      }
//      d->StateTester->Clear();
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);
//    }
//    d->StateTester->Clear();
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  }
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);

//  /// -------------------------------------------------------------------------
//  /// Viewer initialised with the world geometry of an axial renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of an axial renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(axialTimeGeometry);

//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);

//  for (int sagittalIndex = 0; sagittalIndex <= sagittalSlices; sagittalIndex += 32)
//  {
//    d->StateTester->Clear();
////    MITK_INFO << "Setting index: " << sagittalIndex << " 0 0";
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, sagittalIndex);

//    int actualSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//    QVERIFY(actualSagittalSlice == sagittalIndex);

//    for (int coronalIndex = 0; coronalIndex <= coronalSlices; coronalIndex += 32)
//    {
//      d->StateTester->Clear();
////      MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " 0";
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalIndex);

//      int actualCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//      QVERIFY(actualCoronalSlice == coronalIndex);

//      for (int axialIndex = 0; axialIndex <= axialSlices; axialIndex += 32)
//      {
//        d->StateTester->Clear();
////        MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " " << axialIndex;
//        d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, axialIndex);

//        int actualAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//        QVERIFY(actualAxialSlice == axialIndex);
//      }
//      d->StateTester->Clear();
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);
//    }
//    d->StateTester->Clear();
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  }
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);

//  /// -------------------------------------------------------------------------
//  /// Viewer initialised with the world geometry of an sagittal renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of an sagittal renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(sagittalTimeGeometry);

//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);

//  for (int sagittalIndex = 0; sagittalIndex <= sagittalSlices; sagittalIndex += 32)
//  {
//    d->StateTester->Clear();
////    MITK_INFO << "Setting index: " << sagittalIndex << " 0 0";
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, sagittalIndex);

//    int actualSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//    QVERIFY(actualSagittalSlice == sagittalIndex);

//    for (int coronalIndex = 0; coronalIndex <= coronalSlices; coronalIndex += 32)
//    {
//      d->StateTester->Clear();
////      MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " 0";
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalIndex);

//      int actualCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//      QVERIFY(actualCoronalSlice == coronalIndex);

//      for (int axialIndex = 0; axialIndex <= axialSlices; axialIndex += 32)
//      {
//        d->StateTester->Clear();
////        MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " 0";
//        d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, axialIndex);

//        int actualAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//        QVERIFY(actualAxialSlice == axialIndex);
//      }
//      d->StateTester->Clear();
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);
//    }
//    d->StateTester->Clear();
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  }
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);

//  /// -------------------------------------------------------------------------
//  /// Viewer initialised with the world geometry of an coronal renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of an coronal renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(coronalTimeGeometry);

//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);

//  for (int sagittalIndex = 0; sagittalIndex <= sagittalSlices; sagittalIndex += 32)
//  {
//    d->StateTester->Clear();
////    MITK_INFO << "Setting index: " << sagittalIndex << " 0 0";
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, sagittalIndex);

//    int actualSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//    QVERIFY(actualSagittalSlice == sagittalIndex);

//    for (int coronalIndex = 0; coronalIndex <= coronalSlices; coronalIndex += 32)
//    {
//      d->StateTester->Clear();
////      MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " 0";
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalIndex);

//      int actualCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//      QVERIFY(actualCoronalSlice == coronalIndex);

//      for (int axialIndex = 0; axialIndex <= axialSlices; axialIndex += 32)
//      {
//        d->StateTester->Clear();
////        MITK_INFO << "Setting index: " << sagittalIndex << " " << coronalIndex << " " << axialIndex;
//        d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, axialIndex);

//        int actualAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//        QVERIFY(actualAxialSlice == axialIndex);
//      }
//      d->StateTester->Clear();
//      d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_AXIAL, 0);
//    }
//    d->StateTester->Clear();
//    d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, 0);
//  }
//  d->StateTester->Clear();
//  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL, 0);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetTimeGeometry()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  const mitk::TimeGeometry* imageTimeGeometry = d->Image->GetTimeGeometry();
  const mitk::Geometry3D* imageGeometry = imageTimeGeometry->GetGeometryForTimePoint(0);

  const mitk::TimeGeometry* viewerTimeGeometry = d->Viewer->GetTimeGeometry();
  const mitk::Geometry3D* viewerGeometry = viewerTimeGeometry->GetGeometryForTimePoint(0);

  QVERIFY(imageTimeGeometry == viewerTimeGeometry);
  QVERIFY(imageGeometry == viewerGeometry);

  mitk::BaseRenderer* axialRenderer = d->AxialWindow->GetRenderer();
  mitk::BaseRenderer* sagittalRenderer = d->SagittalWindow->GetRenderer();
  mitk::BaseRenderer* coronalRenderer = d->CoronalWindow->GetRenderer();

  const mitk::TimeGeometry::Pointer axialTimeGeometry = axialRenderer->GetWorldTimeGeometry()->Clone();
  const mitk::TimeGeometry::Pointer sagittalTimeGeometry = sagittalRenderer->GetWorldTimeGeometry()->Clone();
  const mitk::TimeGeometry::Pointer coronalTimeGeometry = coronalRenderer->GetWorldTimeGeometry()->Clone();

//  MITK_INFO << "axial time geometry: " << axialTimeGeometry;
//  MITK_INFO << "sagittal time geometry: " << sagittalTimeGeometry;
//  MITK_INFO << "coronal time geometry: " << coronalTimeGeometry;

//  MITK_INFO << "Viewer initialised with the world geometry from an image geometry: ";

  const mitk::Geometry3D* axialGeometry = axialRenderer->GetWorldGeometry();
  const mitk::Geometry3D* sagittalGeometry = sagittalRenderer->GetWorldGeometry();
  const mitk::Geometry3D* coronalGeometry = coronalRenderer->GetWorldGeometry();

  QVERIFY(axialGeometry);
  QVERIFY(sagittalGeometry);
  QVERIFY(coronalGeometry);

//  MITK_INFO << "axial geometry: " << axialGeometry;
//  MITK_INFO << "sagittal geometry: " << sagittalGeometry;
//  MITK_INFO << "coronal geometry: " << coronalGeometry;

  mitk::Point3D axialOrigin = axialGeometry->GetOrigin();
  mitk::Point3D sagittalOrigin = sagittalGeometry->GetOrigin();
  mitk::Point3D coronalOrigin = coronalGeometry->GetOrigin();
  mitk::Point3D axialCentre = axialGeometry->GetCenter();
  mitk::Point3D sagittalCentre = sagittalGeometry->GetCenter();
  mitk::Point3D coronalCentre = coronalGeometry->GetCenter();
  mitk::Point3D axialBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(axialGeometry);
  mitk::Point3D sagittalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(sagittalGeometry);
  mitk::Point3D coronalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(coronalGeometry);

  const mitk::SlicedGeometry3D* axialSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(axialGeometry);
  const mitk::SlicedGeometry3D* sagittalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(sagittalGeometry);
  const mitk::SlicedGeometry3D* coronalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(coronalGeometry);

  QVERIFY(axialSlicedGeometry);
  QVERIFY(sagittalSlicedGeometry);
  QVERIFY(coronalSlicedGeometry);

  const mitk::PlaneGeometry* axialFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* sagittalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* coronalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(0));
  const mitk::PlaneGeometry* axialSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(1));
  const mitk::PlaneGeometry* sagittalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(1));
  const mitk::PlaneGeometry* coronalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(1));

  QVERIFY(axialFirstPlaneGeometry);
  QVERIFY(sagittalFirstPlaneGeometry);
  QVERIFY(coronalFirstPlaneGeometry);
  QVERIFY(axialSecondPlaneGeometry);
  QVERIFY(sagittalSecondPlaneGeometry);
  QVERIFY(coronalSecondPlaneGeometry);

  mitk::Point3D axialFirstPlaneOrigin = axialFirstPlaneGeometry->GetOrigin();
  mitk::Point3D sagittalFirstPlaneOrigin = sagittalFirstPlaneGeometry->GetOrigin();
  mitk::Point3D coronalFirstPlaneOrigin = coronalFirstPlaneGeometry->GetOrigin();
  mitk::Point3D axialFirstPlaneCentre = axialFirstPlaneGeometry->GetCenter();
  mitk::Point3D sagittalFirstPlaneCentre = sagittalFirstPlaneGeometry->GetCenter();
  mitk::Point3D coronalFirstPlaneCentre = coronalFirstPlaneGeometry->GetCenter();
  mitk::Point3D axialSecondPlaneOrigin = axialSecondPlaneGeometry->GetOrigin();
  mitk::Point3D sagittalSecondPlaneOrigin = sagittalSecondPlaneGeometry->GetOrigin();
  mitk::Point3D coronalSecondPlaneOrigin = coronalSecondPlaneGeometry->GetOrigin();
  mitk::Point3D axialSecondPlaneCentre = axialSecondPlaneGeometry->GetCenter();
  mitk::Point3D sagittalSecondPlaneCentre = sagittalSecondPlaneGeometry->GetCenter();
  mitk::Point3D coronalSecondPlaneCentre = coronalSecondPlaneGeometry->GetCenter();

//  MITK_INFO << "axial origin: " << axialOrigin;
//  MITK_INFO << "sagittal origin: " << sagittalOrigin;
//  MITK_INFO << "coronal origin: " << coronalOrigin;
//  MITK_INFO << "axial centre: " << axialCentre;
//  MITK_INFO << "sagittal centre: " << sagittalCentre;
//  MITK_INFO << "coronal centre: " << coronalCentre;
//  MITK_INFO << "axial bottom left back corner: " << axialBottomLeftBackCorner;
//  MITK_INFO << "sagittal bottom left back corner: " << sagittalBottomLeftBackCorner;
//  MITK_INFO << "coronal bottom left back corner: " << coronalBottomLeftBackCorner;
//  MITK_INFO << "axial first plane origin: " << axialFirstPlaneOrigin;
//  MITK_INFO << "sagittal first plane origin: " << sagittalFirstPlaneOrigin;
//  MITK_INFO << "coronal first plane origin: " << coronalFirstPlaneOrigin;
//  MITK_INFO << "axial first plane centre: " << axialFirstPlaneCentre;
//  MITK_INFO << "sagittal first plane centre: " << sagittalFirstPlaneCentre;
//  MITK_INFO << "coronal first plane centre: " << coronalFirstPlaneCentre;
//  MITK_INFO << "axial second plane origin: " << axialSecondPlaneOrigin;
//  MITK_INFO << "sagittal second plane origin: " << sagittalSecondPlaneOrigin;
//  MITK_INFO << "coronal second plane origin: " << coronalSecondPlaneOrigin;
//  MITK_INFO << "axial second plane centre: " << axialSecondPlaneCentre;
//  MITK_INFO << "sagittal second plane centre: " << sagittalSecondPlaneCentre;
//  MITK_INFO << "coronal second plane centre: " << coronalSecondPlaneCentre;

  /// Note:
  /// According to the MITK documentation, the origin of a world geometry is
  /// always at of its bottom-left-back voxel, and the mm coordinates
  /// increase from left to right (sagittal), from bottom to top (axial), and
  /// from back to front (coronal).

  mitk::Point3D worldBottomLeftBackCorner = d->WorldOrigin;
  worldBottomLeftBackCorner[0] -= d->WorldSpacings[0] / 2.0;
  worldBottomLeftBackCorner[1] -= d->WorldSpacings[1] / 2.0;
  worldBottomLeftBackCorner[2] -= d->WorldSpacings[2] / 2.0;

  mitk::Point3D worldCentre = worldBottomLeftBackCorner;
  worldCentre[0] += d->WorldExtents[0] * d->WorldSpacings[0] / 2.0;
  worldCentre[1] += d->WorldExtents[1] * d->WorldSpacings[1] / 2.0;
  worldCentre[2] += d->WorldExtents[2] * d->WorldSpacings[2] / 2.0;

//  MITK_INFO << "world bottom left back corner: " << worldBottomLeftBackCorner;
//  MITK_INFO << "world centre: " << worldCentre;

  mitk::Point3D expectedAxialOrigin = worldBottomLeftBackCorner;
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialOrigin[1] += d->WorldExtents[1] * d->WorldSpacings[1];
  mitk::Point3D expectedSagittalOrigin = worldBottomLeftBackCorner;
  mitk::Point3D expectedCoronalOrigin = worldBottomLeftBackCorner;

  mitk::Point3D expectedAxialCentre = worldCentre;
  mitk::Point3D expectedSagittalCentre = worldCentre;
  mitk::Point3D expectedCoronalCentre = worldCentre;

  mitk::Point3D expectedAxialFirstPlaneOrigin = worldBottomLeftBackCorner;
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialFirstPlaneOrigin[1] += d->WorldExtents[1] * d->WorldSpacings[1];
  expectedAxialFirstPlaneOrigin[2] += 0.5 * d->WorldSpacings[2];

  mitk::Point3D expectedSagittalFirstPlaneOrigin = worldBottomLeftBackCorner;
  expectedSagittalFirstPlaneOrigin[0] += 0.5 * d->WorldSpacings[0];

  mitk::Point3D expectedCoronalFirstPlaneOrigin = worldBottomLeftBackCorner;
  expectedCoronalFirstPlaneOrigin[1] += 0.5 * d->WorldSpacings[1];

  mitk::Point3D expectedAxialFirstPlaneCentre = expectedAxialFirstPlaneOrigin;
  expectedAxialFirstPlaneCentre[0] += 0.5 * d->WorldExtents[0] * d->WorldSpacings[0];
  /// Why is the y axis of the axial renderer geometry flipped? Is this correct?
  expectedAxialFirstPlaneCentre[1] -= 0.5 * d->WorldExtents[1] * d->WorldSpacings[1];
  expectedAxialFirstPlaneCentre[2] -= 0.5 * d->WorldSpacings[2];

  mitk::Point3D expectedSagittalFirstPlaneCentre = expectedSagittalFirstPlaneOrigin;
  expectedSagittalFirstPlaneCentre[0] += 0.5 * d->WorldSpacings[0];
  expectedSagittalFirstPlaneCentre[1] += 0.5 * d->WorldExtents[1] * d->WorldSpacings[1];
  expectedSagittalFirstPlaneCentre[2] += 0.5 * d->WorldExtents[2] * d->WorldSpacings[2];

  mitk::Point3D expectedCoronalFirstPlaneCentre = expectedCoronalFirstPlaneOrigin;
  expectedCoronalFirstPlaneCentre[0] += 0.5 * d->WorldExtents[0] * d->WorldSpacings[0];
  /// Why is this minus and not plus?
  expectedCoronalFirstPlaneCentre[1] -= 0.5 * d->WorldSpacings[1];
  expectedCoronalFirstPlaneCentre[2] += 0.5 * d->WorldExtents[2] * d->WorldSpacings[2];

  mitk::Point3D expectedAxialSecondPlaneOrigin = expectedAxialFirstPlaneOrigin;
  expectedAxialSecondPlaneOrigin[2] += d->WorldSpacings[2];

  mitk::Point3D expectedSagittalSecondPlaneOrigin = expectedSagittalFirstPlaneOrigin;
  expectedSagittalSecondPlaneOrigin[0] += d->WorldSpacings[0];

  mitk::Point3D expectedCoronalSecondPlaneOrigin = expectedCoronalFirstPlaneOrigin;
  expectedCoronalSecondPlaneOrigin[1] += d->WorldSpacings[1];

  mitk::Point3D expectedAxialSecondPlaneCentre = expectedAxialFirstPlaneCentre;
  expectedAxialSecondPlaneCentre[2] += d->WorldSpacings[2];

  mitk::Point3D expectedSagittalSecondPlaneCentre = expectedSagittalFirstPlaneCentre;
  expectedSagittalSecondPlaneCentre[0] += d->WorldSpacings[0];

  mitk::Point3D expectedCoronalSecondPlaneCentre = expectedCoronalFirstPlaneCentre;
  expectedCoronalSecondPlaneCentre[1] += d->WorldSpacings[1];

//  MITK_INFO << "expected axial origin: " << expectedAxialOrigin;
//  MITK_INFO << "expected sagittal origin: " << expectedSagittalOrigin;
//  MITK_INFO << "expected coronal origin: " << expectedCoronalOrigin;
//  MITK_INFO << "expected axial centre: " << expectedAxialCentre;
//  MITK_INFO << "expected sagittal centre: " << expectedSagittalCentre;
//  MITK_INFO << "expected coronal centre: " << expectedCoronalCentre;
//  MITK_INFO << "expected axial first plane origin: " << expectedAxialFirstPlaneOrigin;
//  MITK_INFO << "expected sagittal first plane origin: " << expectedSagittalFirstPlaneOrigin;
//  MITK_INFO << "expected coronal first plane origin: " << expectedCoronalFirstPlaneOrigin;
//  MITK_INFO << "expected axial first plane centre: " << expectedAxialFirstPlaneCentre;
//  MITK_INFO << "expected sagittal first plane centre: " << expectedSagittalFirstPlaneCentre;
//  MITK_INFO << "expected coronal first plane centre: " << expectedCoronalFirstPlaneCentre;
//  MITK_INFO << "expected axial second plane origin: " << expectedAxialSecondPlaneOrigin;
//  MITK_INFO << "expected sagittal second plane origin: " << expectedSagittalSecondPlaneOrigin;
//  MITK_INFO << "expected coronal second plane origin: " << expectedCoronalSecondPlaneOrigin;
//  MITK_INFO << "expected axial second plane centre: " << expectedAxialSecondPlaneCentre;
//  MITK_INFO << "expected sagittal second plane centre: " << expectedSagittalSecondPlaneCentre;
//  MITK_INFO << "expected coronal second plane centre: " << expectedCoronalSecondPlaneCentre;

//  /// -------------------------------------------------------------------------
//  /// Viewer initialised with the world geometry of an axial renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of an axial renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(axialTimeGeometry);

//  axialGeometry = axialRenderer->GetWorldGeometry();
//  sagittalGeometry = sagittalRenderer->GetWorldGeometry();
//  coronalGeometry = coronalRenderer->GetWorldGeometry();

//  QVERIFY(axialGeometry);
//  QVERIFY(sagittalGeometry);
//  QVERIFY(coronalGeometry);

////  MITK_INFO << "axial geometry: " << axialGeometry;
////  MITK_INFO << "sagittal geometry: " << sagittalGeometry;
////  MITK_INFO << "coronal geometry: " << coronalGeometry;

//  axialSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(axialGeometry);
//  sagittalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(sagittalGeometry);
//  coronalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(coronalGeometry);

//  QVERIFY(axialSlicedGeometry);
//  QVERIFY(sagittalSlicedGeometry);
//  QVERIFY(coronalSlicedGeometry);

//  axialFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(0));
//  sagittalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(0));
//  coronalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(0));
//  axialSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(1));
//  sagittalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(1));
//  coronalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(1));

//  QVERIFY(axialFirstPlaneGeometry);
//  QVERIFY(sagittalFirstPlaneGeometry);
//  QVERIFY(coronalFirstPlaneGeometry);
//  QVERIFY(axialSecondPlaneGeometry);
//  QVERIFY(sagittalSecondPlaneGeometry);
//  QVERIFY(coronalSecondPlaneGeometry);

//  axialOrigin = axialGeometry->GetOrigin();
//  sagittalOrigin = sagittalGeometry->GetOrigin();
//  coronalOrigin = coronalGeometry->GetOrigin();
//  axialCentre = axialGeometry->GetCenter();
//  sagittalCentre = sagittalGeometry->GetCenter();
//  coronalCentre = coronalGeometry->GetCenter();
//  axialBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(axialGeometry);
//  sagittalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(sagittalGeometry);
//  coronalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(coronalGeometry);
//  axialFirstPlaneOrigin = axialFirstPlaneGeometry->GetOrigin();
//  sagittalFirstPlaneOrigin = sagittalFirstPlaneGeometry->GetOrigin();
//  coronalFirstPlaneOrigin = coronalFirstPlaneGeometry->GetOrigin();
//  axialFirstPlaneCentre = axialFirstPlaneGeometry->GetCenter();
//  sagittalFirstPlaneCentre = sagittalFirstPlaneGeometry->GetCenter();
//  coronalFirstPlaneCentre = coronalFirstPlaneGeometry->GetCenter();
//  axialSecondPlaneOrigin = axialSecondPlaneGeometry->GetOrigin();
//  sagittalSecondPlaneOrigin = sagittalSecondPlaneGeometry->GetOrigin();
//  coronalSecondPlaneOrigin = coronalSecondPlaneGeometry->GetOrigin();
//  axialSecondPlaneCentre = axialSecondPlaneGeometry->GetCenter();
//  sagittalSecondPlaneCentre = sagittalSecondPlaneGeometry->GetCenter();
//  coronalSecondPlaneCentre = coronalSecondPlaneGeometry->GetCenter();

////  MITK_INFO << "axial origin: " << axialOrigin;
////  MITK_INFO << "sagittal origin: " << sagittalOrigin;
////  MITK_INFO << "coronal origin: " << coronalOrigin;
////  MITK_INFO << "axial centre: " << axialCentre;
////  MITK_INFO << "sagittal centre: " << sagittalCentre;
////  MITK_INFO << "coronal centre: " << coronalCentre;
////  MITK_INFO << "axial bottom left back corner: " << axialBottomLeftBackCorner;
////  MITK_INFO << "sagittal bottom left back corner: " << sagittalBottomLeftBackCorner;
////  MITK_INFO << "coronal bottom left back corner: " << coronalBottomLeftBackCorner;
////  MITK_INFO << "expected axial first plane origin: " << expectedAxialFirstPlaneOrigin;
////  MITK_INFO << "expected sagittal first plane origin: " << expectedSagittalFirstPlaneOrigin;
////  MITK_INFO << "expected coronal first plane origin: " << expectedCoronalFirstPlaneOrigin;
////  MITK_INFO << "expected axial first plane centre: " << expectedAxialFirstPlaneCentre;
////  MITK_INFO << "expected sagittal first plane centre: " << expectedSagittalFirstPlaneCentre;
////  MITK_INFO << "expected coronal first plane centre: " << expectedCoronalFirstPlaneCentre;
////  MITK_INFO << "expected axial second plane origin: " << expectedAxialSecondPlaneOrigin;
////  MITK_INFO << "expected sagittal second plane origin: " << expectedSagittalSecondPlaneOrigin;
////  MITK_INFO << "expected coronal second plane origin: " << expectedCoronalSecondPlaneOrigin;
////  MITK_INFO << "expected axial second plane centre: " << expectedAxialSecondPlaneCentre;
////  MITK_INFO << "expected sagittal second plane centre: " << expectedSagittalSecondPlaneCentre;
////  MITK_INFO << "expected coronal second plane centre: " << expectedCoronalSecondPlaneCentre;

//  QVERIFY(Self::Equals(axialOrigin, expectedAxialOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalOrigin, expectedSagittalOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalOrigin, expectedCoronalOrigin, 0.001));
//  QVERIFY(Self::Equals(axialCentre, expectedAxialCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalCentre, expectedSagittalCentre, 0.001));
//  QVERIFY(Self::Equals(coronalCentre, expectedCoronalCentre, 0.001));
//  QVERIFY(Self::Equals(axialBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(sagittalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(coronalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneOrigin, expectedAxialFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneOrigin, expectedSagittalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneOrigin, expectedCoronalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneCentre, expectedAxialFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneCentre, expectedSagittalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneCentre, expectedCoronalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneOrigin, expectedAxialSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneOrigin, expectedSagittalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneOrigin, expectedCoronalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneCentre, expectedAxialSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneCentre, expectedSagittalSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneCentre, expectedCoronalSecondPlaneCentre, 0.001));

//  /// -------------------------------------------------------------------------
//  /// Initialising the viewer with the world geometry from a sagittal renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of a sagittal renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(sagittalTimeGeometry);

//  axialGeometry = axialRenderer->GetWorldGeometry();
//  sagittalGeometry = sagittalRenderer->GetWorldGeometry();
//  coronalGeometry = coronalRenderer->GetWorldGeometry();

//  QVERIFY(axialGeometry);
//  QVERIFY(sagittalGeometry);
//  QVERIFY(coronalGeometry);

////  MITK_INFO << "axial geometry: " << axialGeometry;
////  MITK_INFO << "sagittal geometry: " << sagittalGeometry;
////  MITK_INFO << "coronal geometry: " << coronalGeometry;

//  axialSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(axialGeometry);
//  sagittalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(sagittalGeometry);
//  coronalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(coronalGeometry);

//  QVERIFY(axialSlicedGeometry);
//  QVERIFY(sagittalSlicedGeometry);
//  QVERIFY(coronalSlicedGeometry);

//  axialFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(0));
//  sagittalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(0));
//  coronalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(0));
//  axialSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(1));
//  sagittalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(1));
//  coronalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(1));

//  QVERIFY(axialFirstPlaneGeometry);
//  QVERIFY(sagittalFirstPlaneGeometry);
//  QVERIFY(coronalFirstPlaneGeometry);
//  QVERIFY(axialSecondPlaneGeometry);
//  QVERIFY(sagittalSecondPlaneGeometry);
//  QVERIFY(coronalSecondPlaneGeometry);

//  axialOrigin = axialGeometry->GetOrigin();
//  sagittalOrigin = sagittalGeometry->GetOrigin();
//  coronalOrigin = coronalGeometry->GetOrigin();
//  axialCentre = axialGeometry->GetCenter();
//  sagittalCentre = sagittalGeometry->GetCenter();
//  coronalCentre = coronalGeometry->GetCenter();
//  axialBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(axialGeometry);
//  sagittalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(sagittalGeometry);
//  coronalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(coronalGeometry);
//  axialFirstPlaneOrigin = axialFirstPlaneGeometry->GetOrigin();
//  sagittalFirstPlaneOrigin = sagittalFirstPlaneGeometry->GetOrigin();
//  coronalFirstPlaneOrigin = coronalFirstPlaneGeometry->GetOrigin();
//  axialFirstPlaneCentre = axialFirstPlaneGeometry->GetCenter();
//  sagittalFirstPlaneCentre = sagittalFirstPlaneGeometry->GetCenter();
//  coronalFirstPlaneCentre = coronalFirstPlaneGeometry->GetCenter();
//  axialSecondPlaneOrigin = axialSecondPlaneGeometry->GetOrigin();
//  sagittalSecondPlaneOrigin = sagittalSecondPlaneGeometry->GetOrigin();
//  coronalSecondPlaneOrigin = coronalSecondPlaneGeometry->GetOrigin();
//  axialSecondPlaneCentre = axialSecondPlaneGeometry->GetCenter();
//  sagittalSecondPlaneCentre = sagittalSecondPlaneGeometry->GetCenter();
//  coronalSecondPlaneCentre = coronalSecondPlaneGeometry->GetCenter();

////  MITK_INFO << "axial origin: " << axialOrigin;
////  MITK_INFO << "sagittal origin: " << sagittalOrigin;
////  MITK_INFO << "coronal origin: " << coronalOrigin;
////  MITK_INFO << "axial centre: " << axialCentre;
////  MITK_INFO << "sagittal centre: " << sagittalCentre;
////  MITK_INFO << "coronal centre: " << coronalCentre;
////  MITK_INFO << "axial bottom left back corner: " << axialBottomLeftBackCorner;
////  MITK_INFO << "sagittal bottom left back corner: " << sagittalBottomLeftBackCorner;
////  MITK_INFO << "coronal bottom left back corner: " << coronalBottomLeftBackCorner;
////  MITK_INFO << "expected axial first plane origin: " << expectedAxialFirstPlaneOrigin;
////  MITK_INFO << "expected sagittal first plane origin: " << expectedSagittalFirstPlaneOrigin;
////  MITK_INFO << "expected coronal first plane origin: " << expectedCoronalFirstPlaneOrigin;
////  MITK_INFO << "expected axial first plane centre: " << expectedAxialFirstPlaneCentre;
////  MITK_INFO << "expected sagittal first plane centre: " << expectedSagittalFirstPlaneCentre;
////  MITK_INFO << "expected coronal first plane centre: " << expectedCoronalFirstPlaneCentre;
////  MITK_INFO << "expected axial second plane origin: " << expectedAxialSecondPlaneOrigin;
////  MITK_INFO << "expected sagittal second plane origin: " << expectedSagittalSecondPlaneOrigin;
////  MITK_INFO << "expected coronal second plane origin: " << expectedCoronalSecondPlaneOrigin;
////  MITK_INFO << "expected axial second plane centre: " << expectedAxialSecondPlaneCentre;
////  MITK_INFO << "expected sagittal second plane centre: " << expectedSagittalSecondPlaneCentre;
////  MITK_INFO << "expected coronal second plane centre: " << expectedCoronalSecondPlaneCentre;

//  QVERIFY(Self::Equals(axialOrigin, expectedAxialOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalOrigin, expectedSagittalOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalOrigin, expectedCoronalOrigin, 0.001));
//  QVERIFY(Self::Equals(axialCentre, expectedAxialCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalCentre, expectedSagittalCentre, 0.001));
//  QVERIFY(Self::Equals(coronalCentre, expectedCoronalCentre, 0.001));
//  QVERIFY(Self::Equals(axialBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(sagittalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(coronalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneOrigin, expectedAxialFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneOrigin, expectedSagittalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneOrigin, expectedCoronalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneCentre, expectedAxialFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneCentre, expectedSagittalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneCentre, expectedCoronalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneOrigin, expectedAxialSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneOrigin, expectedSagittalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneOrigin, expectedCoronalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneCentre, expectedAxialSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneCentre, expectedSagittalSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneCentre, expectedCoronalSecondPlaneCentre, 0.001));

//  /// -------------------------------------------------------------------------
//  /// Initialising the viewer with a world geometry from a coronal renderer
//  /// -------------------------------------------------------------------------

////  MITK_INFO << "Viewer initialised with the world geometry of a coronal renderer.";

//  d->StateTester->Clear();

//  d->Viewer->SetTimeGeometry(coronalTimeGeometry);

//  axialGeometry = axialRenderer->GetWorldGeometry();
//  sagittalGeometry = sagittalRenderer->GetWorldGeometry();
//  coronalGeometry = coronalRenderer->GetWorldGeometry();

//  QVERIFY(axialGeometry);
//  QVERIFY(sagittalGeometry);
//  QVERIFY(coronalGeometry);

////  MITK_INFO << "axial geometry: " << axialGeometry;
////  MITK_INFO << "sagittal geometry: " << sagittalGeometry;
////  MITK_INFO << "coronal geometry: " << coronalGeometry;

//  axialSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(axialGeometry);
//  sagittalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(sagittalGeometry);
//  coronalSlicedGeometry = dynamic_cast<const mitk::SlicedGeometry3D*>(coronalGeometry);

//  QVERIFY(axialSlicedGeometry);
//  QVERIFY(sagittalSlicedGeometry);
//  QVERIFY(coronalSlicedGeometry);

//  axialFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(0));
//  sagittalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(0));
//  coronalFirstPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(0));
//  axialSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(axialSlicedGeometry->GetGeometry2D(1));
//  sagittalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(sagittalSlicedGeometry->GetGeometry2D(1));
//  coronalSecondPlaneGeometry = dynamic_cast<const mitk::PlaneGeometry*>(coronalSlicedGeometry->GetGeometry2D(1));

//  QVERIFY(axialFirstPlaneGeometry);
//  QVERIFY(sagittalFirstPlaneGeometry);
//  QVERIFY(coronalFirstPlaneGeometry);
//  QVERIFY(axialSecondPlaneGeometry);
//  QVERIFY(sagittalSecondPlaneGeometry);
//  QVERIFY(coronalSecondPlaneGeometry);

//  axialOrigin = axialGeometry->GetOrigin();
//  sagittalOrigin = sagittalGeometry->GetOrigin();
//  coronalOrigin = coronalGeometry->GetOrigin();
//  axialCentre = axialGeometry->GetCenter();
//  sagittalCentre = sagittalGeometry->GetCenter();
//  coronalCentre = coronalGeometry->GetCenter();
//  axialBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(axialGeometry);
//  sagittalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(sagittalGeometry);
//  coronalBottomLeftBackCorner = Self::GetWorldBottomLeftBackCorner(coronalGeometry);
//  axialFirstPlaneOrigin = axialFirstPlaneGeometry->GetOrigin();
//  sagittalFirstPlaneOrigin = sagittalFirstPlaneGeometry->GetOrigin();
//  coronalFirstPlaneOrigin = coronalFirstPlaneGeometry->GetOrigin();
//  axialFirstPlaneCentre = axialFirstPlaneGeometry->GetCenter();
//  sagittalFirstPlaneCentre = sagittalFirstPlaneGeometry->GetCenter();
//  coronalFirstPlaneCentre = coronalFirstPlaneGeometry->GetCenter();
//  axialSecondPlaneOrigin = axialSecondPlaneGeometry->GetOrigin();
//  sagittalSecondPlaneOrigin = sagittalSecondPlaneGeometry->GetOrigin();
//  coronalSecondPlaneOrigin = coronalSecondPlaneGeometry->GetOrigin();
//  axialSecondPlaneCentre = axialSecondPlaneGeometry->GetCenter();
//  sagittalSecondPlaneCentre = sagittalSecondPlaneGeometry->GetCenter();
//  coronalSecondPlaneCentre = coronalSecondPlaneGeometry->GetCenter();

////  MITK_INFO << "axial origin: " << axialOrigin;
////  MITK_INFO << "sagittal origin: " << sagittalOrigin;
////  MITK_INFO << "coronal origin: " << coronalOrigin;
////  MITK_INFO << "axial centre: " << axialCentre;
////  MITK_INFO << "sagittal centre: " << sagittalCentre;
////  MITK_INFO << "coronal centre: " << coronalCentre;
////  MITK_INFO << "axial bottom left back corner: " << axialBottomLeftBackCorner;
////  MITK_INFO << "sagittal bottom left back corner: " << sagittalBottomLeftBackCorner;
////  MITK_INFO << "coronal bottom left back corner: " << coronalBottomLeftBackCorner;
////  MITK_INFO << "expected axial first plane origin: " << expectedAxialFirstPlaneOrigin;
////  MITK_INFO << "expected sagittal first plane origin: " << expectedSagittalFirstPlaneOrigin;
////  MITK_INFO << "expected coronal first plane origin: " << expectedCoronalFirstPlaneOrigin;
////  MITK_INFO << "expected axial first plane centre: " << expectedAxialFirstPlaneCentre;
////  MITK_INFO << "expected sagittal first plane centre: " << expectedSagittalFirstPlaneCentre;
////  MITK_INFO << "expected coronal first plane centre: " << expectedCoronalFirstPlaneCentre;
////  MITK_INFO << "expected axial second plane origin: " << expectedAxialSecondPlaneOrigin;
////  MITK_INFO << "expected sagittal second plane origin: " << expectedSagittalSecondPlaneOrigin;
////  MITK_INFO << "expected coronal second plane origin: " << expectedCoronalSecondPlaneOrigin;
////  MITK_INFO << "expected axial second plane centre: " << expectedAxialSecondPlaneCentre;
////  MITK_INFO << "expected sagittal second plane centre: " << expectedSagittalSecondPlaneCentre;
////  MITK_INFO << "expected coronal second plane centre: " << expectedCoronalSecondPlaneCentre;

//  QVERIFY(Self::Equals(axialOrigin, expectedAxialOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalOrigin, expectedSagittalOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalOrigin, expectedCoronalOrigin, 0.001));
//  QVERIFY(Self::Equals(axialCentre, expectedAxialCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalCentre, expectedSagittalCentre, 0.001));
//  QVERIFY(Self::Equals(coronalCentre, expectedCoronalCentre, 0.001));
//  QVERIFY(Self::Equals(axialBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(sagittalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(coronalBottomLeftBackCorner, worldBottomLeftBackCorner, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneOrigin, expectedAxialFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneOrigin, expectedSagittalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneOrigin, expectedCoronalFirstPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialFirstPlaneCentre, expectedAxialFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalFirstPlaneCentre, expectedSagittalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalFirstPlaneCentre, expectedCoronalFirstPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneOrigin, expectedAxialSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneOrigin, expectedSagittalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneOrigin, expectedCoronalSecondPlaneOrigin, 0.001));
//  QVERIFY(Self::Equals(axialSecondPlaneCentre, expectedAxialSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(sagittalSecondPlaneCentre, expectedSagittalSecondPlaneCentre, 0.001));
//  QVERIFY(Self::Equals(coronalSecondPlaneCentre, expectedCoronalSecondPlaneCentre, 0.001));
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
    if (static_cast<int>(d->WorldExtents[i]) % 2 == 0)
    {
      /// If the number of slices is an even number then the selected position
      /// must be exactly at the centre position.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(std::abs(centre[i] - selectedPosition[i]) < 0.001);
    }
    else
    {
      /// If the number of slices is an odd number then the selected position
      /// must be exactly half voxel far from the centre downwards, leftwards or
      /// backwards, respectively.
      /// Tolerance is 0.001 millimetre because of float precision.
      QVERIFY(std::abs(centre[i] - selectedPosition[i] - 0.5 * d->WorldSpacings[i]) < 0.001);
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
  expectedSagittalSlice += d->WorldUpDirections[SagittalAxis] * 20;
  std::vector<mitk::Vector2D> cursorPositions = expectedState->GetCursorPositions();
  std::vector<double> scaleFactors = expectedState->GetScaleFactors();
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 20 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
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
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * 20;
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  /// Note: The position change is orthogonal to the render window plane. The cursor position does not change.
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  selectedPosition[AxialAxis] += 20 * d->WorldSpacings[AxialAxis];
  expectedState->SetSelectedPosition(selectedPosition);
  expectedAxialSlice += d->WorldUpDirections[AxialAxis] * 20;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 20 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
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
  expectedSagittalSlice -= d->WorldUpDirections[SagittalAxis] * 30;
  expectedCoronalSlice -= d->WorldUpDirections[CoronalAxis] * 30;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 30 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
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
  expectedSagittalSlice -= d->WorldUpDirections[SagittalAxis] * 40;
  expectedAxialSlice -= d->WorldUpDirections[AxialAxis] * 40;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] -= 40 * d->WorldSpacings[SagittalAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->width();
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 40 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
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
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * 50;
  expectedAxialSlice += d->WorldUpDirections[AxialAxis] * 50;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] += 50 * d->WorldSpacings[AxialAxis] / scaleFactors[MIDAS_ORIENTATION_CORONAL] / d->CoronalWindow->height();
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedPosition(selectedPosition);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QVERIFY(Self::Equals(d->Viewer->GetSelectedPosition(), selectedPosition, 0.001));
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
//void niftkSingleViewerWidgetTestClass::testGetSelectedSlice()
//{
//  Q_D(niftkSingleViewerWidgetTestClass);

//  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

//  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

//  selectedPosition[CoronalAxis] += 20 * d->WorldSpacings[CoronalAxis];
//  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * 20;

//  d->Viewer->SetSelectedPosition(selectedPosition);
//  d->StateTester->Clear();

//  int coronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//  QCOMPARE(coronalSlice, expectedCoronalSlice);

//  selectedPosition[AxialAxis] += 30 * d->WorldSpacings[AxialAxis];
//  expectedAxialSlice += d->WorldUpDirections[AxialAxis] * 30;

//  d->Viewer->SetSelectedPosition(selectedPosition);
//  d->StateTester->Clear();

//  int axialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  QCOMPARE(axialSlice, expectedAxialSlice);

//  selectedPosition[SagittalAxis] += 40 * d->WorldSpacings[SagittalAxis];
//  expectedSagittalSlice += d->WorldUpDirections[SagittalAxis] * 40;

//  d->Viewer->SetSelectedPosition(selectedPosition);
//  d->StateTester->Clear();

//  int sagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//  QCOMPARE(sagittalSlice, expectedSagittalSlice);
//}

void niftkSingleViewerWidgetTestClass::testGetSelectedSlice()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//  unsigned expectedAxialSliceInSnc = d->Viewer->GetAxialWindow()->GetSliceNavigationController()->GetSlice()->GetPos();
//  unsigned expectedSagittalSliceInSnc = d->Viewer->GetSagittalWindow()->GetSliceNavigationController()->GetSlice()->GetPos();
//  unsigned expectedCoronalSliceInSnc = d->Viewer->GetCoronalWindow()->GetSliceNavigationController()->GetSlice()->GetPos();
  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();

  selectedPosition[CoronalAxis] += 20 * d->WorldSpacings[CoronalAxis];
//  expectedCoronalSliceInSnc += 20;
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * 20;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int coronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);
//  unsigned coronalSliceInSnc = d->CoronalSnc->GetSlice()->GetPos();
  QCOMPARE(coronalSlice, expectedCoronalSlice);
//  QCOMPARE(coronalSliceInSnc, expectedCoronalSliceInSnc);

  selectedPosition[AxialAxis] += 30 * d->WorldSpacings[AxialAxis];
//  expectedAxialSliceInSnc += 30;
  expectedAxialSlice += d->WorldUpDirections[AxialAxis] * 30;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int axialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  unsigned axialSliceInSnc = d->AxialSnc->GetSlice()->GetPos();
  QCOMPARE(axialSlice, expectedAxialSlice);
//  QCOMPARE(axialSliceInSnc, expectedAxialSliceInSnc);

  selectedPosition[SagittalAxis] += 40 * d->WorldSpacings[SagittalAxis];
//  expectedSagittalSliceInSnc = expectedSagittalSlice + 40;
  expectedSagittalSlice += d->WorldUpDirections[SagittalAxis] * 40;

  d->Viewer->SetSelectedPosition(selectedPosition);
  d->StateTester->Clear();

  int sagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//  unsigned sagittalSliceInSnc = d->SagittalSnc->GetSlice()->GetPos();
  QCOMPARE(sagittalSlice, expectedSagittalSlice);
//  QCOMPARE(sagittalSliceInSnc, expectedSagittalSliceInSnc);
}

// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetSelectedSlice()
{
  Q_D(niftkSingleViewerWidgetTestClass);

//  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

//  int axialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  int sagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
  int coronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

//  unsigned expectedCoronalSncPos = d->CoronalSnc->GetSlice()->GetSteps() - 1 - coronalSlice;

  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();

  int delta;

  delta = +20;
  coronalSlice += d->WorldUpDirections[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += delta * d->WorldSpacings[CoronalAxis];
//  expectedCoronalSncPos += delta;

  d->Viewer->SetSelectedSlice(MIDAS_ORIENTATION_CORONAL, coronalSlice);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), coronalSlice);
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSncPos);

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

//  MITK_INFO << "viewer window layout: " << d->Viewer->GetWindowLayout();
//  MITK_INFO << "viewer cursor positions: " << d->Viewer->GetCursorPositions()[0] << " " << d->Viewer->GetCursorPositions()[1] << " " << d->Viewer->GetCursorPositions()[2];
  QVERIFY(ViewerState::New(d->Viewer)->EqualsWithTolerance(d->Viewer->GetCursorPositions(), centrePositions, 0.01));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QVERIFY(d->StateTester->GetQtSignals().empty());
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetCursorPositions()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  /// Note:
  /// The GetCursorPositions/SetCursorPositions functions return/accept
  /// a vector of size 3, but they consider only the elements that
  /// correspond to the actually visible render windows.
  /// Accordingly, the CursorPositionChanged and ScaleFactorChanged
  /// signals are emitted only for the visible windows.

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  d->StateTester->Clear();

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

  std::vector<mitk::Vector2D> cursorPositions = d->Viewer->GetCursorPositions();
  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.41;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.61;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), cursorPositions[MIDAS_ORIENTATION_AXIAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  d->StateTester->Clear();

  /// Note:
  /// Changing the window layout may change the orientation, selected render window,
  /// scale factors, cursor and scale factor binding, but we do not want to test
  /// those changes here. Therefore, we retrieve the current state again, and change
  /// just the cursor positions.

  expectedState = ViewerState::New(d->Viewer);

  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.52;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.72;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(Self::Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), cursorPositions[MIDAS_ORIENTATION_SAGITTAL]));
  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);

  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.33;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.23;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.44;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.74;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.64;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.84;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);

  cursorPositions[MIDAS_ORIENTATION_AXIAL][0] = 0.25;
  cursorPositions[MIDAS_ORIENTATION_AXIAL][1] = 0.35;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.75;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.95;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);

  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] = 0.16;
  cursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] = 0.56;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][0] = 0.46;
  cursorPositions[MIDAS_ORIENTATION_CORONAL][1] = 0.86;
  expectedState->SetCursorPositions(cursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetCursorPositions(cursorPositions);

  QVERIFY(d->StateTester->GetItkSignals().empty());
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);

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
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));
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
  QCOMPARE(d->Viewer->IsFocused(), true);

  d->StateTester->Clear();

  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);
  expectedState->SetSelectedRenderWindow(d->AxialWindow);
  expectedState->SetOrientation(MIDAS_ORIENTATION_AXIAL);
  d->StateTester->SetExpectedState(expectedState);

  d->Viewer->SetSelectedRenderWindow(d->AxialWindow);

  QVERIFY(!qApp->focusWidget() || d->AxialWindow->hasFocus());
  QCOMPARE(focusManager->GetFocused(), d->AxialWindow->GetRenderer());
  QCOMPARE(d->Viewer->IsFocused(), true);

  QCOMPARE(d->StateTester->GetItkSignals(d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(0));
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSetWindowLayout()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();

  mitk::Point3D centreWorldPosition = d->Image->GetGeometry()->GetCenter();

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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->ScaleFactorChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->ScaleFactorChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges == std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges == std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(d->AxialWindow->isVisible());
  QVERIFY(!d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

  d->StateTester->Clear();

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);

  QVERIFY(!qApp->focusWidget() || d->CoronalWindow->hasFocus());
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);
  QCOMPARE(focusManager->GetFocused(), d->CoronalWindow->GetRenderer());
  QVERIFY(!d->AxialWindow->isVisible());
  QVERIFY(d->SagittalWindow->isVisible());
  QVERIFY(d->CoronalWindow->isVisible());
  QVERIFY(!d->_3DWindow->isVisible());
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL), centreDisplayPosition));
  QVERIFY(::EqualsWithTolerance(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), centreDisplayPosition));
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(0));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

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
  QCOMPARE(d->StateTester->GetItkSignals(focusManager, d->FocusEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(2));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);

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
  QCOMPARE(d->StateTester->GetQtSignals(d->WindowLayoutChanged).size(), std::size_t(1));
  cursorPositionChanges = d->StateTester->GetQtSignals(d->CursorPositionChanged).size();
  QVERIFY(cursorPositionChanges <= std::size_t(3));
  scaleFactorChanges = d->StateTester->GetQtSignals(d->ScaleFactorChanged).size();
  QVERIFY(scaleFactorChanges <= std::size_t(3));
  QCOMPARE(d->StateTester->GetQtSignals().size(), 1 + cursorPositionChanges + scaleFactorChanges);
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
  std::vector<mitk::Vector2D> axialCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  this->SetRandomPositions();
  ViewerState::Pointer sagittalState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> sagittalCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  this->SetRandomPositions();
  ViewerState::Pointer coronalState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> coronalCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);
  this->SetRandomPositions();
  ViewerState::Pointer _3DState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> _3DCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);
  this->SetRandomPositions();
  ViewerState::Pointer _3HState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> _3HCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);
  this->SetRandomPositions();
  ViewerState::Pointer _3VState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> _3VCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);
  this->SetRandomPositions();
  ViewerState::Pointer corAxHState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> corAxHCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);
  this->SetRandomPositions();
  ViewerState::Pointer corAxVState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> corAxVCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);
  this->SetRandomPositions();
  ViewerState::Pointer corSagHState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> corSagHCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);
  this->SetRandomPositions();
  ViewerState::Pointer corSagVState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> corSagVCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);
  this->SetRandomPositions();
  ViewerState::Pointer sagAxHState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> sagAxHCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);
  this->SetRandomPositions();
  ViewerState::Pointer sagAxVState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> sagAxVCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  this->SetRandomPositions();
  ViewerState::Pointer orthoState = ViewerState::New(d->Viewer);
  std::vector<mitk::Vector2D> orthoCentres = this->GetCentrePositions();

  ///
  /// Switch back to each layout and check if the states are correctly restored.
  /// Note that the selected position is not restored, and therefore the cursor position
  /// will also be different, but the centre of the image should be at the same position
  /// as last time.
  ///

  mitk::Point3D selectedPosition = d->Viewer->GetSelectedPosition();
  ViewerState::Pointer newState;

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  axialState->SetSelectedPosition(selectedPosition);
  axialState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *axialState);
  QVERIFY(this->Equals(this->GetCentrePositions(), axialCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  sagittalState->SetSelectedPosition(selectedPosition);
  sagittalState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *sagittalState);
  QVERIFY(this->Equals(this->GetCentrePositions(), sagittalCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  coronalState->SetSelectedPosition(selectedPosition);
  coronalState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *coronalState);
  QVERIFY(this->Equals(this->GetCentrePositions(), coronalCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);
  _3DState->SetSelectedPosition(selectedPosition);
  _3DState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *_3DState);
  QVERIFY(this->Equals(this->GetCentrePositions(), _3DCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3H);
  _3HState->SetSelectedPosition(selectedPosition);
  _3HState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *_3HState);
  QVERIFY(this->Equals(this->GetCentrePositions(), _3HCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_3V);
  _3VState->SetSelectedPosition(selectedPosition);
  _3VState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *_3VState);
  QVERIFY(this->Equals(this->GetCentrePositions(), _3VCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_H);
  corAxHState->SetSelectedPosition(selectedPosition);
  corAxHState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *corAxHState);
  QVERIFY(this->Equals(this->GetCentrePositions(), corAxHCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_AX_V);
  corAxVState->SetSelectedPosition(selectedPosition);
  corAxVState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *corAxVState);
  QVERIFY(this->Equals(this->GetCentrePositions(), corAxVCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_H);
  corSagHState->SetSelectedPosition(selectedPosition);
  corSagHState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *corSagHState);
  QVERIFY(this->Equals(this->GetCentrePositions(), corSagHCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_COR_SAG_V);
  corSagVState->SetSelectedPosition(selectedPosition);
  corSagVState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *corSagVState);
  QVERIFY(this->Equals(this->GetCentrePositions(), corSagVCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_H);
  sagAxHState->SetSelectedPosition(selectedPosition);
  sagAxHState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *sagAxHState);
  QVERIFY(this->Equals(this->GetCentrePositions(), sagAxHCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_SAG_AX_V);
  sagAxVState->SetSelectedPosition(selectedPosition);
  sagAxVState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *sagAxVState);
  QVERIFY(this->Equals(this->GetCentrePositions(), sagAxVCentres));

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  orthoState->SetSelectedPosition(selectedPosition);
  orthoState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *orthoState);
  QVERIFY(this->Equals(this->GetCentrePositions(), orthoCentres));

  //// -------------------------------------------------------------------------

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);

  mitk::Vector2D cursorPosition;
  cursorPosition[0] = 0.4;
  cursorPosition[1] = 0.6;
  QPoint pointCursorPosition = this->GetPointAtCursorPosition(d->CoronalWindow, cursorPosition);
//  MITK_INFO << "window size: " << d->CoronalWindow->width() << " " << d->CoronalWindow->height();
//  MITK_INFO << "point position: " << pointCursorPosition.x() << " " << pointCursorPosition.y();
//  MITK_INFO << "actual coronal cursor position 1: " << d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  d->StateTester->Clear();
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, pointCursorPosition);

  /// Note:
  /// If you click in a window, the cursor will be placed at the closest voxel centre.
  /// Here first we find the exaxt world position at the given point, then we find the
  /// world coordinates of the closest voxel centre, finally we translate it back to
  /// display coordinates (normalised by the render window size).
  selectedPosition = this->GetWorldPositionAtDisplayPosition(CoronalAxis, cursorPosition);
  selectedPosition = this->GetVoxelCentrePosition(selectedPosition);
  cursorPosition = this->GetDisplayPositionAtWorldPosition(CoronalAxis, selectedPosition);

  /// TODO Check is disabled for the moment.
//  QVERIFY(this->Equals(d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL), cursorPosition));

  coronalState = ViewerState::New(d->Viewer);
  coronalCentres = this->GetCentrePositions();

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);

  d->StateTester->Clear();
  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  coronalState->SetSelectedPosition(d->Viewer->GetSelectedPosition());
  coronalState->SetCursorPositions(d->Viewer->GetCursorPositions());
  newState = ViewerState::New(d->Viewer);
  QVERIFY(*newState == *coronalState);
  QVERIFY(this->Equals(this->GetCentrePositions(), coronalCentres));
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
  expectedSelectedPosition[SagittalAxis] += this->GetVoxelCentreCoordinate(SagittalAxis, 120.0 * scaleFactor);
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
  expectedSelectedPosition[AxialAxis] -= this->GetVoxelCentreCoordinate(AxialAxis, 60.0 * scaleFactor);
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
  expectedSelectedPosition[SagittalAxis] -= this->GetVoxelCentreCoordinate(SagittalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[AxialAxis] += this->GetVoxelCentreCoordinate(AxialAxis, 40.0 * scaleFactor);
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
  expectedSelectedPosition[SagittalAxis] += this->GetVoxelCentreCoordinate(SagittalAxis, 120.0 * scaleFactor);
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
  expectedSelectedPosition[CoronalAxis] += this->GetVoxelCentreCoordinate(CoronalAxis, 60.0 * scaleFactor);
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
  expectedSelectedPosition[SagittalAxis] -= this->GetVoxelCentreCoordinate(SagittalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[CoronalAxis] -= this->GetVoxelCentreCoordinate(CoronalAxis, 40.0 * scaleFactor);
  expectedSelectedPosition = this->GetVoxelCentrePosition(expectedSelectedPosition);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] -= 30.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] += 40.0 / d->AxialWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  /// TODO Test disabled for the moment.
//  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

//  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(2));
//  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(1));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(2));

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
  expectedSelectedPosition[CoronalAxis] += this->GetVoxelCentreCoordinate(CoronalAxis, 120.0 * scaleFactor);
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
  expectedSelectedPosition[AxialAxis] -= this->GetVoxelCentreCoordinate(AxialAxis, 60.0 * scaleFactor);
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
  expectedSelectedPosition[CoronalAxis] -= this->GetVoxelCentreCoordinate(CoronalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[AxialAxis] += this->GetVoxelCentreCoordinate(AxialAxis, 40.0 * scaleFactor);
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
  d->Viewer->SetCursorPositionBinding(false);
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
  expectedSelectedPosition[SagittalAxis] += this->GetVoxelCentreCoordinate(SagittalAxis, 120.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 120.0 / d->CoronalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] += 120.0 / d->AxialWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= this->GetVoxelCentreCoordinate(AxialAxis, 60.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 60.0 / d->CoronalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] -= 60.0 / d->SagittalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= this->GetVoxelCentreCoordinate(SagittalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[AxialAxis] += this->GetVoxelCentreCoordinate(AxialAxis, 40.0 * scaleFactor);
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
//  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(0));

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
  expectedSelectedPosition[SagittalAxis] += this->GetVoxelCentreCoordinate(SagittalAxis, 120.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][0] += 120.0 / d->AxialWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][0] += 120.0 / d->AxialWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->SagittalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[CoronalAxis] += this->GetVoxelCentreCoordinate(CoronalAxis, 60.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] -= 60.0 / d->AxialWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] += 60.0 / d->SagittalWindow->width();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->AxialWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[SagittalAxis] -= this->GetVoxelCentreCoordinate(SagittalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[CoronalAxis] -= this->GetVoxelCentreCoordinate(CoronalAxis, 40.0 * scaleFactor);
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
//  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
//  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(0));

  d->StateTester->Clear();

  expectedState = ViewerState::New(d->Viewer);
  expectedSelectedPosition = expectedState->GetSelectedPosition();
  expectedCursorPositions = expectedState->GetCursorPositions();

  scaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_SAGITTAL);

  newPoint.rx() += 120;
  expectedSelectedPosition[CoronalAxis] += this->GetVoxelCentreCoordinate(CoronalAxis, 120.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][0] += 120.0 / d->SagittalWindow->width();
  expectedCursorPositions[MIDAS_ORIENTATION_AXIAL][1] -= 120.0 / d->AxialWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.ry() += 60;
  expectedSelectedPosition[AxialAxis] -= this->GetVoxelCentreCoordinate(AxialAxis, 60.0 * scaleFactor);
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  expectedCursorPositions[MIDAS_ORIENTATION_SAGITTAL][1] -= 60.0 / d->SagittalWindow->height();
  expectedCursorPositions[MIDAS_ORIENTATION_CORONAL][1] -= 60.0 / d->CoronalWindow->height();
  expectedState->SetCursorPositions(expectedCursorPositions);
  d->StateTester->SetExpectedState(expectedState);

  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, newPoint);

  QCOMPARE(d->StateTester->GetItkSignals(d->AxialSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(2));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(3));

  d->StateTester->Clear();

  newPoint.rx() -= 30;
  newPoint.ry() -= 40;
  expectedSelectedPosition[CoronalAxis] -= this->GetVoxelCentreCoordinate(CoronalAxis, 30.0 * scaleFactor);
  expectedSelectedPosition[AxialAxis] += this->GetVoxelCentreCoordinate(AxialAxis, 40.0 * scaleFactor);
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
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  d->StateTester->SetExpectedState(expectedState);

  Self::MouseWheel(d->CoronalWindow, Qt::NoButton, Qt::NoModifier, centre, delta);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
  /// Note: The position change is orthogonal to the render window plane. The cursor position does not change.
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  delta = -1;
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  d->StateTester->SetExpectedState(expectedState);

  Self::MouseWheel(d->CoronalWindow, Qt::NoButton, Qt::NoModifier, centre, delta);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->CursorPositionChanged).size(), std::size_t(0));
  /// Note: The position change is orthogonal to the render window plane. The cursor position does not change.
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

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
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  d->StateTester->SetExpectedState(expectedState);

  QTest::keyClick(d->CoronalWindow, Qt::Key_A, Qt::NoModifier);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();

  delta = -1;
  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * delta;
  expectedSelectedPosition[CoronalAxis] += delta * d->WorldSpacings[CoronalAxis];
  expectedState->SetSelectedPosition(expectedSelectedPosition);
  d->StateTester->SetExpectedState(expectedState);

  QTest::keyClick(d->CoronalWindow, Qt::Key_Z, Qt::NoModifier);

  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
  QCOMPARE(d->StateTester->GetItkSignals(d->CoronalSnc, d->GeometrySliceEvent).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetItkSignals().size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals(d->SelectedPositionChanged).size(), std::size_t(1));
  QCOMPARE(d->StateTester->GetQtSignals().size(), std::size_t(1));

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectSliceThroughSliceNavigationController()
{
//  Q_D(niftkSingleViewerWidgetTestClass);

////  ViewerState::Pointer expectedState = ViewerState::New(d->Viewer);

//  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();
//  int expectedAxialSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL);
//  int expectedSagittalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL);
//  int expectedCoronalSlice = d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL);

//  unsigned axialSncPos = d->AxialSnc->GetSlice()->GetPos();
//  unsigned sagittalSncPos = d->SagittalSnc->GetSlice()->GetPos();
//  unsigned coronalSncPos = d->CoronalSnc->GetSlice()->GetPos();

//  int axialSliceDelta;
//  int sagittalSliceDelta;
//  int coronalSliceDelta;

//  axialSliceDelta = +2;
//  axialSncPos += axialSliceDelta;
//  expectedAxialSlice += d->WorldUpDirections[AxialAxis] * axialSliceDelta;
//  expectedSelectedPosition[AxialAxis] += axialSliceDelta * d->WorldSpacings[AxialAxis];

//  d->AxialSnc->GetSlice()->SetPos(axialSncPos);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
//  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

//  d->StateTester->Clear();

//  sagittalSliceDelta = -3;
//  sagittalSncPos += sagittalSliceDelta;
//  expectedSagittalSlice += d->WorldUpDirections[SagittalAxis] * sagittalSliceDelta;
//  expectedSelectedPosition[SagittalAxis] += sagittalSliceDelta * d->WorldSpacings[SagittalAxis];

//  d->SagittalSnc->GetSlice()->SetPos(sagittalSncPos);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
//  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

//  d->StateTester->Clear();

//  coronalSliceDelta = +5;
//  coronalSncPos += coronalSliceDelta;
//  expectedCoronalSlice += d->WorldUpDirections[CoronalAxis] * coronalSliceDelta;
//  expectedSelectedPosition[CoronalAxis] += coronalSliceDelta * d->WorldSpacings[CoronalAxis];

//  d->CoronalSnc->GetSlice()->SetPos(coronalSncPos);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
//  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

//  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectPositionThroughSliceNavigationController()
{
//  Q_D(niftkSingleViewerWidgetTestClass);

//  const mitk::TimeGeometry* timeGeometry = d->Viewer->GetTimeGeometry();
//  mitk::Geometry3D* worldGeometry = timeGeometry->GetGeometryForTimeStep(0);

//  mitk::Point3D expectedSelectedPosition = d->Viewer->GetSelectedPosition();
//  mitk::Point3D randomWorldPosition = this->GetRandomWorldPosition();

//  expectedSelectedPosition[AxialAxis] = randomWorldPosition[AxialAxis];

//  mitk::Index3D expectedSelectedIndex;
//  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

//  int expectedAxialSlice = expectedSelectedIndex[1];
//  int expectedSagittalSlice = expectedSelectedIndex[0];
//  int expectedCoronalSlice = expectedSelectedIndex[2];

//  unsigned expectedAxialSncPos = d->WorldUpDirections[2] > 0 ? expectedAxialSlice : d->AxialSnc->GetSlice()->GetSteps() - 1 - expectedAxialSlice;
//  unsigned expectedSagittalSncPos = d->WorldUpDirections[0] > 0 ? expectedSagittalSlice : d->SagittalSnc->GetSlice()->GetSteps() - 1 - expectedSagittalSlice;
//  unsigned expectedCoronalSncPos = d->WorldUpDirections[1] > 0 ? expectedCoronalSlice: d->CoronalSnc->GetSlice()->GetSteps() - 1 - expectedCoronalSlice;

//  d->AxialSnc->SelectSliceByPoint(expectedSelectedPosition);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_AXIAL), expectedAxialSlice);
//  QCOMPARE(d->AxialSnc->GetSlice()->GetPos(), expectedAxialSncPos);

//  d->StateTester->Clear();

//  expectedSelectedPosition[SagittalAxis] = randomWorldPosition[SagittalAxis];

//  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

//  expectedSagittalSlice = expectedSelectedIndex[0];
//  expectedSagittalSncPos = d->WorldUpDirections[0] > 0 ? expectedSagittalSlice : d->SagittalSnc->GetSlice()->GetSteps() - 1 - expectedSagittalSlice;

//  d->SagittalSnc->SelectSliceByPoint(expectedSelectedPosition);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_SAGITTAL), expectedSagittalSlice);
//  QCOMPARE(d->SagittalSnc->GetSlice()->GetPos(), expectedSagittalSncPos);

//  d->StateTester->Clear();

//  expectedSelectedPosition[CoronalAxis] = randomWorldPosition[CoronalAxis];

//  worldGeometry->WorldToIndex(expectedSelectedPosition, expectedSelectedIndex);

//  expectedCoronalSlice = expectedSelectedIndex[2];
//  expectedCoronalSncPos = d->WorldUpDirections[1] > 0 ? expectedCoronalSlice: d->CoronalSnc->GetSlice()->GetSteps() - 1 - expectedCoronalSlice;

//  d->CoronalSnc->SelectSliceByPoint(expectedSelectedPosition);

//  QCOMPARE(d->Viewer->GetSelectedSlice(MIDAS_ORIENTATION_CORONAL), expectedCoronalSlice);
//  QCOMPARE(d->CoronalSnc->GetSlice()->GetPos(), expectedCoronalSncPos);

//  QVERIFY(this->Equals(d->Viewer->GetSelectedPosition(), expectedSelectedPosition));

//  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testSelectRenderWindowByInteraction()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  QCOMPARE(d->Viewer->IsFocused(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);

  QCOMPARE(d->Viewer->IsFocused(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->CoronalWindow);

  d->StateTester->Clear();

  QPoint centre = d->SagittalWindow->rect().center();
  QTest::mouseClick(d->SagittalWindow, Qt::LeftButton, Qt::NoModifier, centre);

  QCOMPARE(d->Viewer->IsFocused(), true);
  QCOMPARE(d->Viewer->GetSelectedRenderWindow(), d->SagittalWindow);
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testCursorPositionBinding()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  d->StateTester->Clear();
  d->Viewer->SetCursorPositionBinding(true);
  d->Viewer->SetScaleFactorBinding(false);
  d->StateTester->Clear();

  mitk::Point3D worldPosition;
  mitk::Vector2D displayPosition;

  mitk::Vector2D axialCursorPosition;
  mitk::Vector2D sagittalCursorPosition;
  mitk::Vector2D coronalCursorPosition;

  worldPosition = this->GetRandomWorldPosition();
  d->Viewer->SetSelectedPosition(worldPosition);

  axialCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);
  sagittalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
  coronalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QCOMPARE(axialCursorPosition[0], coronalCursorPosition[0]);
  QCOMPARE(sagittalCursorPosition[1], coronalCursorPosition[1]);

  d->StateTester->Clear();

  double axialScaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_AXIAL);
  double sagittalScaleFactor = d->Viewer->GetScaleFactor(MIDAS_ORIENTATION_SAGITTAL);

  d->Viewer->SetScaleFactor(MIDAS_ORIENTATION_AXIAL, axialScaleFactor * 2);
  d->StateTester->Clear();
  d->Viewer->SetScaleFactor(MIDAS_ORIENTATION_SAGITTAL, sagittalScaleFactor * 0.5);
  d->StateTester->Clear();

  axialCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);
  sagittalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
  coronalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QCOMPARE(axialCursorPosition[0], coronalCursorPosition[0]);
  QCOMPARE(sagittalCursorPosition[1], coronalCursorPosition[1]);

  worldPosition = this->GetRandomWorldPosition();
  d->Viewer->SetSelectedPosition(worldPosition);

  axialCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);
  sagittalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
  coronalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QCOMPARE(axialCursorPosition[0], coronalCursorPosition[0]);
  QCOMPARE(sagittalCursorPosition[1], coronalCursorPosition[1]);

  d->StateTester->Clear();

  displayPosition = this->GetRandomDisplayPosition();

  QPoint pointAtDisplayPosition = this->GetPointAtCursorPosition(d->CoronalWindow, displayPosition);

  d->StateTester->Clear();
  QTest::mouseClick(d->CoronalWindow, Qt::LeftButton, Qt::NoModifier, pointAtDisplayPosition);

  axialCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_AXIAL);
  sagittalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_SAGITTAL);
  coronalCursorPosition = d->Viewer->GetCursorPosition(MIDAS_ORIENTATION_CORONAL);

  QCOMPARE(axialCursorPosition[0], coronalCursorPosition[0]);
  QCOMPARE(sagittalCursorPosition[1], coronalCursorPosition[1]);

  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testScaleFactorBinding()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  d->StateTester->Clear();
  d->Viewer->SetScaleFactorBinding(true);
  d->StateTester->Clear();
}


// --------------------------------------------------------------------------
void niftkSingleViewerWidgetTestClass::testCursorPositionAndScaleFactorBinding()
{
  Q_D(niftkSingleViewerWidgetTestClass);

  d->Viewer->SetWindowLayout(WINDOW_LAYOUT_ORTHO);
  d->StateTester->Clear();
  d->Viewer->SetCursorPositionBinding(true);
  d->StateTester->Clear();
  d->Viewer->SetScaleFactorBinding(true);
  d->StateTester->Clear();
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
