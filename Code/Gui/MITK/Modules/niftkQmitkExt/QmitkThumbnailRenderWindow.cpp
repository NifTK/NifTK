/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkThumbnailRenderWindow.h"
#include "QmitkMouseEventEater.h"
#include "QmitkWheelEventEater.h"
#include <QtGui>
#include "mitkGlobalInteraction.h"
#include "mitkFocusManager.h"
#include "mitkDataStorage.h"
#include "mitkDisplayGeometry.h"
#include "itkCommand.h"
#include "vtkPolyData.h"
#include "vtkSmartPointer.h"
#include "vtkCubeSource.h"

QmitkThumbnailRenderWindow::QmitkThumbnailRenderWindow(QWidget *parent)
  : QmitkRenderWindow(parent)
, m_FocusManagerObserverTag(0)
, m_FocusedWindowWorldGeometryTag(0)
, m_FocusedWindowDisplayGeometryTag(0)
, m_FocusedWindowSliceSelectorTag(0)
, m_FocusedWindowTimeStepSelectorTag(0)
, m_DataStorage(NULL)
, m_BoundingBoxNode(NULL)
, m_BoundingBox(NULL)
, m_BaseRenderer(NULL)
, m_TrackedRenderWindow(NULL)
, m_TrackedSliceNavigator(NULL)
, m_MouseEventEater(NULL)
, m_WheelEventEater(NULL)
, m_InDataStorageChanged(false)
{
  m_BoundingBox = mitk::Cuboid::New();
  m_BoundingBoxNode = mitk::DataNode::New();
  m_BoundingBoxNode->SetData( m_BoundingBox );
  m_BoundingBoxNode->SetProperty( "name", mitk::StringProperty::New( "ThumbnailBoundingBox" ) );
  m_BoundingBoxNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
  m_BoundingBoxNode->SetProperty( "helper object", mitk::BoolProperty::New(true) );
  m_BoundingBoxNode->SetBoolProperty("visible", false); // globally turn it off, then we only turn it on in thumbnail (this) window.

  this->setBoundingBoxVisible(false);
  this->setBoundingBoxColor(255, 0, 0);
  this->setBoundingBoxOpacity(1);
  this->setBoundingBoxLineThickness(1);
  this->setBoundingBoxLayer(99);// arbitrary, copied from segmentation functionality

  m_MouseEventEater = new QmitkMouseEventEater();
  m_MouseEventEater->SetIsEating(false);
  this->installEventFilter(m_MouseEventEater);

  m_WheelEventEater = new QmitkWheelEventEater();
  m_WheelEventEater->SetIsEating(true);
  this->installEventFilter(m_WheelEventEater);

  m_BaseRenderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());

  std::vector<mitk::BaseRenderer*> renderers;
  renderers.push_back(m_BaseRenderer);

  m_NodeAddedSetter = mitk::MIDASNodeAddedVisibilitySetter::New();
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetRenderersToUpdate(renderers);
}

QmitkThumbnailRenderWindow::~QmitkThumbnailRenderWindow()
{
  if (m_MouseEventEater != NULL)
  {
    delete m_MouseEventEater;
  }

  if (m_WheelEventEater != NULL)
  {
    delete m_WheelEventEater;
  }
}

void QmitkThumbnailRenderWindow::Activated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkThumbnailRenderWindow::OnFocusChanged );

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.AddListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.AddListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeChangedProxy ) );
  }
}

void QmitkThumbnailRenderWindow::Deactivated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL && m_FocusManagerObserverTag != 0)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeChangedProxy ) );
  }

  this->RemoveObserversFromTrackedObjects();
}

void QmitkThumbnailRenderWindow::RemoveObserversFromTrackedObjects()
{
  // NOTE the following curiosity:
  //
  // Initially my m_TrackedWorldGeometry and m_TrackedDisplayGeometry were regular pointers
  // not smart pointers. However, when exiting the application, it appeared that they were
  // not valid, causing seg faults.  It appeared to me (Matt), that as the focus changed between
  // windows, the slice navigation controller is constant for each render window, but the
  // geometries are created/deleted as needed, as the geometry changed.
  // So, as a work-around, I made these pointers smart pointers, so that if the geometry objects are
  // deleted by their renderers, then this smart pointer in this class will still have a valid reference
  // to the object that it originally promised to listen to.  This will avoid crashes, and the geometry
  // object will go out of scope when it is replaced with a new one, or this object is destroyed.

  if (m_TrackedWorldGeometry.IsNotNull() && m_TrackedWorldGeometry->GetCommand(m_FocusedWindowWorldGeometryTag) != NULL)
  {
    m_TrackedWorldGeometry->RemoveObserver(m_FocusedWindowWorldGeometryTag);
  }

  if (m_TrackedDisplayGeometry.IsNotNull() && m_TrackedDisplayGeometry->GetCommand(m_FocusedWindowDisplayGeometryTag) != NULL)
  {
    m_TrackedDisplayGeometry->RemoveObserver(m_FocusedWindowDisplayGeometryTag);
  }

  if (m_TrackedSliceNavigator.IsNotNull() && m_TrackedSliceNavigator->GetCommand(m_FocusedWindowSliceSelectorTag) != NULL)
  {
    m_TrackedSliceNavigator->RemoveObserver(m_FocusedWindowSliceSelectorTag);
  }

  if (m_TrackedSliceNavigator.IsNotNull() && m_TrackedSliceNavigator->GetCommand(m_FocusedWindowTimeStepSelectorTag) != NULL)
  {
    m_TrackedSliceNavigator->RemoveObserver(m_FocusedWindowTimeStepSelectorTag);
  }
}

void QmitkThumbnailRenderWindow::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  // Don't allow anyone to pass in a null dataStorage.
  assert(dataStorage);
  m_DataStorage = dataStorage;

  mitk::BaseRenderer::Pointer thumbnailWindowRenderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());
  thumbnailWindowRenderer->SetDataStorage(dataStorage);

  m_NodeAddedSetter->SetDataStorage(dataStorage);
  m_VisibilityTracker->SetDataStorage(dataStorage);
}

mitk::DataStorage::Pointer QmitkThumbnailRenderWindow::GetDataStorage()
{
  // This MUST be set before you actually use this widget.
  assert(m_DataStorage);
  return m_DataStorage;
}

mitk::Point3D QmitkThumbnailRenderWindow::Get3DPoint(int x, int y)
{
  mitk::Point3D pointInMillimetres3D;

  if (m_TrackedDisplayGeometry.IsNotNull())
  {
    mitk::Point2D pointInVoxels2D;
    mitk::Point2D pointInMillimetres2D;

    pointInVoxels2D[0] = x;
    pointInVoxels2D[1] = y;

    m_TrackedDisplayGeometry->DisplayToWorld(pointInVoxels2D, pointInMillimetres2D);
    m_TrackedDisplayGeometry->Map(pointInMillimetres2D, pointInMillimetres3D);
  }

  return pointInMillimetres3D;
}

void QmitkThumbnailRenderWindow::OnDisplayGeometryChanged()
{
  this->UpdateBoundingBox();
}

void QmitkThumbnailRenderWindow::UpdateBoundingBox()
{
  if (m_TrackedDisplayGeometry.IsNotNull())
  {
    // Get min and max extent of the focused render window's display geometry.
    mitk::Point3D points[4];
    points[0] = this->Get3DPoint(0, 0);
    points[1] = this->Get3DPoint(m_TrackedDisplayGeometry->GetDisplayWidth()-1, 0);
    points[2] = this->Get3DPoint(0, m_TrackedDisplayGeometry->GetDisplayHeight()-1);
    points[3] = this->Get3DPoint(m_TrackedDisplayGeometry->GetDisplayWidth()-1, m_TrackedDisplayGeometry->GetDisplayHeight()-1);

    mitk::Point3D min = points[0];
    mitk::Point3D max = points[0];

    for (int i = 1; i < 4; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        if (points[i][j] < min[j])
        {
          min[j] = points[i][j];
        }
        if (points[i][j] > max[j])
        {
          max[j] = points[i][j];
        }
      }
    }

    // Work out axis that changes the least (axis towards plane).
    mitk::Point3D diff;
    for (int i = 0; i < 3; i++)
    {
      diff[i] = max[i] - min[i];
    }

    double bestChange = fabs(diff[0]);
    int bestIndex = 0;
    for (int i = 1; i< 3; i++)
    {
      if (fabs(diff[i]) < bestChange)
      {
        bestIndex = i;
        bestChange = fabs(diff[i]);
      }
    }

    // Add a bit of jitter
    min[bestIndex] -= 1;
    max[bestIndex] += 1;

    // Create a cube.
    vtkCubeSource* cube = vtkCubeSource::New();
    cube->SetBounds(min[0], max[0], min[1], max[1], min[2], max[2]);
    cube->Update();

    // Update bounding box.
    m_BoundingBox->SetVtkPolyData(cube->GetOutput());
    m_BoundingBox->Modified();
    m_BoundingBoxNode->Modified();

    // Tidy up
    cube->Delete();

    // We shouldn't need this every time, but without it you don't seem to get all the updates.
    this->setBoundingBoxVisible(true);

    // Request a single update at the end of the method.
    mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
  }
}

void QmitkThumbnailRenderWindow::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged && node != m_BoundingBoxNode)
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(node);
    m_InDataStorageChanged = false;
  }
}

void QmitkThumbnailRenderWindow::NodeAdded( const mitk::DataNode* node)
{
  this->UpdateSliceAndTimeStep();
  this->OnDisplayGeometryChanged();
  this->UpdateBoundingBox();

  mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
}

void QmitkThumbnailRenderWindow::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged && node != m_BoundingBoxNode)
  {
    m_InDataStorageChanged = true;
    this->NodeChanged(node);
    m_InDataStorageChanged = false;
  }
}

void QmitkThumbnailRenderWindow::NodeChanged( const mitk::DataNode* node)
{
/*
  this->UpdateSliceAndTimeStep();
  this->OnDisplayGeometryChanged();
  this->UpdateBoundingBox();
*/
  mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
}

void QmitkThumbnailRenderWindow::OnSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->UpdateSliceAndTimeStep();
}

void QmitkThumbnailRenderWindow::OnTimeStepChanged(const itk::EventObject & geometrySliceEvent)
{
  this->UpdateSliceAndTimeStep();
}

void QmitkThumbnailRenderWindow::UpdateSliceAndTimeStep()
{
  if (m_TrackedRenderWindow != NULL)
  {
    mitk::BaseRenderer::Pointer mitkRendererForThumbnail = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());
    mitk::BaseRenderer::Pointer mitkRendererForTrackedWidget = mitk::BaseRenderer::GetInstance(m_TrackedRenderWindow);

    if (mitkRendererForThumbnail.IsNotNull() && mitkRendererForTrackedWidget.IsNotNull())
    {
      if (mitkRendererForTrackedWidget->GetTimeStep() != mitkRendererForThumbnail->GetTimeStep())
      {
        mitkRendererForThumbnail->SetTimeStep(mitkRendererForTrackedWidget->GetTimeStep());
      }

      if (mitkRendererForTrackedWidget->GetSlice() != mitkRendererForThumbnail->GetSlice())
      {
        mitkRendererForThumbnail->SetSlice(mitkRendererForTrackedWidget->GetSlice());
      }

      this->UpdateWorldGeometry(true);
    }
  }
}

void QmitkThumbnailRenderWindow::OnWorldGeometryChanged()
{
  this->UpdateWorldGeometry(true);
}

void QmitkThumbnailRenderWindow::UpdateWorldGeometry(bool fitToDisplay)
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer::ConstPointer focusedWindowRenderer = focusManager->GetFocused();
    if (focusedWindowRenderer.IsNotNull())
    {
      // Retrieve the correct slice of world geometry to display.
      //mitk::Geometry2D::ConstPointer focusedWindowWorldGeometry = focusedWindowRenderer->GetCurrentWorldGeometry2D();

      // World geometry of thumbnail must be same (or larger) as world geometry of the focused window.
      //thumbnailWindowRenderer->SetWorldGeometry(const_cast<mitk::Geometry2D*>(focusedWindowWorldGeometry.GetPointer()));
      m_BaseRenderer->SetWorldGeometry(const_cast<mitk::Geometry3D*>(focusedWindowRenderer->GetWorldGeometry()));

      // Display geometry of widget must encompass whole of world geometry
      if (fitToDisplay)
      {
        m_BaseRenderer->GetDisplayGeometry()->Fit();
      }

      // Request a single update at the end of the method.
      mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
    }
  }
}

void QmitkThumbnailRenderWindow::OnFocusChanged()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNotNull())
  {
    mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
    if (focusManager != NULL)
    {
      mitk::BaseRenderer::ConstPointer focusedWindowRenderer = focusManager->GetFocused();
      if (focusedWindowRenderer.IsNotNull())
      {
        vtkRenderWindow* focusedWindowRenderWindow = focusedWindowRenderer->GetRenderWindow();

        if (focusedWindowRenderer->GetMapperID() == mitk::BaseRenderer::Standard3D)
        {
          // Remove any existing geometry observers
          this->RemoveObserversFromTrackedObjects();
        }
        else if (focusedWindowRenderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
        {
          if (!(focusedWindowRenderWindow == this->GetVtkRenderWindow()))
          {
            // Remove any existing geometry observers
            this->RemoveObserversFromTrackedObjects();

            // Store pointers to the display and world geometry, and render window
            m_TrackedWorldGeometry = const_cast<mitk::Geometry3D*>(focusedWindowRenderer->GetWorldGeometry());
            m_TrackedDisplayGeometry = const_cast<mitk::DisplayGeometry*>(focusedWindowRenderer->GetDisplayGeometry());

            if (m_TrackedWorldGeometry.IsNotNull()
                && m_TrackedDisplayGeometry.IsNotNull())
            {

              m_TrackedSliceNavigator = (const_cast<mitk::BaseRenderer*>(focusedWindowRenderer.GetPointer()))->GetSliceNavigationController();
              m_TrackedRenderWindow = focusedWindowRenderWindow;

              // Add Observers to track when these geometries change
              itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::Pointer onWorldGeometryChangedCommand =
                itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::New();
              onWorldGeometryChangedCommand->SetCallbackFunction( this, &QmitkThumbnailRenderWindow::OnWorldGeometryChanged );
              m_FocusedWindowWorldGeometryTag = m_TrackedWorldGeometry->AddObserver(itk::ModifiedEvent(), onWorldGeometryChangedCommand);

              itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::Pointer onDisplayGeometryChangedCommand =
                itk::SimpleMemberCommand<QmitkThumbnailRenderWindow>::New();
              onDisplayGeometryChangedCommand->SetCallbackFunction( this, &QmitkThumbnailRenderWindow::OnDisplayGeometryChanged );
              m_FocusedWindowDisplayGeometryTag = m_TrackedDisplayGeometry->AddObserver(itk::ModifiedEvent(), onDisplayGeometryChangedCommand);

              itk::ReceptorMemberCommand<QmitkThumbnailRenderWindow>::Pointer onSliceChangedCommand =
                itk::ReceptorMemberCommand<QmitkThumbnailRenderWindow>::New();
              onSliceChangedCommand->SetCallbackFunction( this, &QmitkThumbnailRenderWindow::OnSliceChanged );
              m_FocusedWindowSliceSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSliceChangedCommand);

              itk::ReceptorMemberCommand<QmitkThumbnailRenderWindow>::Pointer onTimeChangedCommand =
                itk::ReceptorMemberCommand<QmitkThumbnailRenderWindow>::New();
              onTimeChangedCommand->SetCallbackFunction( this, &QmitkThumbnailRenderWindow::OnTimeStepChanged );
              m_FocusedWindowTimeStepSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometryTimeEvent(NULL, 0), onTimeChangedCommand);

              // Check we added bounding box to data storage.
              // I'm doing this in this method so that when the initial first
              // window starts (i.e. before any data is loaded),
              // the bounding box will not be included, and not visible.
              if (!dataStorage->Exists(m_BoundingBoxNode))
              {
                this->GetDataStorage()->Add(m_BoundingBoxNode);
              }

              // This sets the tracked render window to the same slice and time step and re-computes the world geometry.
              this->UpdateSliceAndTimeStep();

              // This computes the bounding box.
              this->OnDisplayGeometryChanged();

              // Setup the visibility tracker.
              std::vector<mitk::BaseRenderer*> windowsToTrack;
              windowsToTrack.push_back(const_cast<mitk::BaseRenderer*>(focusedWindowRenderer.GetPointer()));
              this->m_VisibilityTracker->SetRenderersToTrack(windowsToTrack);
              this->m_VisibilityTracker->OnPropertyChanged(); // force update

              // Finally to get the box to update
              this->UpdateBoundingBox();
            }
          } // end if focused window is not thumbnail window.
        } // end if 2D mapper
      } // end if we do actually have a focussed window
    } // end if we have a focus manager
  } // end if we have a data storage
}

QColor QmitkThumbnailRenderWindow::boundingBoxColor() const
{
  float colour[3];
  m_BoundingBoxNode->GetColor(colour);

  QColor qtColour(colour[0] * 255, colour[1] * 255, colour[2] * 255);
  return qtColour;
}

void QmitkThumbnailRenderWindow::setBoundingBoxColor(QColor &colour)
{
  m_BoundingBoxNode->SetColor(colour.redF(), colour.greenF(), colour.blueF());
}

void QmitkThumbnailRenderWindow::setBoundingBoxColor(float r, float g, float b)
{
  m_BoundingBoxNode->SetColor(r, g, b);
}

int QmitkThumbnailRenderWindow::boundingBoxLineThickness() const
{
  int thickness = 0;
  m_BoundingBoxNode->GetIntProperty("line width", thickness);
  return thickness;
}

void QmitkThumbnailRenderWindow::setBoundingBoxLineThickness(int thickness)
{
  m_BoundingBoxNode->SetIntProperty("line width", thickness);
}

float QmitkThumbnailRenderWindow::boundingBoxOpacity() const
{
  float opacity = 0;
  m_BoundingBoxNode->GetFloatProperty("opacity", opacity);
  return opacity;
}

void QmitkThumbnailRenderWindow::setBoundingBoxOpacity(float opacity)
{
  m_BoundingBoxNode->SetOpacity(opacity);
}

bool QmitkThumbnailRenderWindow::boundingBoxVisible() const
{
  bool visible = false;
  m_BoundingBoxNode->GetBoolProperty("visible", visible, m_BaseRenderer);
  return visible;
}

void QmitkThumbnailRenderWindow::setBoundingBoxVisible(bool visible)
{
  m_BoundingBoxNode->SetBoolProperty("visible", visible, m_BaseRenderer);
}

int QmitkThumbnailRenderWindow::boundingBoxLayer() const
{
  bool layer = 0;
  m_BoundingBoxNode->GetBoolProperty("layer", layer);
  return layer;
}

void QmitkThumbnailRenderWindow::setBoundingBoxLayer(int layer)
{
  m_BoundingBoxNode->SetIntProperty("layer", layer);
}

void QmitkThumbnailRenderWindow::setRespondToMouseEvents(bool on)
{
  m_MouseEventEater->SetIsEating(!on);
}

bool QmitkThumbnailRenderWindow::respondToMouseEvents() const
{
  return !m_MouseEventEater->GetIsEating();
}

void QmitkThumbnailRenderWindow::setRespondToWheelEvents(bool on)
{
  m_WheelEventEater->SetIsEating(!on);
}

bool QmitkThumbnailRenderWindow::respondToWheelEvents() const
{
  return !m_WheelEventEater->GetIsEating();
}
