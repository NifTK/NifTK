/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkThumbnailRenderWindow.h"
#include <QtGui>

#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkDataStorage.h>
#include <mitkDisplayGeometry.h>
#include <itkCommand.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkCubeSource.h>

#include <QmitkMouseEventEater.h>
#include <QmitkWheelEventEater.h>
#include <mitkDataStorageUtils.h>

#include <mitkDataNodeStringPropertyFilter.h>

//-----------------------------------------------------------------------------
QmitkThumbnailRenderWindow::QmitkThumbnailRenderWindow(QWidget *parent)
  : QmitkRenderWindow(parent)
, m_FocusManagerObserverTag(-1)
, m_FocusedWindowWorldGeometryTag(-1)
, m_FocusedWindowDisplayGeometryTag(-1)
, m_FocusedWindowSliceSelectorTag(-1)
, m_FocusedWindowTimeStepSelectorTag(-1)
, m_DataStorage(NULL)
, m_BoundingBoxNode(NULL)
, m_BoundingBox(NULL)
, m_BaseRenderer(NULL)
, m_TrackedRenderWindow(NULL)
, m_TrackedWorldGeometry(NULL)
, m_TrackedDisplayGeometry(NULL)
, m_TrackedSliceNavigator(NULL)
, m_MouseEventEater(NULL)
, m_WheelEventEater(NULL)
, m_InDataStorageChanged(false)
, m_NodeAddedSetter(NULL)
, m_VisibilityTracker(NULL)
{
  // This should come early on, as we are setting renderer specific properties,
  // and when you set a renderer specific property, if the renderer is NULL,
  // it is an equivalent function call to setting a global property.
  m_BaseRenderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());

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

  std::vector<mitk::BaseRenderer*> renderers;
  renderers.push_back(m_BaseRenderer);

  std::vector<mitk::DataNode*> nodesToIgnore;
  nodesToIgnore.push_back(m_BoundingBoxNode);

  m_NodeAddedSetter = mitk::DataNodeAddedVisibilitySetter::New();

  // TODO Very ugly. This should be done in the other way round, from the MIDAS tools.
  //  mitk::MIDASDataNodeNameStringFilter::Pointer filter = mitk::MIDASDataNodeNameStringFilter::New();
  mitk::DataNodeStringPropertyFilter::Pointer filter = mitk::DataNodeStringPropertyFilter::New();
  filter->SetPropertyName("name");
  filter->AddToList("FeedbackContourTool");
  filter->AddToList("MIDASContourTool");
  filter->AddToList("MIDAS_SEEDS");
  filter->AddToList("MIDAS_CURRENT_CONTOURS");
  filter->AddToList("MIDAS_REGION_GROWING_IMAGE");
  filter->AddToList("MIDAS_PRIOR_CONTOURS");
  filter->AddToList("MIDAS_NEXT_CONTOURS");
  filter->AddToList("MIDAS_DRAW_CONTOURS");
  filter->AddToList("MORPH_EDITS_EROSIONS_SUBTRACTIONS");
  filter->AddToList("MORPH_EDITS_EROSIONS_ADDITIONS");
  filter->AddToList("MORPH_EDITS_DILATIONS_SUBTRACTIONS");
  filter->AddToList("MORPH_EDITS_DILATIONS_ADDITIONS");
  filter->AddToList("MIDAS PolyTool anchor points");
  filter->AddToList("MIDAS PolyTool previous contour");
  filter->AddToList("Paintbrush_Node");

  m_NodeAddedSetter->AddFilter(filter.GetPointer());
  m_NodeAddedSetter->SetRenderers(renderers);
  m_NodeAddedSetter->SetVisibility(false);

  m_VisibilityTracker = mitk::DataStorageVisibilityTracker::New();
  m_VisibilityTracker->SetRenderersToUpdate(renderers);
  m_VisibilityTracker->SetNodesToIgnore(nodesToIgnore);
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::AddBoundingBoxToDataStorage(const bool &add)
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  if (dataStorage.IsNotNull())
  {
    if (add && !dataStorage->Exists(m_BoundingBoxNode))
    {

      dataStorage->Add(m_BoundingBoxNode);
      this->setBoundingBoxVisible(true);

    } else if (!add && dataStorage->Exists(m_BoundingBoxNode))
    {
      dataStorage->Remove(m_BoundingBoxNode);
      this->setBoundingBoxVisible(false);
    }
  }
}


//-----------------------------------------------------------------------------
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

  this->AddBoundingBoxToDataStorage(false);
  this->OnFocusChanged();

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.AddListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.AddListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeChangedProxy ) );
  }
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::Deactivated()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
    m_FocusManagerObserverTag = -1;
  }

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener( mitk::MessageDelegate1<QmitkThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &QmitkThumbnailRenderWindow::NodeChangedProxy ) );
  }

  this->RemoveObserversFromTrackedObjects();
  this->AddBoundingBoxToDataStorage(false);
}


//-----------------------------------------------------------------------------
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

  if (m_TrackedWorldGeometry.IsNotNull() && m_TrackedDisplayGeometry.IsNotNull())
  {
    m_TrackedWorldGeometry->RemoveObserver(m_FocusedWindowWorldGeometryTag);
    m_FocusedWindowWorldGeometryTag = -1;
    m_TrackedDisplayGeometry->RemoveObserver(m_FocusedWindowDisplayGeometryTag);
    m_FocusedWindowDisplayGeometryTag = -1;
    m_TrackedSliceNavigator->RemoveObserver(m_FocusedWindowSliceSelectorTag);
    m_FocusedWindowSliceSelectorTag = -1;
    m_TrackedSliceNavigator->RemoveObserver(m_FocusedWindowTimeStepSelectorTag);
    m_FocusedWindowTimeStepSelectorTag = -1;
  }
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer QmitkThumbnailRenderWindow::GetDataStorage()
{
  // This MUST be set before you actually use this widget.
  assert(m_DataStorage);
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkThumbnailRenderWindow::Get3DPoint(int x, int y)
{
  mitk::Point3D pointInMillimetres3D;
  pointInMillimetres3D.Fill(0);

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


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::OnDisplayGeometryChanged()
{
  this->UpdateBoundingBox();
}


//-----------------------------------------------------------------------------
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

    // Add a bit of jitter so bounding box is on 2D.
    // So, this jitter adds depth to the bounding box in the through plane direction.
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

    // Request a single update at the end of the method.
    mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
  }
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged
      && node != NULL
      && !mitk::IsNodeAHelperObject(node)
      && node != m_BoundingBoxNode
      )
  {
    m_InDataStorageChanged = true;
    this->NodeAdded(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::NodeAdded( const mitk::DataNode* node)
{
  this->UpdateSliceAndTimeStep();
  this->OnDisplayGeometryChanged();
  this->UpdateBoundingBox();

  mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged
      && node != NULL
      && !mitk::IsNodeAHelperObject(node)
      && node != m_BoundingBoxNode
      )
  {
    m_InDataStorageChanged = true;
    this->NodeChanged(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::NodeChanged( const mitk::DataNode* node)
{
  mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::OnSliceChanged(const itk::EventObject & geometrySliceEvent)
{
  this->UpdateSliceAndTimeStep();
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::OnTimeStepChanged(const itk::EventObject & geometrySliceEvent)
{
  this->UpdateSliceAndTimeStep();
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::OnWorldGeometryChanged()
{
  this->UpdateWorldGeometry(true);
}


//-----------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------
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
        if (focusedWindowRenderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
        {
          vtkRenderWindow* focusedWindowRenderWindow = focusedWindowRenderer->GetRenderWindow();

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

              // I'm doing this in this method so that when the initial first
              // window starts (i.e. before any data is loaded),
              // the bounding box will not be included, and not visible.
              this->AddBoundingBoxToDataStorage(true);
              if (!dataStorage->Exists(m_BoundingBoxNode))
              {
                this->GetDataStorage()->Add(m_BoundingBoxNode);

              }

              this->UpdateSliceAndTimeStep();
              this->OnDisplayGeometryChanged();

              // Setup the visibility tracker.
              std::vector<mitk::BaseRenderer*> windowsToTrack;
              windowsToTrack.push_back(const_cast<mitk::BaseRenderer*>(focusedWindowRenderer.GetPointer()));
              m_VisibilityTracker->SetRenderersToTrack(windowsToTrack);
              m_VisibilityTracker->OnPropertyChanged(); // force update

              // Get the box to update
              this->UpdateBoundingBox();

              // Request a single update at the end of the method.
              mitk::RenderingManager::GetInstance()->RequestUpdate(this->GetVtkRenderWindow());
            }
          } // end if focused window is not thumbnail window.
        } // end if 2D mapper
      } // end if we do actually have a focused window
    } // end if we have a focus manager
  } // end if we have a data storage
}


//-----------------------------------------------------------------------------
QColor QmitkThumbnailRenderWindow::boundingBoxColor() const
{
  float colour[3];
  m_BoundingBoxNode->GetColor(colour);

  QColor qtColour(static_cast<int>(colour[0] * 255), static_cast<int>(colour[1] * 255), static_cast<int>(colour[2] * 255));
  return qtColour;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxColor(QColor &colour)
{
  m_BoundingBoxNode->SetColor(colour.redF(), colour.greenF(), colour.blueF());
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxColor(float r, float g, float b)
{
  m_BoundingBoxNode->SetColor(r, g, b);
}


//-----------------------------------------------------------------------------
int QmitkThumbnailRenderWindow::boundingBoxLineThickness() const
{
  int thickness = 0;
  m_BoundingBoxNode->GetIntProperty("line width", thickness);
  return thickness;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxLineThickness(int thickness)
{
  m_BoundingBoxNode->SetIntProperty("line width", thickness);
}


//-----------------------------------------------------------------------------
float QmitkThumbnailRenderWindow::boundingBoxOpacity() const
{
  float opacity = 0;
  m_BoundingBoxNode->GetFloatProperty("opacity", opacity);
  return opacity;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxOpacity(float opacity)
{
  m_BoundingBoxNode->SetOpacity(opacity);
}


//-----------------------------------------------------------------------------
bool QmitkThumbnailRenderWindow::boundingBoxVisible() const
{
  bool visible = false;
  m_BoundingBoxNode->GetBoolProperty("visible", visible, m_BaseRenderer);
  return visible;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxVisible(bool visible)
{
  m_BoundingBoxNode->SetBoolProperty("visible", visible, m_BaseRenderer);
}


//-----------------------------------------------------------------------------
int QmitkThumbnailRenderWindow::boundingBoxLayer() const
{
  bool layer = 0;
  m_BoundingBoxNode->GetBoolProperty("layer", layer);
  return layer;
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setBoundingBoxLayer(int layer)
{
  m_BoundingBoxNode->SetIntProperty("layer", layer);
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setRespondToMouseEvents(bool on)
{
  m_MouseEventEater->SetIsEating(!on);
}


//-----------------------------------------------------------------------------
bool QmitkThumbnailRenderWindow::respondToMouseEvents() const
{
  return !m_MouseEventEater->GetIsEating();
}


//-----------------------------------------------------------------------------
void QmitkThumbnailRenderWindow::setRespondToWheelEvents(bool on)
{
  m_WheelEventEater->SetIsEating(!on);
}


//-----------------------------------------------------------------------------
bool QmitkThumbnailRenderWindow::respondToWheelEvents() const
{
  return !m_WheelEventEater->GetIsEating();
}
