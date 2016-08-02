/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkThumbnailRenderWindow.h"

#include <itkCommand.h>

#include <mitkDataStorage.h>
#include <mitkDisplayGeometry.h>
#include <mitkGlobalInteraction.h>

#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleContext.h>
#include <usModuleRegistry.h>

#include <vtkCubeSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <niftkDataStorageUtils.h>
#include <niftkMouseEventEater.h>
#include <niftkWheelEventEater.h>


namespace niftk
{

//-----------------------------------------------------------------------------
ThumbnailRenderWindow::ThumbnailRenderWindow(QWidget *parent, mitk::RenderingManager* renderingManager)
: QmitkRenderWindow(parent, "thumbnail viewer", 0, renderingManager)
, m_DataStorage(NULL)
, m_BoundingBoxNode(NULL)
, m_BoundingBox(NULL)
, m_Renderer(NULL)
, m_TrackedRenderer(NULL)
, m_TrackedRenderingManager(NULL)
, m_TrackedWorldGeometry(NULL)
, m_TrackedDisplayGeometry(NULL)
, m_TrackedSliceNavigator(NULL)
, m_TrackedRendererTag(-1)
, m_TrackedWorldGeometryTag(-1)
, m_TrackedDisplayGeometryTag(-1)
, m_TrackedSliceSelectorTag(-1)
, m_TrackedTimeStepSelectorTag(-1)
, m_MouseEventEater(NULL)
, m_WheelEventEater(NULL)
, m_InDataStorageChanged(false)
, m_VisibilityTracker(NULL)
{
  m_DataStorage = renderingManager->GetDataStorage();
  assert(m_DataStorage.IsNotNull());

  // This should come early on, as we are setting renderer specific properties,
  // and when you set a renderer specific property, if the renderer is NULL,
  // it is an equivalent function call to setting a global property.
  m_Renderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());

  m_BoundingBox = mitk::Cuboid::New();
  m_BoundingBoxNode = mitk::DataNode::New();
  m_BoundingBoxNode->SetData(m_BoundingBox);
  m_BoundingBoxNode->SetProperty("name", mitk::StringProperty::New("ThumbnailBoundingBox"));
  m_BoundingBoxNode->SetProperty("includeInBoundingBox", mitk::BoolProperty::New(false));
  m_BoundingBoxNode->SetProperty("helper object", mitk::BoolProperty::New(true));
  m_BoundingBoxNode->SetBoolProperty("visible", false); // globally turn it off, then we only turn it on in thumbnail (this) window.

  m_BoundingBoxNode->SetBoolProperty("visible", false, m_Renderer);
  this->SetBoundingBoxOpacity(1);
  this->SetBoundingBoxLineThickness(1);
  this->SetBoundingBoxLayer(99);// arbitrary, copied from segmentation functionality

  /// TODO Very ugly. This should be done in the other way round, from the MIDAS tools.

  m_ToolNodeNameFilter = DataNodeStringPropertyFilter::New();
  m_ToolNodeNameFilter->SetPropertyName("name");
  m_ToolNodeNameFilter->AddToList("One of FeedbackContourTool's feedback nodes");
  m_ToolNodeNameFilter->AddToList("MIDAS Background Contour");
  m_ToolNodeNameFilter->AddToList("MIDAS_SEEDS");
  m_ToolNodeNameFilter->AddToList("MIDAS_CURRENT_CONTOURS");
  m_ToolNodeNameFilter->AddToList("MIDAS_REGION_GROWING_IMAGE");
  m_ToolNodeNameFilter->AddToList("MIDAS_PRIOR_CONTOURS");
  m_ToolNodeNameFilter->AddToList("MIDAS_NEXT_CONTOURS");
  m_ToolNodeNameFilter->AddToList("MIDAS_DRAW_CONTOURS");
  m_ToolNodeNameFilter->AddToList("MORPH_EDITS_EROSIONS_SUBTRACTIONS");
  m_ToolNodeNameFilter->AddToList("MORPH_EDITS_EROSIONS_ADDITIONS");
  m_ToolNodeNameFilter->AddToList("MORPH_EDITS_DILATIONS_SUBTRACTIONS");
  m_ToolNodeNameFilter->AddToList("MORPH_EDITS_DILATIONS_ADDITIONS");
  m_ToolNodeNameFilter->AddToList("MORPHO_SEGMENTATION_OF_LAST_STAGE");
  m_ToolNodeNameFilter->AddToList("PolyTool anchor points");
  m_ToolNodeNameFilter->AddToList("PolyTool previous contour");
  m_ToolNodeNameFilter->AddToList("Paintbrush_Node");

  m_VisibilityTracker = DataNodeVisibilityTracker::New(m_DataStorage);

  std::vector<const mitk::BaseRenderer*> renderers;
  renderers.push_back(m_Renderer);
  m_VisibilityTracker->SetManagedRenderers(renderers);

  std::vector<mitk::DataNode*> nodesToIgnore;
  nodesToIgnore.push_back(m_BoundingBoxNode);
  m_VisibilityTracker->SetNodesToIgnore(nodesToIgnore);

  m_VisibilityTracker->AddFilter(m_ToolNodeNameFilter.GetPointer());

  m_MouseEventEater = new MouseEventEater();
  m_MouseEventEater->SetIsEating(false);
  this->installEventFilter(m_MouseEventEater);

  m_WheelEventEater = new WheelEventEater();
  m_WheelEventEater->SetIsEating(true);
  this->installEventFilter(m_WheelEventEater);
}


//-----------------------------------------------------------------------------
ThumbnailRenderWindow::~ThumbnailRenderWindow()
{
  this->SetTrackedRenderer(0);

  // Release the display interactor.
  this->SetDisplayInteractionsEnabled(false);

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
bool ThumbnailRenderWindow::AreDisplayInteractionsEnabled() const
{
  return m_DisplayInteractor.IsNotNull();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetDisplayInteractionsEnabled(bool enabled)
{
  if (enabled == this->AreDisplayInteractionsEnabled())
  {
    // Already enabled/disabled.
    return;
  }

  if (enabled)
  {
    // Here we create our own display interactor...
    m_DisplayInteractor = ThumbnailInteractor::New(this);

    us::Module* thisModule = us::ModuleRegistry::GetModule("niftkThumbnail");
    m_DisplayInteractor->LoadStateMachine("ThumbnailInteraction.xml", thisModule);
    m_DisplayInteractor->SetEventConfig("DisplayConfigMITK.xml");

    // ... and register it as listener via the micro services.
    us::ServiceProperties props;
    props["name"] = std::string("ThumbnailInteractor");

    us::ModuleContext* moduleContext = us::GetModuleContext();
    m_DisplayInteractorService = moduleContext->RegisterService<mitk::InteractionEventObserver>(m_DisplayInteractor.GetPointer(), props);
  }
  else
  {
    // Unregister the display interactor service.
    m_DisplayInteractorService.Unregister();
    // Release the display interactor to let it be desctructed.
    m_DisplayInteractor = 0;
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::AddBoundingBoxToDataStorage()
{
  if (!m_DataStorage->Exists(m_BoundingBoxNode))
  {
    m_BoundingBoxNode->SetBoolProperty("visible", true, m_Renderer);
    m_DataStorage->Add(m_BoundingBoxNode);
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::RemoveBoundingBoxFromDataStorage()
{
  if (m_DataStorage->Exists(m_BoundingBoxNode))
  {
    m_DataStorage->Remove(m_BoundingBoxNode);
    m_BoundingBoxNode->SetBoolProperty("visible", false, m_Renderer);
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::Activated()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.AddListener( mitk::MessageDelegate1<ThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &ThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.AddListener( mitk::MessageDelegate1<ThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &ThumbnailRenderWindow::NodeChangedProxy ) );
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::Deactivated()
{
  if (m_TrackedWorldGeometry.IsNotNull())
  {
    this->RemoveObserversFromTrackedObjects();
    this->RemoveBoundingBoxFromDataStorage();
  }

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.RemoveListener( mitk::MessageDelegate1<ThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &ThumbnailRenderWindow::NodeAddedProxy ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener( mitk::MessageDelegate1<ThumbnailRenderWindow, const mitk::DataNode*>
      ( this, &ThumbnailRenderWindow::NodeChangedProxy ) );
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::AddObserversToTrackedObjects()
{
  assert(m_TrackedRenderer.IsNotNull());
  assert(m_TrackedWorldGeometry.IsNotNull());
  assert(m_TrackedDisplayGeometry.IsNotNull());
  assert(m_TrackedSliceNavigator.IsNotNull());

  // Add Observers to track when these geometries change
  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onRendererChangedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onRendererChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::UpdateWorldGeometry);
  m_TrackedRendererTag = m_TrackedRenderer->AddObserver(itk::ModifiedEvent(), onRendererChangedCommand);

  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onWorldGeometryChangedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onWorldGeometryChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::UpdateWorldGeometry);
  m_TrackedWorldGeometryTag = m_TrackedWorldGeometry->AddObserver(itk::ModifiedEvent(), onWorldGeometryChangedCommand);

  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onDisplayGeometryChangedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onDisplayGeometryChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::UpdateBoundingBox);
  m_TrackedDisplayGeometryTag = m_TrackedDisplayGeometry->AddObserver(itk::ModifiedEvent(), onDisplayGeometryChangedCommand);

  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onSliceChangedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onSliceChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::UpdateSliceAndTimeStep);
  m_TrackedSliceSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(NULL, 0), onSliceChangedCommand);

  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onTimeChangedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onTimeChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::UpdateSliceAndTimeStep);
  m_TrackedTimeStepSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometryTimeEvent(NULL, 0), onTimeChangedCommand);
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::RemoveObserversFromTrackedObjects()
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

  assert(m_TrackedRenderer.IsNotNull());
  assert(m_TrackedWorldGeometry.IsNotNull());
  assert(m_TrackedDisplayGeometry.IsNotNull());
  assert(m_TrackedSliceNavigator.IsNotNull());

  m_TrackedRenderer->RemoveObserver(m_TrackedRendererTag);
  m_TrackedRendererTag = -1;
  m_TrackedWorldGeometry->RemoveObserver(m_TrackedWorldGeometryTag);
  m_TrackedWorldGeometryTag = -1;
  m_TrackedDisplayGeometry->RemoveObserver(m_TrackedDisplayGeometryTag);
  m_TrackedDisplayGeometryTag = -1;
  m_TrackedSliceNavigator->RemoveObserver(m_TrackedSliceSelectorTag);
  m_TrackedSliceSelectorTag = -1;
  m_TrackedSliceNavigator->RemoveObserver(m_TrackedTimeStepSelectorTag);
  m_TrackedTimeStepSelectorTag = -1;
}


//-----------------------------------------------------------------------------
mitk::Point3D ThumbnailRenderWindow::Get3DPoint(int x, int y)
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
void ThumbnailRenderWindow::UpdateBoundingBox()
{
  if (m_TrackedDisplayGeometry.IsNotNull())
  {
    // Get min and max extent of the tracked render window's display geometry.
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

    if (m_TrackedSliceNavigator)
    {
      mitk::SliceNavigationController::ViewDirection viewDirection = m_TrackedSliceNavigator->GetViewDirection();
      if (viewDirection == mitk::SliceNavigationController::Frontal)
      {
        m_BoundingBoxNode->SetColor(0, 0, 255);
      }
      else if (viewDirection == mitk::SliceNavigationController::Sagittal)
      {
        m_BoundingBoxNode->SetColor(0, 255, 0);
      }
      else if (viewDirection == mitk::SliceNavigationController::Axial)
      {
        m_BoundingBoxNode->SetColor(255, 0, 0);
      }
      else
      {
        m_BoundingBoxNode->SetColor(0, 255, 255);
      }
    }

    m_BoundingBox->Modified();
    m_BoundingBoxNode->Modified();

    // Tidy up
    cube->Delete();

    // Request a single update at the end of the method.
    m_Renderer->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::NodeAddedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged
      && node != NULL
      && !niftk::IsNodeAHelperObject(node)
      && node != m_BoundingBoxNode
      )
  {
    m_InDataStorageChanged = true;
    this->OnNodeAdded(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnNodeAdded( const mitk::DataNode* node)
{
  this->UpdateWorldGeometry();
  this->UpdateSliceAndTimeStep();

  this->UpdateBoundingBox();

  m_Renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::NodeChangedProxy( const mitk::DataNode* node )
{
  // Guarantee no recursions when a new node event is created in NodeAdded()
  if(!m_InDataStorageChanged
      && node != NULL
      && !niftk::IsNodeAHelperObject(node)
      && node != m_BoundingBoxNode
      )
  {
    m_InDataStorageChanged = true;
    this->OnNodeChanged(node);
    m_InDataStorageChanged = false;
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnNodeChanged( const mitk::DataNode* node)
{
  m_Renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::UpdateSliceAndTimeStep()
{
  if (m_TrackedRenderer.IsNotNull())
  {
    if (m_TrackedRenderer->GetTimeStep() != m_Renderer->GetTimeStep())
    {
      m_Renderer->SetTimeStep(m_TrackedRenderer->GetTimeStep());
    }

    if (m_TrackedRenderer->GetSlice() != m_Renderer->GetSlice())
    {
      m_Renderer->SetSlice(m_TrackedRenderer->GetSlice());
    }
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::UpdateWorldGeometry()
{
  if (m_TrackedRenderer && m_TrackedRenderer->GetWorldTimeGeometry())
  {
    // World geometry of thumbnail must be same (or larger) as world geometry of the tracked window.
    m_Renderer->SetWorldTimeGeometry(m_TrackedRenderer->GetWorldTimeGeometry());

    // Display geometry of widget must encompass whole of world geometry
    m_Renderer->GetDisplayGeometry()->Fit();

    // Request a single update at the end of the method.
    m_Renderer->RequestUpdate();
  }
}


//-----------------------------------------------------------------------------
mitk::BaseRenderer::Pointer ThumbnailRenderWindow::GetTrackedRenderer() const
{
  return m_TrackedRenderer;
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetTrackedRenderer(mitk::BaseRenderer::Pointer rendererToTrack)
{
  if (rendererToTrack == m_TrackedRenderer)
  {
    return;
  }

  if (m_TrackedRenderer.IsNotNull())
  {
    if (m_TrackedWorldGeometry.IsNotNull())
    {
      assert(m_TrackedDisplayGeometry.IsNotNull());
      assert(m_TrackedSliceNavigator.IsNotNull());

      this->RemoveObserversFromTrackedObjects();
      this->RemoveBoundingBoxFromDataStorage();

      m_TrackedWorldGeometry = 0;
      m_TrackedDisplayGeometry = 0;
      m_TrackedSliceNavigator = 0;
    }

    m_TrackedRenderer = 0;
  }

  mitk::RenderingManager* trackedRenderingManager = rendererToTrack ? rendererToTrack->GetRenderingManager() : 0;
  if (trackedRenderingManager != m_TrackedRenderingManager)
  {
    if (m_TrackedRenderingManager)
    {
      m_TrackedRenderingManager->RemoveRenderWindow(this->GetRenderWindow());
    }

    m_TrackedRenderingManager = trackedRenderingManager;

    if (m_TrackedRenderingManager)
    {
      m_TrackedRenderingManager->AddRenderWindow(this->GetRenderWindow());
    }
  }

  m_TrackedRenderer = rendererToTrack;

  if (m_TrackedRenderer.IsNotNull())
  {
    m_TrackedWorldGeometry = const_cast<mitk::BaseGeometry*>(rendererToTrack->GetWorldGeometry());
    m_TrackedDisplayGeometry = const_cast<mitk::DisplayGeometry*>(rendererToTrack->GetDisplayGeometry());
    m_TrackedSliceNavigator = (const_cast<mitk::BaseRenderer*>(rendererToTrack.GetPointer()))->GetSliceNavigationController();

    if (m_TrackedWorldGeometry.IsNotNull())
    {
      // I'm doing this in this method so that when the initial first
      // window starts (i.e. before any data is loaded),
      // the bounding box will not be included, and not visible.
      this->AddBoundingBoxToDataStorage();

      this->UpdateWorldGeometry();
      this->UpdateSliceAndTimeStep();

      this->AddObserversToTrackedObjects();
    }
  }


  // Setup the visibility tracker.
  m_VisibilityTracker->SetTrackedRenderer(const_cast<mitk::BaseRenderer*>(rendererToTrack.GetPointer()));

  // Get the box to update
  this->UpdateBoundingBox();

  // Request a single update at the end of the method.
  m_Renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnBoundingBoxPanned(const mitk::Vector2D& displacement)
{
  if (!m_TrackedDisplayGeometry)
  {
    return;
  }

  double ownScaleFactor = m_Renderer->GetDisplayGeometry()->GetScaleFactorMMPerDisplayUnit();
  double trackedGeometryScaleFactor = m_TrackedDisplayGeometry->GetScaleFactorMMPerDisplayUnit();
  mitk::Vector2D trackedGeometryDisplacement = displacement * ownScaleFactor / trackedGeometryScaleFactor;
  m_TrackedDisplayGeometry->MoveBy(trackedGeometryDisplacement);
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnBoundingBoxZoomed(double scaleFactor)
{
  if (!m_TrackedDisplayGeometry)
  {
    return;
  }

  mitk::Vector2D displaySize = m_TrackedDisplayGeometry->GetSizeInDisplayUnits();
  mitk::Point2D centreInPx;
  centreInPx[0] = displaySize[0] / 2;
  centreInPx[1] = displaySize[1] / 2;
  m_TrackedDisplayGeometry->Zoom(scaleFactor, centreInPx);
}


//-----------------------------------------------------------------------------
int ThumbnailRenderWindow::GetBoundingBoxLineThickness() const
{
  int thickness = 0;
  m_BoundingBoxNode->GetIntProperty("line width", thickness);
  return thickness;
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetBoundingBoxLineThickness(int thickness)
{
  m_BoundingBoxNode->SetIntProperty("line width", thickness);
}


//-----------------------------------------------------------------------------
float ThumbnailRenderWindow::GetBoundingBoxOpacity() const
{
  float opacity = 0;
  m_BoundingBoxNode->GetFloatProperty("opacity", opacity);
  return opacity;
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetBoundingBoxOpacity(float opacity)
{
  m_BoundingBoxNode->SetOpacity(opacity);
}


//-----------------------------------------------------------------------------
bool ThumbnailRenderWindow::GetBoundingBoxVisible() const
{
  bool visible = false;
  m_BoundingBoxNode->GetBoolProperty("visible", visible, m_Renderer);
  return visible;
}


//-----------------------------------------------------------------------------
int ThumbnailRenderWindow::GetBoundingBoxLayer() const
{
  bool layer = 0;
  m_BoundingBoxNode->GetBoolProperty("layer", layer);
  return layer;
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetBoundingBoxLayer(int layer)
{
  m_BoundingBoxNode->SetIntProperty("layer", layer);
}


//-----------------------------------------------------------------------------
bool ThumbnailRenderWindow::GetRespondToMouseEvents() const
{
  return !m_MouseEventEater->GetIsEating();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetRespondToMouseEvents(bool on)
{
  m_MouseEventEater->SetIsEating(!on);
}


//-----------------------------------------------------------------------------
bool ThumbnailRenderWindow::GetRespondToWheelEvents() const
{
  return !m_WheelEventEater->GetIsEating();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetRespondToWheelEvents(bool on)
{
  m_WheelEventEater->SetIsEating(!on);
}

}