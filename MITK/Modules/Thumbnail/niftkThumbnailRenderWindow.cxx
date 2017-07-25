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
: QmitkRenderWindow(parent, "thumbnail viewer", nullptr, renderingManager),
  m_DataStorage(nullptr),
  m_BoundingBoxNode(nullptr),
  m_BoundingBox(nullptr),
  m_Renderer(nullptr),
  m_TrackedRenderer(nullptr),
  m_TrackedRenderingManager(nullptr),
  m_TrackedWorldTimeGeometry(nullptr),
  m_TrackedSliceNavigator(nullptr),
  m_TrackedDisplayGeometry(nullptr),
  m_TrackedRendererTag(-1),
  m_TrackedWorldTimeGeometryTag(-1),
  m_TrackedTimeStepSelectorTag(-1),
  m_TrackedSliceSelectorTag(-1),
  m_TrackedDisplayGeometryTag(-1),
  m_MouseEventEater(nullptr),
  m_WheelEventEater(nullptr),
  m_VisibilityTracker(nullptr)
{
  m_DataStorage = renderingManager->GetDataStorage();
  assert(m_DataStorage.IsNotNull());

  // This should come early on, as we are setting renderer specific properties,
  // and when you set a renderer specific property, if the renderer is nullptr,
  // it is an equivalent function call to setting a global property.
  m_Renderer = mitk::BaseRenderer::GetInstance(this->GetVtkRenderWindow());

  m_BoundingBox = mitk::Cuboid::New();
  m_BoundingBoxNode = mitk::DataNode::New();
  m_BoundingBoxNode->SetData(m_BoundingBox);
  m_BoundingBoxNode->SetName("ThumbnailBoundingBox");
  m_BoundingBoxNode->SetBoolProperty("includeInBoundingBox", false);
  m_BoundingBoxNode->SetBoolProperty("helper object", true);
  m_BoundingBoxNode->SetVisibility(false); // globally turn it off, then we only turn it on in thumbnail (this) window.

  this->SetBoundingBoxOpacity(1);
  this->SetBoundingBoxLineThickness(1.0f);
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

  this->RegisterInteractor();
}


//-----------------------------------------------------------------------------
ThumbnailRenderWindow::~ThumbnailRenderWindow()
{
  this->UnregisterInteractor();

  if (m_TrackedRenderer.IsNotNull())
  {
    this->UntrackRenderer();
  }

  if (m_TrackedRenderingManager)
  {
    m_TrackedRenderingManager->RemoveRenderWindow(this->GetRenderWindow());
  }

  if (m_DataStorage->Exists(m_BoundingBoxNode))
  {
    m_DataStorage->Remove(m_BoundingBoxNode);
  }

  if (m_MouseEventEater != nullptr)
  {
    delete m_MouseEventEater;
  }

  if (m_WheelEventEater != nullptr)
  {
    delete m_WheelEventEater;
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::RegisterInteractor()
{
  assert(m_DisplayInteractor.IsNull());

  // Here we create our own display interactor...
  m_DisplayInteractor = ThumbnailInteractor::New(this);

  us::Module* thisModule = us::ModuleRegistry::GetModule("niftkThumbnail");
  m_DisplayInteractor->LoadStateMachine("ThumbnailInteraction.xml", thisModule);
  m_DisplayInteractor->SetEventConfig("ThumbnailConfig.xml", thisModule);

  /// The interactor will be dynamically enabled/disabled as the container view is activated/deactivated.
  m_DisplayInteractor->Disable();

  // ... and register it as listener via the micro services.
  us::ServiceProperties props;
  props["name"] = std::string("ThumbnailInteractor");

  us::ModuleContext* moduleContext = us::GetModuleContext();
  m_DisplayInteractorService = moduleContext->RegisterService<mitk::InteractionEventObserver>(m_DisplayInteractor.GetPointer(), props);
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::UnregisterInteractor()
{
  assert(m_DisplayInteractor.IsNotNull());

  // Unregister the display interactor service.
  m_DisplayInteractorService.Unregister();
  // Release the display interactor to let it be desctructed.
  m_DisplayInteractor = nullptr;
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
    this->UntrackRenderer();
  }

  mitk::RenderingManager* renderingManagerToTrack = rendererToTrack ? rendererToTrack->GetRenderingManager() : nullptr;
  if (renderingManagerToTrack != m_TrackedRenderingManager)
  {
    if (m_TrackedRenderingManager)
    {
      m_TrackedRenderingManager->RemoveRenderWindow(this->GetRenderWindow());
    }

    m_TrackedRenderingManager = renderingManagerToTrack;

    if (m_TrackedRenderingManager)
    {
      m_TrackedRenderingManager->AddRenderWindow(this->GetRenderWindow());
    }
  }

  m_TrackedRenderer = rendererToTrack;

  if (m_TrackedRenderer.IsNotNull())
  {
    this->TrackRenderer();
  }

  // Setup the visibility tracker.
  m_VisibilityTracker->SetTrackedRenderer(rendererToTrack);

  // Request a single update at the end of the method.
  m_Renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
float ThumbnailRenderWindow::GetBoundingBoxLineThickness() const
{
  float thickness = 0.0f;
  m_BoundingBoxNode->GetFloatProperty("line width", thickness);
  return thickness;
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::SetBoundingBoxLineThickness(float thickness)
{
  m_BoundingBoxNode->SetFloatProperty("line width", thickness);
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
  m_TrackedRenderingManager->RequestUpdateAll();
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
  m_TrackedRenderingManager->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::TrackRenderer()
{
  itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onRendererModifiedCommand =
    itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
  onRendererModifiedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnRendererModified);
  m_TrackedRendererTag = m_TrackedRenderer->AddObserver(itk::ModifiedEvent(), onRendererModifiedCommand);

  this->OnRendererModified();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::UntrackRenderer()
{
  // NOTE the following curiosity:
  //
  // Initially my m_TrackedWorldTimeGeometry and m_TrackedDisplayGeometry were regular pointers
  // not smart pointers. However, when exiting the application, it appeared that they were
  // not valid, causing seg faults.  It appeared to me (Matt), that as the focus changed between
  // windows, the slice navigation controller is constant for each render window, but the
  // geometries are created/deleted as needed, as the geometry changed.
  // So, as a work-around, I made these pointers smart pointers, so that if the geometry objects are
  // deleted by their renderers, then this smart pointer in this class will still have a valid reference
  // to the object that it originally promised to listen to.  This will avoid crashes, and the geometry
  // object will go out of scope when it is replaced with a new one, or this object is destroyed.

  m_TrackedRenderer->RemoveObserver(m_TrackedRendererTag);
  m_TrackedRendererTag = -1;
  m_TrackedRenderer = nullptr;

  if (m_TrackedWorldTimeGeometry.IsNotNull())
  {
    m_TrackedWorldTimeGeometry->RemoveObserver(m_TrackedWorldTimeGeometryTag);
    m_TrackedWorldTimeGeometryTag = -1;
    m_TrackedWorldTimeGeometry = nullptr;
  }

  if (m_TrackedSliceNavigator.IsNotNull())
  {
    m_TrackedSliceNavigator->RemoveObserver(m_TrackedSliceSelectorTag);
    m_TrackedSliceSelectorTag = -1;
    m_TrackedSliceNavigator->RemoveObserver(m_TrackedTimeStepSelectorTag);
    m_TrackedTimeStepSelectorTag = -1;
    m_TrackedSliceNavigator = nullptr;
  }

  if (m_TrackedDisplayGeometry.IsNotNull())
  {
    m_TrackedDisplayGeometry->RemoveObserver(m_TrackedDisplayGeometryTag);
    m_TrackedDisplayGeometryTag = -1;
    m_TrackedDisplayGeometry = nullptr;
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnRendererModified()
{
  assert(m_TrackedRenderer.IsNotNull());

  mitk::TimeGeometry* worldTimeGeometryToTrack = m_TrackedRenderer->GetWorldTimeGeometry();
  mitk::SliceNavigationController* sliceNavigatorToTrack = m_TrackedRenderer->GetSliceNavigationController();
  mitk::DisplayGeometry* displayGeometryToTrack = m_TrackedRenderer->GetDisplayGeometry();

  if (worldTimeGeometryToTrack != m_TrackedWorldTimeGeometry)
  {
    if (m_TrackedWorldTimeGeometry.IsNotNull())
    {
      m_TrackedWorldTimeGeometry->RemoveObserver(m_TrackedWorldTimeGeometryTag);
      m_TrackedWorldTimeGeometryTag = -1;
    }

    m_TrackedWorldTimeGeometry = worldTimeGeometryToTrack;

    if (m_TrackedWorldTimeGeometry.IsNotNull())
    {
      itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onWorldTimeGeometryChangedCommand =
        itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
      onWorldTimeGeometryChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnWorldTimeGeometryModified);
      m_TrackedWorldTimeGeometryTag = m_TrackedWorldTimeGeometry->AddObserver(itk::ModifiedEvent(), onWorldTimeGeometryChangedCommand);

      /// The renderer has a display geometry even without a valid world time geometry.
      /// We only start listening to display events when the renderer gets a valid
      /// world time geometry.
      /// So, if the display geometry has not changed but it did not have a listener
      /// because it has not had a world geometry so far, we assign a new listener now.
      /// If the display geometry has changed now, a new listener will be assigned
      /// later below.
      if (m_TrackedDisplayGeometry == displayGeometryToTrack
          && m_TrackedDisplayGeometry.IsNotNull()
          && m_TrackedDisplayGeometryTag == -1)
      {
        itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onDisplayGeometryChangedCommand =
          itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
        onDisplayGeometryChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnDisplayGeometryModified);
        m_TrackedDisplayGeometryTag = m_TrackedDisplayGeometry->AddObserver(itk::ModifiedEvent(), onDisplayGeometryChangedCommand);
      }
    }
  }

  if (sliceNavigatorToTrack != m_TrackedSliceNavigator)
  {
    if (m_TrackedSliceNavigator.IsNotNull())
    {
      m_TrackedSliceNavigator->RemoveObserver(m_TrackedTimeStepSelectorTag);
      m_TrackedTimeStepSelectorTag = -1;
      m_TrackedSliceNavigator->RemoveObserver(m_TrackedSliceSelectorTag);
      m_TrackedSliceSelectorTag = -1;
    }

    m_TrackedSliceNavigator = sliceNavigatorToTrack;

    if (m_TrackedSliceNavigator.IsNotNull())
    {
      itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onSelectedTimeStepChangedCommand =
        itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
      onSelectedTimeStepChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnSelectedTimeStepChanged);
      m_TrackedTimeStepSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometryTimeEvent(nullptr, 0), onSelectedTimeStepChangedCommand);

      itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onSelectedSliceChangedCommand =
        itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
      onSelectedSliceChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnSelectedSliceChanged);
      m_TrackedSliceSelectorTag = m_TrackedSliceNavigator->AddObserver(mitk::SliceNavigationController::GeometrySliceEvent(nullptr, 0), onSelectedSliceChangedCommand);
    }
  }

  if (displayGeometryToTrack != m_TrackedDisplayGeometry)
  {
    if (m_TrackedDisplayGeometry.IsNotNull())
    {
      m_TrackedDisplayGeometry->RemoveObserver(m_TrackedDisplayGeometryTag);
      m_TrackedDisplayGeometryTag = -1;
    }

    m_TrackedDisplayGeometry = displayGeometryToTrack;

    if (m_TrackedDisplayGeometry.IsNotNull() && m_TrackedWorldTimeGeometry.IsNotNull())
    {
      itk::SimpleMemberCommand<ThumbnailRenderWindow>::Pointer onDisplayGeometryChangedCommand =
        itk::SimpleMemberCommand<ThumbnailRenderWindow>::New();
      onDisplayGeometryChangedCommand->SetCallbackFunction(this, &ThumbnailRenderWindow::OnDisplayGeometryModified);
      m_TrackedDisplayGeometryTag = m_TrackedDisplayGeometry->AddObserver(itk::ModifiedEvent(), onDisplayGeometryChangedCommand);
    }
  }

  if (m_TrackedWorldTimeGeometry.IsNotNull()
      && m_TrackedSliceNavigator.IsNotNull()
      && m_TrackedDisplayGeometry.IsNotNull())
  {
    /// The bounding box is not added to the data storage until there is a tracked
    /// renderer with a valid geometry. This function is called when a new geometry
    /// is set for the renderer (it might not have had one) or when its current
    /// geometry is modified.
    /// Also, the data node can be removed from the data storage unintentedly, e.g.
    /// by closing the project.
    /// Therefore, here we add the bounding box to the data storage if necessary.
    if (!m_DataStorage->Exists(m_BoundingBoxNode))
    {
      m_DataStorage->Add(m_BoundingBoxNode);
    }

    this->OnWorldTimeGeometryModified();
    this->OnSelectedTimeStepChanged();
    this->OnSelectedSliceChanged();
    this->OnDisplayGeometryModified();
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnWorldTimeGeometryModified()
{
  assert(m_TrackedRenderer);
  assert(m_TrackedWorldTimeGeometry);

  // World geometry of thumbnail must be same (or larger) as world geometry of the tracked window.
  m_Renderer->SetWorldTimeGeometry(m_TrackedWorldTimeGeometry);

  // Display geometry of widget must encompass whole of world geometry
  m_Renderer->GetDisplayGeometry()->Fit();

  // Request a single update at the end of the method.
  m_Renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnSelectedTimeStepChanged()
{
  assert(m_TrackedRenderer.IsNotNull());

  if (m_TrackedRenderer->GetTimeStep() != m_Renderer->GetTimeStep())
  {
    m_Renderer->SetTimeStep(m_TrackedRenderer->GetTimeStep());
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnSelectedSliceChanged()
{
  assert(m_TrackedRenderer.IsNotNull());

  if (m_TrackedRenderer->GetSlice() != m_Renderer->GetSlice())
  {
    m_Renderer->SetSlice(m_TrackedRenderer->GetSlice());
  }
}


//-----------------------------------------------------------------------------
void ThumbnailRenderWindow::OnDisplayGeometryModified()
{
  assert(m_TrackedRenderer.IsNotNull());
  assert(m_TrackedWorldTimeGeometry.IsNotNull());
  assert(m_TrackedSliceNavigator.IsNotNull());
  assert(m_TrackedDisplayGeometry.IsNotNull());

  // Get min and max extent of the tracked render window's display geometry.
  mitk::Point3D min, max;

  mitk::Point2D point2D;
  point2D[0] = 0;
  point2D[1] = 0;
  m_TrackedDisplayGeometry->DisplayToWorld(point2D, point2D);
  m_TrackedDisplayGeometry->Map(point2D, min);
  m_TrackedRenderer->GetWorldGeometry()->WorldToIndex(min, min);

  point2D[0] = m_TrackedDisplayGeometry->GetDisplayWidth() - 1;
  point2D[1] = m_TrackedDisplayGeometry->GetDisplayHeight() - 1;
  m_TrackedDisplayGeometry->DisplayToWorld(point2D, point2D);
  m_TrackedDisplayGeometry->Map(point2D, max);
  m_TrackedRenderer->GetWorldGeometry()->WorldToIndex(max, max);

  int planeAxis = -1;
  for (int axis = 0; axis < 3; ++axis)
  {
    if (std::abs(min[axis] - max[axis]) < 0.0001)
    {
      planeAxis = axis;
    }
  }

  if (planeAxis == -1)
  {
    MITK_DEBUG << "ThumbnailRenderWindow::UpdateBoundingBox(): Cannot find plane axis.";
    m_BoundingBoxNode->SetVisibility(false, m_Renderer);
    return;
  }

  if (!m_BoundingBoxNode->IsVisible(m_Renderer))
  {
    m_BoundingBoxNode->SetVisibility(true, m_Renderer);
  }

  // Add a bit of jitter so bounding box is on 2D.
  // So, this jitter adds depth to the bounding box in the through plane direction.
  min[planeAxis] -= 0.5;
  max[planeAxis] += 0.5;

  // Create a cube.
  vtkCubeSource* cube = vtkCubeSource::New();
  cube->SetBounds(min[0], max[0], min[1], max[1], min[2], max[2]);
  cube->Update();

  // Update bounding box.
  m_BoundingBox->SetVtkPolyData(cube->GetOutput());
  m_BoundingBox->SetGeometry(m_TrackedRenderer->GetWorldGeometry());

  // Tidy up
  cube->Delete();

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

  m_BoundingBox->Modified();
  m_BoundingBoxNode->Modified();

  // Request a single update at the end of the method.
  m_Renderer->RequestUpdate();
}

}
