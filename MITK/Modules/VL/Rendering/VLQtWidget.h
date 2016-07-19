/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VLQtWidget_INCLUDE_ONCE
#define VLQtWidget_INCLUDE_ONCE

#include <niftkVLExports.h>

#include <vlQt5/Qt5Widget.hpp>
#include <vlGraphics/OpenGLContext.hpp>
#include <vlVivid/VividRenderer.hpp>
#include <vlVivid/VividRendering.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/Camera.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/RenderingTree.hpp>
#include <vlGraphics/SceneManagerActorTree.hpp>
#include <vlGraphics/Geometry.hpp>
#include <vlGraphics/Uniform.hpp>
#include <vlGraphics/BlitFramebuffer.hpp>
#include <vlGraphics/Texture.hpp>
#include <vlCore/VisualizationLibrary.hpp>
#include "TrackballManipulator.h"
#include <QMouseEvent>
#include <QWidget>
#include <QTimer>
#include <QObject>
#include <QGLWidget>
#include <mitkOclResourceService.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <mitkCoordinateAxesData.h>
#include <mitkDataNodePropertyListener.h>
#include <map>
#include <set>

// Forward declarations

namespace mitk
{
  class DataStorage;
}

namespace niftk
{
  class PCLData;
}

struct VLUserData;

class VLSceneView;

#ifdef _USE_CUDA

  class CudaTest;

#endif

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

// VLMapper
// - makeCurrent(): when creating, updating and deleting? Or should we do it externally and remove m_OpenGLContext

/**
 * Takes care of managing all VL related aspects with regard to a given mitk::DataNode, ie, maps a mitk::DataNode to VL/Vivid.
 */
class VLMapper : public vl::Object {
public:
  VLMapper(const mitk::DataNode* node, VLSceneView* sv);

  virtual ~VLMapper() {
    remove();
  }

  /** Initializes all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual void init() = 0;

  /** Updates all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual void update() = 0;

  /** Removes all the relevant Actor(s) from the scene. */
  virtual void remove() {
    m_VividRendering->sceneManager()->tree()->eraseActor(m_Actor.get());
    m_Actor = NULL;
  }

  /** Factory method: creates the right VLMapper subclass according to the node's type. */
  static vl::ref<VLMapper> create(const mitk::DataNode* node, VLSceneView*);

  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  vl::Actor* actor() { return m_Actor.get(); }
  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  const vl::Actor* actor() const { return m_Actor.get(); }

  /** Updates visibility, opacity, color, etc. and Vivid related common settings. */
  void updateCommon();

protected:
  // Initialize an Actor to be used with the Vivid renderer
  vl::ref<vl::Actor> initActor(vl::Geometry* geom, vl::Effect* fx = NULL, vl::Transform* tr = NULL);

protected:
  vl::OpenGLContext* m_OpenGLContext;
  vl::VividRendering* m_VividRendering;
  mitk::DataStorage* m_DataStorage;
  VLSceneView* m_VLSceneView;
  const mitk::DataNode* m_DataNode;
  vl::ref<vl::Actor> m_Actor;
};

//-----------------------------------------------------------------------------
// VLSceneView
// We could further divide this into:
// * VLScene:
//  - view-independent data that can be shared across VLViews and VLQtWidgets.
//  - receives data store events
//  - upon data store update signals an update request to all VLViews.
// * VLSceneView:
//  - view-dependent data and manipulators: camera, trackball, background settings, etc.
//  - receives use interaction events
// This to allow the VL and OpenGL data to be instanced only once and shared across OpenGL contexts.
// For the moment we use the simpler and safer approach of instancing data for every VLQtWidget.
//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLSceneView : public vl::UIEventListener
{
public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLMapper> > DataNodeVLMapperMapType;

public:
  VLSceneView();
  ~VLSceneView();

  // Called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void setDataStorage(const mitk::DataStorage::Pointer& dataStorage);
  // Called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void setOclResourceService(OclResourceService* oclserv);
  // Called by QmitkIGIVLEditor (via IGIVLEditor)
  void setBackgroundColour(float r, float g, float b);
  // Called by VLRendererView
  void updateThresholdVal(int isoVal);

  void scheduleNodeAdd(const mitk::DataNode* node);
  void scheduleNodeRemove(const mitk::DataNode* node);
  void scheduleNodeUpdate(const mitk::DataNode* node);
  void scheduleTrackballAdjustView(bool do_it = true) { m_ScheduleTrackballAdjustView = do_it; }
  void scheduleSceneRebuild() { clearScene(); m_ScheduleInitScene = true; openglContext()->update(); }

  mitk::DataStorage* dataStorage() { return m_DataStorage.GetPointer(); }
  const mitk::DataStorage* dataStorage() const { return m_DataStorage.GetPointer(); }

  vl::VividRendering* vividRendering() { return m_VividRendering.get(); }
  const vl::VividRendering* vividRendering() const { return m_VividRendering.get(); }

  // Called by QmitkIGIVLEditor::OnImageSelected(), VLRendererView::OnBackgroundNodeSelected()
  /**
   * node can have as data object:
   * - mitk::Image
   * - CUDAImage
   * - mitk::Image with CUDAImageProperty attached.
   * And for now the image has to be 2D.
   * Anything else will just be ignored.
   */
  bool setBackgroundNode(const mitk::DataNode* node);

  // Called by QmitkIGIVLEditor::OnTransformSelected(), VLRendererView::OnCameraNodeSelected()/OnCameraNodeEnabled()
  bool setCameraTrackingNode(const mitk::DataNode* node);

  VLTrackballManipulator* trackball() { return m_Trackball.get(); }
  const VLTrackballManipulator* trackball() const { return m_Trackball.get(); }
  vl::Camera* camera() { return m_Camera.get(); }
  const vl::Camera* camera() const { return m_Camera.get(); }

protected:
  bool contextIsCurrent() { return openglContext() && QGLContext::currentContext() == openglContext()->as<vlQt5::Qt5Widget>()->QGLWidget::context(); }

  void initSceneFromDataStorage();
  void clearScene();
  void updateScene();
  void renderScene();

  // Returned VLMapper can be NULL
  VLMapper* addDataNode(const mitk::DataNode* node);
  void removeDataNode(const mitk::DataNode* node);
  void updateDataNode(const mitk::DataNode* node);

  virtual void addDataStorageListeners();
  virtual void removeDataStorageListeners();

  // void createAndUpdateFBOSizes(int width, int height); // obsolete
  void updateViewportAndCameraAfterResize();
  void updateCameraParameters();

  VLMapper* getVLMapper(const mitk::DataNode* node);

protected:
  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::SceneManagerActorTree> m_SceneManager;
  vl::ref<vl::Camera>                m_Camera;
  vl::ref<VLTrackballManipulator>    m_Trackball;

  mitk::DataStorage::Pointer              m_DataStorage;
  mitk::DataNodePropertyListener::Pointer m_NodeVisibilityListener;
  mitk::DataNodePropertyListener::Pointer m_NodeColorPropertyListener;
  mitk::DataNodePropertyListener::Pointer m_NodeOpacityPropertyListener;

  DataNodeVLMapperMapType                m_DataNodeVLMapperMap;
  std::set<mitk::DataNode::ConstPointer> m_NodesToUpdate;
  std::set<mitk::DataNode::ConstPointer> m_NodesToAdd;
  std::set<mitk::DataNode::ConstPointer> m_NodesToRemove;
  mitk::DataNode::ConstPointer           m_BackgroundNode;
  mitk::DataNode::ConstPointer           m_CameraNode;

  bool m_ScheduleTrackballAdjustView;
  bool m_ScheduleInitScene;

  // MIC FIXME: remove this
  vl::ivec2 m_BackgroundSize;

  // Lgacy OpenCL service

  OclResourceService* m_OclService;

  // CUDA support

#ifdef _USE_CUDA
protected:
  CudaTest* m_CudaTest;
#endif

protected:
  // --------------------------------------------------------------------------
  // vl::UIEventListener implementation
  // --------------------------------------------------------------------------

  virtual void initEvent();
  virtual void resizeEvent(int width, int height);
  virtual void updateEvent();
  virtual void destroyEvent();

  virtual void addedListenerEvent(vl::OpenGLContext *) { }
  virtual void removedListenerEvent(vl::OpenGLContext *) { }
  virtual void enableEvent(bool) { }
  virtual void visibilityEvent(bool) { }
  virtual void mouseMoveEvent(int, int) { }
  virtual void mouseUpEvent(vl::EMouseButton, int, int) { }
  virtual void mouseDownEvent(vl::EMouseButton, int, int) { }
  virtual void mouseWheelEvent(int) { }
  virtual void keyPressEvent(unsigned short, vl::EKey) { }
  virtual void keyReleaseEvent(unsigned short, vl::EKey) { }
  virtual void fileDroppedEvent(const std::vector<vl::String>&) { }
};

//-----------------------------------------------------------------------------
// VLQtWidget
//-----------------------------------------------------------------------------

class VLQtWidget : public vlQt5::Qt5Widget {
public:
  VLQtWidget(QWidget* parent = NULL, const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0)
    : Qt5Widget(parent, shareWidget, f) {
    m_VLSceneView = new VLSceneView;
    addEventListener(m_VLSceneView.get());
    setRefreshRate(1000 / 30); // 30 fps in milliseconds
    setContinuousUpdate(false);
    setMouseTracking(true);
    setAutoBufferSwap(false);
    setAcceptDrops(false);
  }

  void setVLSceneView(VLSceneView* vl_view) { m_VLSceneView = vl_view; }
  VLSceneView* vlSceneView() { return m_VLSceneView.get(); }
  const VLSceneView* vlSceneView() const { return m_VLSceneView.get(); }

protected:
  vl::ref<VLSceneView> m_VLSceneView;
};

#endif
