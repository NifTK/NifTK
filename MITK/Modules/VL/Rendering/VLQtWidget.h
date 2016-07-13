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
#include <vlVivid/VividVolume.hpp>
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

#ifdef _USE_CUDA

  struct cudaGraphicsResource;

  typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

  struct CUDAInterop;

  namespace niftk
  {
    class CUDAImage;
    class CUDAImageProperty;
    class LightweightCUDAImage;
  }

#endif

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

// VLMapper
// - makeCurrent(): when creating, updating and deleting? Or should we do it externally and remove m_OpenGLContext

/** 
 * Takes care of managing all VL related aspects with regard to a given mitk::DataNode, ie, maps a mitk::DataNode to VL/Vivid.
 */
class VLMapper: public vl::Object {
public:
  VLMapper( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node ) {
    // Init
    m_OpenGLContext = gl;
    m_VividRendering = vr;
    m_DataStorage = ds;
    m_DataNode = node;
    // Activate OpenGL context
    gl->makeCurrent();
    // Initialize properties
    initVLPropertiesGlobal();
  }

  virtual ~VLMapper() {
    remove();
  }

  /** Initializes all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual void init() = 0;

  /** Updates all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual void update() = 0;

  /** Removes all the relevant Actor(s) from the scene. */
  virtual void remove() {
    m_VividRendering->sceneManager()->tree()->eraseActor( m_Actor.get() );
    m_Actor = NULL;
  }

  /** Factory method: creates the right VLMapper subclass according to the node's type. */
  static vl::ref<VLMapper> create( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node );

  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  const vl::Actor* actor() const { return m_Actor.get(); }
  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  vl::Actor* actor() { return m_Actor.get(); }

  /** Updates visibility, opacity, color, etc. and Vivid related common settings. */
  void updateCommon();

protected:
  // VL global properties
  void initVLPropertiesGlobal();
  // VL surface properties
  void initVLPropertiesSurface();
  // VL point set properties
  void initVLPropertiesPointSet();
  // VL volume properties
  void initVLPropertiesVolume();
  // Initialize an Actor to be used with the Vivid renderer
  vl::ref<vl::Actor> initActor(vl::Geometry* geom, vl::Effect* fx);

protected:
  vl::OpenGLContext* m_OpenGLContext;
  vl::VividRendering* m_VividRendering;
  mitk::DataStorage* m_DataStorage;
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

class NIFTKVL_EXPORT VLSceneView: public vl::UIEventListener
{
public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLMapper> > DataNodeVLMapperMapType;

public:
  VLSceneView();

  // Called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void SetDataStorage(const mitk::DataStorage::Pointer& dataStorage);
  // Called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void SetOclResourceService(OclResourceService* oclserv);
  // Called by QmitkIGIVLEditor (via IGIVLEditor)
  void SetBackgroundColour(float r, float g, float b); 
  // Called by VLRendererView
  void UpdateThresholdVal(int isoVal); 

  void ScheduleNodeAdd(const mitk::DataNode* node);
  void ScheduleNodeRemove(const mitk::DataNode* node);
  void ScheduleNodeUpdate(const mitk::DataNode* node);
  void ScheduleTrackballAdjustView( bool do_it =  true ) { m_ScheduleTrackballAdjustView = do_it; }
  void ScheduleSceneRebuild() { ClearScene(); m_ScheduleInitScene = true; openglContext()->update(); }

  // Called by QmitkIGIVLEditor::OnImageSelected(), VLRendererView::OnBackgroundNodeSelected()
  /** 
   * node can have as data object:
   * - mitk::Image
   * - CUDAImage
   * - mitk::Image with CUDAImageProperty attached.
   * And for now the image has to be 2D.
   * Anything else will just be ignored.
   */
  bool SetBackgroundNode(const mitk::DataNode::ConstPointer& node);

  // Called by QmitkIGIVLEditor::OnTransformSelected(), VLRendererView::OnCameraNodeSelected()/OnCameraNodeEnabled()
  bool SetCameraTrackingNode(const mitk::DataNode::ConstPointer& node);

  VLTrackballManipulator* trackball() { return m_Trackball.get(); }
  const VLTrackballManipulator* trackball() const { return m_Trackball.get(); }
  vl::Camera* camera() { return m_Camera.get(); }
  const vl::Camera* camera() const { return m_Camera.get(); }

protected:
  bool contextIsCurrent() { return openglContext() && QGLContext::currentContext() == openglContext()->as<vlQt5::Qt5Widget>()->QGLWidget::context(); }
  
  void InitSceneFromDataStorage();
  void ClearScene();
  void UpdateScene();
  void RenderScene();

  void AddDataNode(const mitk::DataNode::ConstPointer& node);
  void RemoveDataNode(const mitk::DataNode::ConstPointer& node);
  void UpdateDataNode(const mitk::DataNode::ConstPointer& node);

  virtual void AddDataStorageListeners();
  virtual void RemoveDataStorageListeners();

  void CreateAndUpdateFBOSizes(int width, int height);
  void UpdateViewportAndCameraAfterResize();
  void UpdateCameraParameters();

  void PrepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node);
  VLMapper* GetVLMapper( const mitk::DataNode::ConstPointer& node );

protected:
  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::VividVolume>           m_VividVolume;
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
  // these two will go away once we render the background using Vivid
  int m_BackgroundWidth;  
  int m_BackgroundHeight;

  // Lgacy OpenCL service

  OclResourceService* m_OclService;

  // CUDA support

#ifdef _USE_CUDA
public:
  void EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage = 0, const std::string& nodename = "");

protected:
  virtual void cudaSwapBuffers();

  void PrepareBackgroundActor(const niftk::LightweightCUDAImage* lwci, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node);

  void UpdateGLTexturesFromCUDA(const mitk::DataNode::ConstPointer& node);

  void FreeCUDAInteropTextures();

  vl::ref<vl::Actor> AddCUDAImageActor(const mitk::BaseData* cudaImg);

  CUDAInterop* m_CUDAInteropPimpl;

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
  virtual void mouseMoveEvent(int,int) { }
  virtual void mouseUpEvent(vl::EMouseButton,int,int) { }
  virtual void mouseDownEvent(vl::EMouseButton,int,int) { }
  virtual void mouseWheelEvent(int) { }
  virtual void keyPressEvent(unsigned short,vl::EKey) { }
  virtual void keyReleaseEvent(unsigned short,vl::EKey) { }
  virtual void fileDroppedEvent(const std::vector<vl::String>&) { }
};

//-----------------------------------------------------------------------------
// VLQtWidget
//-----------------------------------------------------------------------------

class VLQtWidget: public vlQt5::Qt5Widget {
public:
  VLQtWidget(QWidget* parent=NULL, const QGLWidget* shareWidget=NULL, Qt::WindowFlags f=0)
  : Qt5Widget(parent, shareWidget, f) {
    m_VLSceneView = new VLSceneView;
    addEventListener( m_VLSceneView.get() );
    setRefreshRate( 1000 / 30 ); // 30 fps in milliseconds
    setContinuousUpdate(false);
    setMouseTracking(true);
    setAutoBufferSwap(false);
    setAcceptDrops(false);
  }

  void setVLSceneView( VLSceneView* vl_view ) { m_VLSceneView = vl_view; }
  VLSceneView* vlSceneView() { return m_VLSceneView.get(); }
  const VLSceneView* vlSceneView() const { return m_VLSceneView.get(); }

protected:
  vl::ref<VLSceneView> m_VLSceneView;
};

#endif
