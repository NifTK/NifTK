/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#ifndef Qt4Window_INCLUDE_ONCE
#define Qt4Window_INCLUDE_ONCE

#include <niftkVLExports.h>

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

class VLTrackballManipulator;

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
// VLNode
//-----------------------------------------------------------------------------

/** 
 * Takes care of managing all VL related aspects with regard to a given mitk::DataNode, ie, maps a mitk::DataNode to VL/Vivid.
 */
class VLNode: public vl::Object {
public:
  VLNode( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node ) {
    // Init
    m_OpenGLContext = gl;
    m_VividRendering = vr;
    m_DataStorage = ds;
    m_DataNode = node;
    // Activate OpenGL context
    gl->makeCurrent();
    // Initialize properties
    initDataStoreProperties();
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

  /** Factory method: creates the right VLNode subclass according to the node's type. */
  static vl::ref<VLNode> create( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node );

  /** Returns the vl::Actor associated with this VLNode. Note: the specific subclass might handle more than one vl::Actor. */
  const vl::Actor* actor() const { return m_Actor.get(); }
  /** Returns the vl::Actor associated with this VLNode. Note: the specific subclass might handle more than one vl::Actor. */
  vl::Actor* actor() { return m_Actor.get(); }

  /** Updates visibility, opacity and color. */
  void updateCommon();

private:
  /** Initializes the value of all Vivid properties in the DataStore. */
  void initDataStoreProperties();

protected:
  vl::OpenGLContext* m_OpenGLContext;
  vl::VividRendering* m_VividRendering;
  mitk::DataStorage* m_DataStorage;
  const mitk::DataNode* m_DataNode;
  vl::ref<vl::Actor> m_Actor;
};

/**
 * This class is not thread-safe! Methods should only ever be called on the main
 * GUI thread.
 */
class NIFTKVL_EXPORT VLQtWidget : public QGLWidget, public vl::OpenGLContext
{
  Q_OBJECT

public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<vl::Actor> > NodeActorMapType;
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLNode> > NodeVLNodeMapType;

public:
  VLQtWidget(QWidget* parent=NULL, const QGLWidget* shareWidget=NULL, Qt::WindowFlags f=0);

  virtual ~VLQtWidget();

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
  void ScheduleSceneRebuild() { ClearScene(); update(); }

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

protected:
  void InitSceneFromDataStorage();
  void ClearScene();
  void UpdateScene();
  void RenderScene();

  void AddDataNode(const mitk::DataNode::ConstPointer& node);
  void RemoveDataNode(const mitk::DataNode::ConstPointer& node);
  void UpdateDataNode(const mitk::DataNode::ConstPointer& node);

  virtual void AddDataStorageListeners();
  virtual void RemoveDataStorageListeners();

  vl::ref<vl::Actor> AddPointsetActor(const mitk::PointSet::Pointer& mitkPS);
  vl::ref<vl::Actor> AddPointCloudActor(niftk::PCLData* pcl);

  void CreateAndUpdateFBOSizes(int width, int height);
  void UpdateViewportAndCameraAfterResize();
  void UpdateCameraParameters();

  void PrepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node);
  vl::Actor* GetNodeActor(const mitk::DataNode::ConstPointer& node);

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

  NodeActorMapType m_NodeActorMap;
  NodeVLNodeMapType m_NodeVLNodeMap;
  std::set<mitk::DataNode::ConstPointer> m_NodesToUpdate;
  std::set<mitk::DataNode::ConstPointer> m_NodesToAdd;
  std::set<mitk::DataNode::ConstPointer> m_NodesToRemove;
  mitk::DataNode::ConstPointer           m_BackgroundNode;
  mitk::DataNode::ConstPointer           m_CameraNode;

  bool m_ScheduleTrackballAdjustView;
  // these two will go away once we render the background using Vivid
  int m_BackgroundWidth;  
  int m_BackgroundHeight;

  // Lgacy OpenCL service
  OclResourceService* m_OclService;

#ifdef _USE_CUDA
public:
  /**
    * Will throw an exception if CUDA has not been enabled at compile time.
    */
  void EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage = 0, const std::string& nodename = "");

protected:
  /** @name CUDA-interop related bits. */
  //@{

  /**
    * @throws an exception if CUDA support was not enabled at compile time.
    */
  void PrepareBackgroundActor(const niftk::LightweightCUDAImage* lwci, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node);

  /** @throws an exception if CUDA support was not enabled at compile time. */
  void UpdateGLTexturesFromCUDA(const mitk::DataNode::ConstPointer& node);

  /** @throws an exception if CUDA support was not enabled at compile time. */
  void FreeCUDAInteropTextures();

    /** Will throw if CUDA-support was not enabled at compile time. */
  vl::ref<vl::Actor> AddCUDAImageActor(const mitk::BaseData* cudaImg);

  // will only be non-null if cuda support is enabled at compile time.
  CUDAInterop* m_CUDAInteropPimpl;

#endif

  // --------------------------------------------------------------------------
  // Things that should be inherited from vl::Qt5Widget
  // --------------------------------------------------------------------------

  // from vl::OpenGLContext
public:
  virtual void setContinuousUpdate(bool continuous);
  virtual void setWindowTitle(const vl::String& title);
  virtual bool setFullscreen(bool fullscreen);
  virtual void show();
  virtual void hide();
  virtual void setPosition(int x, int y);
  virtual vl::ivec2 position() const;
  virtual void update();                // hides non-virtual QWidget::update()?
  virtual void setSize(int w, int h);
  virtual void swapBuffers();           // in QGLWidget too
  virtual void makeCurrent();           // in QGLWidget too
  virtual void setMousePosition(int x, int y);
  virtual void setMouseVisible(bool visible);
  virtual void getFocus();

  virtual vl::ivec2 size() const;       // BEWARE: not a base class method!

  void setRefreshRate(int msec);
  int refreshRate();

protected:
  void translateKeyEvent(QKeyEvent* ev, unsigned short& unicode_out, vl::EKey& key_out);

  // from QGLWidget
protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();
  virtual void mouseMoveEvent(QMouseEvent* ev);
  virtual void mousePressEvent(QMouseEvent* ev);
  virtual void mouseReleaseEvent(QMouseEvent* ev);
  virtual void wheelEvent(QWheelEvent* ev);
  virtual void keyPressEvent(QKeyEvent* ev);
  virtual void keyReleaseEvent(QKeyEvent* ev);
  // void dragEnterEvent(QDragEnterEvent *ev);
  // void dropEvent(QDropEvent* ev);

protected:
  int    m_Refresh;
  QTimer m_UpdateTimer;
  QTimer m_BackgroundUpdateTimer;

};


#endif
