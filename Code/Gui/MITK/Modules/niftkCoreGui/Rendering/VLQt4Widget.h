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

#include <niftkCoreGuiExports.h>

#include <vlQt4/link_config.hpp>
#include <vlCore/VisualizationLibrary.hpp>
#include <vlGraphics/OpenGLContext.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/Camera.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/RenderingTree.hpp>
#include <vlGraphics/SceneManagerActorTree.hpp>
#include <vlGraphics/TrackballManipulator.hpp>
#include <vlGraphics/Geometry.hpp>
#include <vlGraphics/Uniform.hpp>
#include <vlGraphics/BlitFramebuffer.hpp>
#include <vlGraphics/Texture.hpp>
#include <QtGui/QMouseEvent>
#include <QtGui/QWidget>
#include <QtCore/QTimer>
#include <QtCore/QObject>
#include <QtOpenGL/QGLWidget>
#include <mitkOclResourceService.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <map>
#include <set>


// forward-decl
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
struct CUDAInterop;
class CUDAImage;
class CUDAImageProperty;
class LightweightCUDAImage;
namespace mitk
{
class DataStorage;
}

#include "OclTriangleSorter.h"


/**
 * This class is not thread-safe! Methods should only ever be called on the main
 * GUI thread.
 */
class NIFTKCOREGUI_EXPORT VLQt4Widget : public QGLWidget, public vl::OpenGLContext
{
  Q_OBJECT

public:
  using vl::Object::setObjectName;
  using QObject::setObjectName;

  VLQt4Widget(QWidget* parent=NULL, const QGLWidget* shareWidget=NULL, Qt::WindowFlags f=0);

  virtual ~VLQt4Widget();

  //bool initQt4Widget(const vl::String& title/*, const vl::OpenGLContextFormat& info, const QGLContext* shareContext=0*/, int x=0, int y=0, int width=640, int height=480);

  void setRefreshRate(int msec);
  int refreshRate();

  void setOclResourceService(OclResourceService* oclserv);

  void AddDataNode(const mitk::DataNode::ConstPointer& node);
  void RemoveDataNode(const mitk::DataNode::ConstPointer& node);
  void UpdateDataNode(const mitk::DataNode::ConstPointer& node);

  void QueueUpdateDataNode(const mitk::DataNode::ConstPointer& node);

  void ClearScene();

  void UpdateThresholdVal(int isoVal);

  /**
   * node can have as data object:
   * - mitk::Image
   * - CUDAImage
   * - mitk::Image with CUDAImageProperty attached.
   * And for now the image has to be 2D.
   * Anything else will just be ignored.
   */
  bool SetBackgroundNode(const mitk::DataNode::ConstPointer& node);

  /**
   * Returns the FBO that contains the current renderer output, i.e. the stuff that goes on screen.
   * Beware: this can/will return a different object every time you call it!
   */
  vl::FramebufferObject* GetFBO();

  /**
   * Will throw an exception if CUDA has not been enabled at compile time.
   */
  void EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage = 0, const std::string& nodename = "");


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

  virtual vl::ivec2 size() const;       // BEWARE: not a baseclass method!

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
  //void dragEnterEvent(QDragEnterEvent *ev);
  //void dropEvent(QDropEvent* ev);
private:
  QGLContext* context();    // non-const, hiding the one in QGLWidget.



protected:
<<<<<<< HEAD
  void RenderScene();
  void CreateAndUpdateFBOSizes(int width, int height);
  void UpdateViewportAndCamera();
  vl::ref<vl::Actor> AddSurfaceActor(const mitk::Surface::Pointer& mitkSurf);
  vl::ref<vl::Actor> AddImageActor(const mitk::Image::Pointer& mitkImg);
  void ConvertVTKPolyData(vtkPolyData* vtkPoly, vl::ref<vl::Geometry> vlPoly);
  static vl::String LoadGLSLSourceFromResources(const char* filename);

  void sortTranslucentTriangles();

  // side note: default actor block is zero
  static const int      RENDERBLOCK_OPAQUE            = -1000;
  static const int      RENDERBLOCK_TRANSLUCENT       =  1000;
  static const int      ENABLEMASK_OPAQUE             = 1 << 0;
  static const int      ENABLEMASK_TRANSLUCENT        = 1 << 1;
  static const int      ENABLEMASK_VOLUME             = 1 << 2;
  static const int      ENABLEMASK_BACKGROUND         = 1 << 3;

  vl::ref<vl::RenderingTree>            m_RenderingTree;
  vl::ref<vl::Rendering>                m_OpaqueObjectsRendering;
  vl::ref<vl::Rendering>                m_VolumeRendering;
  vl::ref<vl::Rendering>                m_BackgroundRendering;
  vl::ref<vl::BlitFramebuffer>          m_FinalBlit;
  vl::ref<vl::SceneManagerActorTree>    m_SceneManager;
  vl::ref<vl::Camera>                   m_Camera;
  vl::ref<vl::Camera>                   m_BackgroundCamera;
  vl::ref<vl::Light>                    m_Light;
  vl::ref<vl::Transform>                m_LightTr;
  vl::ref<vl::TrackballManipulator>     m_Trackball;

  vl::ref<vl::Uniform>                  m_ThresholdVal;   // iso value for volume

  OclResourceService*                   m_OclService;

  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >     m_NodeToActorMap;
  std::map<vl::ref<vl::Actor>, vl::ref<vl::Renderable> >          m_ActorToRenderableMap;
  std::set<mitk::DataNode::ConstPointer>                          m_NodesQueuedForUpdate;
  mitk::DataNode::ConstPointer                                    m_BackgroundNode;


  /** @name CUDA-interop related bits. */
  //@{

  /**
   * @throws an exception if CUDA support was not enabled at compile time.
   */
  void PrepareBackgroundActor(const LightweightCUDAImage* lwci, const mitk::Geometry3D* geom, const mitk::DataNode::ConstPointer node);

  /** Will throw if CUDA-support was not enabled at compile time. */
  vl::ref<vl::Actor> AddCUDAImageActor(const mitk::BaseData* cudaImg);

  // will only be non-null if cuda support is enabled at compile time.
  CUDAInterop*         m_CUDAInteropPimpl;

  struct TextureDataPOD
  {
    vl::ref<vl::Texture>    m_Texture;            // on the vl side
    unsigned int            m_LastUpdatedID;      // on cuda-manager side
    cudaGraphicsResource_t  m_CUDARes;            // on cuda(-driver) side

    TextureDataPOD();
  };
  std::map<mitk::DataNode::ConstPointer, TextureDataPOD>     m_NodeToTextureMap;
  //@}

  vl::ref<vl::Actor>         m_TranslucentSurfaceActor;
  mitk::OclTriangleSorter *  m_OclTriangleSorter;

protected:
  int       m_Refresh;
  QTimer    m_UpdateTimer;
};


#endif
