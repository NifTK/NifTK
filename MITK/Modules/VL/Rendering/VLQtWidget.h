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

class TrackballManipulator;

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

/**
 * This class is not thread-safe! Methods should only ever be called on the main
 * GUI thread.
 */
class NIFTKVL_EXPORT VLQtWidget : public QGLWidget, public vl::OpenGLContext
{
  Q_OBJECT

public:
  using vl::Object::setObjectName;
  using QObject::setObjectName;

  VLQtWidget(QWidget* parent=NULL, const QGLWidget* shareWidget=NULL, Qt::WindowFlags f=0);

  virtual ~VLQtWidget();

  void ScheduleNodeAdd(const mitk::DataNode* node);
  void ScheduleNodeRemove(const mitk::DataNode* node);
  void ScheduleNodeUpdate(const mitk::DataNode* node);
  void ScheduleTrackballAdjustView( bool do_it =  true ) { m_ScheduleTrackballAdjustView = do_it; }

  void setRefreshRate(int msec);
  int refreshRate();

  // --------------------------------------------------------------------------

  void SetOclResourceService(OclResourceService* oclserv);
  void SetDataStorage(const mitk::DataStorage::Pointer& dataStorage);

  void UpdateThresholdVal(int isoVal);

  // ignore alpha for now.
  Q_SLOT void SetBackgroundColour(float r, float g, float b);

  /**
   * node can have as data object:
   * - mitk::Image
   * - CUDAImage
   * - mitk::Image with CUDAImageProperty attached.
   * And for now the image has to be 2D.
   * Anything else will just be ignored.
   */
  bool SetBackgroundNode(const mitk::DataNode::ConstPointer& node);

  bool SetCameraTrackingNode(const mitk::DataNode::ConstPointer& node);

  void ScheduleSceneRebuild() {
    ClearScene();
    update();
  }

private:
  void ClearScene();

protected:
  void InitSceneFromDataStorage();

  virtual void AddDataStorageListeners();
  virtual void RemoveDataStorageListeners();

  void AddDataNode(const mitk::DataNode::ConstPointer& node);
  void RemoveDataNode(const mitk::DataNode::ConstPointer& node);
  void UpdateDataNode(const mitk::DataNode::ConstPointer& node);

  void UpdateScene();
  virtual void OnNodeModified(const mitk::DataNode* node);
  virtual void OnNodeVisibilityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);
  virtual void OnNodeColorPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);
  virtual void OnNodeOpacityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer = 0);

  void RenderScene();
  void CreateAndUpdateFBOSizes(int width, int height);
  Q_SLOT void UpdateViewportAndCameraAfterResize();
  void UpdateCameraParameters();
  void UpdateTextureFromImage(const mitk::DataNode::ConstPointer& node);
  void UpdateActorTransformFromNode(vl::ref<vl::Actor> actor, const mitk::DataNode::ConstPointer& node);
  void UpdateTransformFromNode(vl::ref<vl::Transform> txf, const mitk::DataNode::ConstPointer& node);
  void UpdateTransformFromData(vl::ref<vl::Transform> txf, const mitk::BaseData::ConstPointer& data);
  vl::mat4 GetVLMatrixFromData(const mitk::BaseData::ConstPointer& data);
  void EnableTrackballManipulator(bool enable);
  vl::ref<vl::Actor> AddPointsetActor(const mitk::PointSet::Pointer& mitkPS);
  vl::ref<vl::Actor> AddPointCloudActor(niftk::PCLData* pcl);
  vl::ref<vl::Actor> AddSurfaceActor(const mitk::Surface::Pointer& mitkSurf);
  vl::ref<vl::Actor> AddImageActor(const mitk::Image::Pointer& mitkImg);
  vl::ref<vl::Actor> Add2DImageActor(const mitk::Image::Pointer& mitkImg);
  vl::ref<vl::Actor> Add3DImageActor(const mitk::Image::Pointer& mitkImg);
  vl::ref<vl::Actor> AddCoordinateAxisActor(const mitk::CoordinateAxesData::Pointer& coord);
  vl::EImageType MapITKPixelTypeToVL(int itkComponentType);
  vl::EImageFormat MapComponentsToVLColourFormat(int components);
  void ConvertVTKPolyData(vtkPolyData* vtkPoly, vl::ref<vl::Geometry> vlPoly);
  static vl::String LoadGLSLSourceFromResources(const char* filename);
  void PrepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node);
  vl::ref<vl::Geometry> CreateGeometryFor2DImage(int width, int height);
  vl::ref<vl::Actor> FindActorForNode(const mitk::DataNode::ConstPointer& node);
  vl::ref<VLUserData> GetUserData(vl::ref<vl::Actor> actor);

  mitk::DataStorage::Pointer              m_DataStorage;
  mitk::DataNodePropertyListener::Pointer m_NodeVisibilityListener;
  mitk::DataNodePropertyListener::Pointer m_NodeColorPropertyListener;
  mitk::DataNodePropertyListener::Pointer m_NodeOpacityPropertyListener;

  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::VividVolume>           m_VividVolume;
  vl::ref<vl::SceneManagerActorTree> m_SceneManager;
  vl::ref<vl::Camera>                m_Camera;
  vl::ref<TrackballManipulator>      m_Trackball;

  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> > m_NodeToActorMap;
  std::set<mitk::DataNode::ConstPointer>                      m_NodesToUpdate;
  std::set<mitk::DataNode::ConstPointer>                      m_NodesToAdd;
  std::set<mitk::DataNode::ConstPointer>                      m_NodesToRemove;
  mitk::DataNode::ConstPointer                                m_BackgroundNode;
  mitk::DataNode::ConstPointer                                m_CameraNode;
  int m_BackgroundWidth;
  int m_BackgroundHeight;
  bool m_ScheduleTrackballAdjustView;

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

    struct TextureDataPOD
    {
      vl::ref<vl::Texture>   m_Texture;       // on the vl side
      unsigned int           m_LastUpdatedID; // on cuda-manager side
      cudaGraphicsResource_t m_CUDARes;       // on cuda(-driver) side

      TextureDataPOD();
    };
    std::map<mitk::DataNode::ConstPointer, TextureDataPOD> m_NodeToTextureMap;
  #endif

  // Currently not used: left here just in case we need it for future fun :)
  OclResourceService* m_OclService;

protected:
  int    m_Refresh;
  QTimer m_UpdateTimer;

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
};


#endif
