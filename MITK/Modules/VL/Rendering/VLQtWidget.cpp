/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <QTextStream>
#include <QFile>

#include "VLQtWidget.h"
#include <vlCore/Log.hpp>
#include <vlCore/Time.hpp>
#include <vlCore/Colors.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/RenderQueueSorter.hpp>
#include <vlGraphics/GLSL.hpp>
#include <vlGraphics/FramebufferObject.hpp>
#include <vlVolume/RaycastVolume.hpp>
#include <cassert>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataNormals.h>
#include <vtkImageData.h>
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <stdexcept>
#include <sstream>
#include <niftkScopedOGLContext.h>
#include "TrackballManipulator.h"
#ifdef BUILD_IGI
#include <CameraCalibration/niftkUndistortion.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkCameraIntrinsics.h>
#endif
#include <mitkCoordinateAxesData.h>

#ifdef _USE_PCL
#include <niftkPCLData.h>
#endif

#ifdef _MSC_VER
#ifdef _USE_NVAPI
#include <nvapi.h>
#endif
#endif

#ifdef _USE_CUDA
#include <Rendering/VLFramebufferToCUDA.h>
#include <niftkCUDAManager.h>
#include <niftkCUDAImage.h>
#include <niftkLightweightCUDAImage.h>
#include <niftkCUDAImageProperty.h>
#include <niftkFlipImageLauncher.h>
#include <cuda_gl_interop.h>


//-----------------------------------------------------------------------------
struct CUDAInterop
{
  std::string                     m_NodeName;
  mitk::DataStorage::Pointer      m_DataStorage;

  VLFramebufferAdaptor*           m_FBOAdaptor;

  CUDAInterop()
    : m_FBOAdaptor(0)
  {
  }

  ~CUDAInterop()
  {
    delete m_FBOAdaptor;
  }
};


//-----------------------------------------------------------------------------
VLQtWidget::TextureDataPOD::TextureDataPOD()
  : m_LastUpdatedID(0)
  , m_CUDARes(0)
{
}


#else
// empty dummy, in case we have no cuda
struct CUDAInterop { };
#endif // _USE_CUDA


//-----------------------------------------------------------------------------
namespace
{

class VLInit
{
public:
  VLInit()
  {
    vl::VisualizationLibrary::init();
  }
  ~VLInit()
  {
    vl::VisualizationLibrary::shutdown();
  }
};
VLInit        s_ModuleInit;

}


struct VLUserData : public vl::Object
{
  VLUserData()
    : m_TransformLastModified(0)
    , m_ImageVtkDataLastModified(0)
  {
  }


  itk::ModifiedTimeType   m_TransformLastModified;
  itk::ModifiedTimeType   m_ImageVtkDataLastModified;
};


//-----------------------------------------------------------------------------
VLQtWidget::VLQtWidget(QWidget* parent, const QGLWidget* shareWidget, Qt::WindowFlags f)
  : QGLWidget(parent, shareWidget, f)
  , m_BackgroundWidth(0)
  , m_BackgroundHeight(0)
  , m_CUDAInteropPimpl(0)
  , m_OclService(0)
  , m_OclTriangleSorter(0)
  , m_TranslucentStructuresMerged(false)
  , m_TranslucentStructuresSorted(false)
  , m_TotalNumOfTranslucentTriangles(0)
  , m_TotalNumOfTranslucentVertices(0)
  , m_MergedTranslucentIndexBuf(0)
  , m_MergedTranslucentVertexBuf(0)
  , m_Refresh(33) // 30 fps
{
  setContinuousUpdate(true);
  setMouseTracking(true);
  setAutoBufferSwap(false);
  setAcceptDrops(false);
  // let Qt take care of object destruction.
  vl::OpenGLContext::setAutomaticDelete(false);

  // remember: all opengl related init should happen in initializeGL()!
}


//-----------------------------------------------------------------------------
VLQtWidget::~VLQtWidget()
{
  niftk::ScopedOGLContext  ctx(this->context());

  RemoveDataStorageListeners();

  if (m_OclTriangleSorter)
    delete m_OclTriangleSorter;

  m_OclTriangleSorter = 0;

  if (m_MergedTranslucentVertexBuf)
  {
    clReleaseMemObject(m_MergedTranslucentVertexBuf);
    m_MergedTranslucentVertexBuf = 0;
  }

  if (m_MergedTranslucentIndexBuf)
  {
    clReleaseMemObject(m_MergedTranslucentIndexBuf);
    m_MergedTranslucentIndexBuf = 0;
  }

  dispatchDestroyEvent();


#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::FreeCUDAInteropTextures()
{
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_CUDA
  for (std::map<mitk::DataNode::ConstPointer, TextureDataPOD>::iterator i = m_NodeToTextureMap.begin(); i != m_NodeToTextureMap.end(); )
  {
    if (i->second.m_CUDARes != 0)
    {
      cudaError_t err = cudaGraphicsUnregisterResource(i->second.m_CUDARes);
      if (err != cudaSuccess)
      {
        MITK_WARN << "Failed to unregister VL texture from CUDA";
      }
    }

    i = m_NodeToTextureMap.erase(i);
  }

  // if no cuda is available then this is most likely a nullptr.
  // and if not a nullptr then it's only a dummy. so unconditionally delete it.
  delete m_CUDAInteropPimpl;
  m_CUDAInteropPimpl = 0;

#else
  throw std::runtime_error("CUDA support was not enabled");
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::OnNodeModified(const mitk::DataNode* node)
{
  mitk::DataNode::ConstPointer   dn(node);
  QueueUpdateDataNode(dn);
}


//-----------------------------------------------------------------------------
bool VLQtWidget::NodeIsTranslucent(const mitk::DataNode::ConstPointer& node)
{
  float opacity = 1.0f;
  mitk::FloatProperty* opacityProp = dynamic_cast<mitk::FloatProperty*>(node->GetProperty("opacity"));
  if (opacityProp != 0)
    opacity = opacityProp->GetValue();
  return opacity < 1.0f;
}


//-----------------------------------------------------------------------------
bool VLQtWidget::NodeIsOnTranslucentList(const mitk::DataNode::ConstPointer& node)
{
  if (!m_TranslucentActors.empty())
  {
    std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::const_iterator i = m_NodeToActorMap.find(node);
    if (i != m_NodeToActorMap.end())
    {
      vl::ref<vl::Actor>    nodeActor = i->second;
      assert(nodeActor.get() != 0);

      std::set<vl::ref<vl::Actor> >::const_iterator j = m_TranslucentActors.find(nodeActor);
      if (j != m_TranslucentActors.end())
      {
        return true;
      }
    }
  }

  return false;
}


//-----------------------------------------------------------------------------
void VLQtWidget::OnNodeVisibilityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  mitk::DataNode::ConstPointer  cdn(node);

  if (NodeIsTranslucent(cdn))
  {
    // we only need to recompute the merged buffer if the changed node is actually translucent.
    m_TranslucentStructuresMerged = false;
  }

  QueueUpdateDataNode(cdn);
}


//-----------------------------------------------------------------------------
void VLQtWidget::OnNodeColorPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  mitk::DataNode::ConstPointer  cdn(node);

  if (NodeIsTranslucent(cdn))
  {
    // we only need to recompute the merged buffer if the changed node is actually translucent.
    m_TranslucentStructuresMerged = false;
  }

  QueueUpdateDataNode(cdn);
}


//-----------------------------------------------------------------------------
void VLQtWidget::OnNodeOpacityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  m_TranslucentStructuresMerged = false;

  mitk::DataNode::ConstPointer  cdn(node);
  QueueUpdateDataNode(cdn);
}


//-----------------------------------------------------------------------------
void VLQtWidget::AddDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    // if someone calls node->Modified() we need to redraw.
    m_DataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<VLQtWidget, const mitk::DataNode*>(this, &VLQtWidget::OnNodeModified));

    m_NodeVisibilityListener = mitk::DataNodePropertyListener::New(m_DataStorage, "visible");
    m_NodeVisibilityListener->NodePropertyChanged += mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>(this, &VLQtWidget::OnNodeVisibilityPropertyChanged);

    m_NodeColorPropertyListener = mitk::DataNodePropertyListener::New(m_DataStorage, "color");
    m_NodeColorPropertyListener->NodePropertyChanged +=  mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>(this, &VLQtWidget::OnNodeColorPropertyChanged);

    m_NodeOpacityPropertyListener = mitk::DataNodePropertyListener::New(m_DataStorage, "opacity");
    m_NodeOpacityPropertyListener->NodePropertyChanged += mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>( this, &VLQtWidget::OnNodeOpacityPropertyChanged);

  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::RemoveDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener(mitk::MessageDelegate1<VLQtWidget, const mitk::DataNode*>(this, &VLQtWidget::OnNodeModified));

    if (m_NodeVisibilityListener)
      m_NodeVisibilityListener->NodePropertyChanged -= mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>(this, &VLQtWidget::OnNodeVisibilityPropertyChanged);
    if (m_NodeColorPropertyListener)
      m_NodeColorPropertyListener->NodePropertyChanged -=  mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>(this, &VLQtWidget::OnNodeColorPropertyChanged);
    if (m_NodeOpacityPropertyListener)
      m_NodeOpacityPropertyListener->NodePropertyChanged -= mitk::MessageDelegate2<VLQtWidget, mitk::DataNode*, const mitk::BaseRenderer*>( this, &VLQtWidget::OnNodeOpacityPropertyChanged);

  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::SetDataStorage(const mitk::DataStorage::Pointer& dataStorage)
{
  niftk::ScopedOGLContext  ctx(this->context());

  RemoveDataStorageListeners();

  ClearScene();

#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif

  m_DataStorage = dataStorage;
  AddDataStorageListeners();

  QMetaObject::invokeMethod(this, "AddAllNodesFromDataStorage", Qt::QueuedConnection);
}


//-----------------------------------------------------------------------------
void VLQtWidget::SetOclResourceService(OclResourceService* oclserv)
{
  // no idea if this is really a necessary restriction.
  // if it is then maybe the ocl-service should be a constructor parameter.
  if (m_OclService != 0)
    throw std::runtime_error("Can set OpenCL service only once");

  m_OclService = oclserv;
}


//-----------------------------------------------------------------------------
vl::FramebufferObject* VLQtWidget::GetFBO()
{
  // createAndUpdateFBOSizes() where we always stuff a proper fbo into the blit.
  return dynamic_cast<vl::FramebufferObject*>(m_FinalBlit->readFramebuffer());
}


//-----------------------------------------------------------------------------
void VLQtWidget::EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage, const std::string& nodename)
{
#ifdef _USE_CUDA
  niftk::ScopedOGLContext  ctx(this->context());

  if (enable)
  {
    if (datastorage == 0)
      throw std::runtime_error("Need data storage object");

    delete m_CUDAInteropPimpl;
    m_CUDAInteropPimpl = new CUDAInterop;
    m_CUDAInteropPimpl->m_FBOAdaptor = 0;
    m_CUDAInteropPimpl->m_DataStorage = datastorage;
    m_CUDAInteropPimpl->m_NodeName = nodename;
    if (m_CUDAInteropPimpl->m_NodeName.empty())
    {
      std::ostringstream    n;
      n << "0x" << std::hex << (void*) this;
      m_CUDAInteropPimpl->m_NodeName = n.str();
    }
  }
  else
  {
    delete m_CUDAInteropPimpl;
    m_CUDAInteropPimpl = 0;
  }
#else
  throw std::runtime_error("CUDA support was not enabled");
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::initializeGL()
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());

  vl::OpenGLContext::initGLContext();

  // use the device that is running our opengl context as the compute-device
  // for sorting triangles in the correct order.
  if (m_OclService)
  {
    // Have to call makeCurrent() otherwise the shared CL-GL context creation fails
    makeCurrent();

    // Force tests to run on the first GPU with shared context
    m_OclService->SpecifyPlatformAndDevice(0, 0, true);
    // Calling this to make sure that the context is created right at startup
    cl_context clContext = m_OclService->GetContext();
  }


#ifdef _MSC_VER
  //NvAPI_OGL_ExpertModeSet(NVAPI_OGLEXPERT_DETAIL_ALL, NVAPI_OGLEXPERT_DETAIL_BASIC_INFO, NVAPI_OGLEXPERT_OUTPUT_TO_ALL, 0);
#endif


  m_Camera = new vl::Camera;
  vl::vec3 eye    = vl::vec3(0,10,35);
  vl::vec3 center = vl::vec3(0,0,0);
  vl::vec3 up     = vl::vec3(0,1,0);
  vl::mat4 view_mat = vl::mat4::getLookAt(eye, center, up);
  m_Camera->setViewMatrix(view_mat);
  m_Camera->setObjectName("m_Camera");
  //m_Camera->viewport()->enableScissorSetup(true);
  //m_CameraTransform = new vl::Transform;
  //m_Camera->bindTransform(m_CameraTransform.get());

  vl::vec3    cameraPos = m_Camera->modelingMatrix().getT();

  m_LightTr = new vl::Transform;

  m_Light = new vl::Light;
  m_Light->setAmbient(vl::fvec4(0.1f, 0.1f, 0.1f, 1.0f));
  m_Light->setDiffuse(vl::white);
  m_Light->bindTransform(m_LightTr.get());

  vl::vec4 lightPos;
  lightPos[0] = cameraPos[0];
  lightPos[1] = cameraPos[1];
  lightPos[2] = cameraPos[2];
  lightPos[3] = 0;
  m_Light->setPosition(lightPos);


  m_SceneManager = new vl::SceneManagerActorTree;

  m_BackgroundCamera = new vl::Camera;
  m_BackgroundCamera->setObjectName("m_BackgroundCamera");
  // a simple "identity" ortho projection. it maps the background geometry perfectly into the 4 window corners.
  // actually, clipspace corners. mapping to pixels is done with the viewport, which is dependent on widget sizing.
  m_BackgroundCamera->setProjectionMatrix(vl::mat4::getOrtho(-1, 1, -1, 1, 1000, -1000), vl::PMT_OrthographicProjection);
  m_BackgroundCamera->setViewMatrix(vl::mat4::getIdentity());

  m_BackgroundRendering = new vl::Rendering;
  m_BackgroundRendering->setEnableMask(ENABLEMASK_BACKGROUND);
  m_BackgroundRendering->setObjectName("m_BackgroundRendering");
  m_BackgroundRendering->setCamera(m_BackgroundCamera.get());
  m_BackgroundRendering->sceneManagers()->push_back(m_SceneManager.get());
  m_BackgroundRendering->setCullingEnabled(false);
  m_BackgroundRendering->renderer()->setClearFlags(vl::CF_CLEAR_COLOR_DEPTH);   // this overrides the per-viewport setting (always!)
  m_BackgroundCamera->viewport()->setClearColor(vl::fuchsia);
  m_BackgroundCamera->viewport()->enableScissorSetup(false);

  // opaque objects dont need any sorting (in theory).
  // but they have to happen before anything else.
  m_OpaqueObjectsRendering = new vl::Rendering;
  m_OpaqueObjectsRendering->setEnableMask(ENABLEMASK_OPAQUE | ENABLEMASK_TRANSLUCENT | ENABLEMASK_SORTEDTRANSLUCENT);
  m_OpaqueObjectsRendering->setObjectName("m_OpaqueObjectsRendering");
  m_OpaqueObjectsRendering->setCamera(m_Camera.get());
  m_OpaqueObjectsRendering->sceneManagers()->push_back(m_SceneManager.get());
  m_OpaqueObjectsRendering->setCullingEnabled(true);
  // we sort them anyway, front-to-back so that early-fragment rejection can work its magic.
  m_OpaqueObjectsRendering->setRenderQueueSorter(new vl::RenderQueueSorterAggressive);
  // dont trash earlier stages.
  m_OpaqueObjectsRendering->renderer()->setClearFlags(vl::CF_CLEAR_DEPTH);

  // volume rendering is a separate stage, after opaque.
  // it needs access to the depth-buffer of the opaque geometry so that raycast can clip properly.
  m_VolumeRendering = new vl::Rendering;
  m_VolumeRendering->setEnableMask(ENABLEMASK_VOLUME);
  m_VolumeRendering->setObjectName("m_VolumeRendering");
  m_VolumeRendering->setCamera(m_Camera.get());
  m_VolumeRendering->sceneManagers()->push_back(m_SceneManager.get());
  m_VolumeRendering->setCullingEnabled(true);
  m_VolumeRendering->renderer()->setClearFlags(vl::CF_DO_NOT_CLEAR);
  // FIXME: only single volume supported for now, so no queue sorting.

  m_RenderingTree = new vl::RenderingTree;
  m_RenderingTree->setObjectName("m_RenderingTree");
  m_RenderingTree->subRenderings()->push_back(m_BackgroundRendering.get());
  m_RenderingTree->subRenderings()->push_back(m_OpaqueObjectsRendering.get());
  m_RenderingTree->subRenderings()->push_back(m_VolumeRendering.get());

  // once rendering to fbo has finished, blit it to the screen's backbuffer.
  // a final swapbuffers in renderScene() and/or paintGL() will show it on screen.
  m_FinalBlit = new vl::BlitFramebuffer;
  m_FinalBlit->setObjectName("m_FinalBlit");
  m_FinalBlit->setLinearFilteringEnabled(false);
  m_FinalBlit->setBufferMask(vl::BB_COLOR_BUFFER_BIT | vl::BB_DEPTH_BUFFER_BIT);
  m_FinalBlit->setDrawFramebuffer(vl::OpenGLContext::framebuffer());
  m_RenderingTree->onFinishedCallbacks()->push_back(m_FinalBlit.get());

  // updating the size of our fbo is a bit of a pain.
  CreateAndUpdateFBOSizes(QGLWidget::width(), QGLWidget::height());

  // moves the light with the main camera.
  // FIXME: attaching this to the rendering looks wrong
  m_OpaqueObjectsRendering->transform()->addChild(m_LightTr.get());

  // trackball is active by default because we do not yet have any camera-tracking data.
  EnableTrackballManipulator(true);

  m_ThresholdVal = new vl::Uniform("val_threshold");
  m_ThresholdVal->setUniformF(0.5f);

  m_GenericGLSLShader = new vl::GLSLProgram;
  m_GenericGLSLShader->attachShader(new vl::GLSLVertexShader(LoadGLSLSourceFromResources("generic.vs")));
  m_GenericGLSLShader->attachShader(new vl::GLSLFragmentShader(LoadGLSLSourceFromResources("generic.fs")));
  bool linkvalid = m_GenericGLSLShader->linkProgram(true);
  if (!linkvalid)
  {
    MITK_ERROR << "Shader didnt link: \n" << m_GenericGLSLShader->infoLog().toStdString();
  }
  bool shadervalid = m_GenericGLSLShader->validateProgram();
  if (!shadervalid)
  {
    MITK_ERROR << "Shader didnt validate: \n" << m_GenericGLSLShader->infoLog().toStdString();
  }

  m_DefaultTextureParams = new vl::TexParameter;
  m_DefaultTextureParams->setMinFilter(vl::TPF_LINEAR);
  m_DefaultTextureParams->setMagFilter(vl::TPF_LINEAR);
  m_DefaultTextureParams->setWrapS(vl::TPW_CLAMP_TO_BORDER);
  m_DefaultTextureParams->setWrapT(vl::TPW_CLAMP_TO_BORDER);
  m_DefaultTextureParams->setWrapR(vl::TPW_CLAMP_TO_BORDER);
  m_DefaultTextureParams->setBorderColor(vl::fvec4(0, 0, 0, 0));

  unsigned int      defaultTextureValue = 0x00000000;
  vl::ref<vl::Image>   tempImg = new vl::Image(1, 1, 0, 1, vl::IF_RGBA, vl::IT_UNSIGNED_BYTE);
  tempImg->allocate();
  *((unsigned int*) tempImg->pixels()) = defaultTextureValue;
  m_DefaultTexture = new vl::Texture(1, 1, vl::TF_RGBA, false);
  m_DefaultTexture->setMipLevel(0, tempImg.get(), false);


  vl::OpenGLContext::dispatchInitEvent();

#if 0
  // debugging
  mitk::DataNode::Pointer   n = mitk::DataNode::New();
  mitk::PCLData::Pointer    p = niftk::PCLData::New();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  c(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i = 0; i < 100; ++i)
  {
    pcl::PointXYZRGB  q(std::rand() % 255, std::rand() % 255, std::rand() % 255);
    q.x = std::rand() % 255;
    q.y = std::rand() % 255;
    q.z = std::rand() % 255;
    c->push_back(q);
  }
  p->SetCloud(c);
  n->SetData(p);

  m_DataStorage->Add(n);
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::SetBackgroundColour(float r, float g, float b)
{
  if (m_BackgroundCamera)
    m_BackgroundCamera->viewport()->setClearColor(vl::fvec4(r, g, b, 1));
  else
    QMetaObject::invokeMethod(this, "SetBackgroundColour", Qt::QueuedConnection, Q_ARG(float, r), Q_ARG(float, g), Q_ARG(float, b));
}


//-----------------------------------------------------------------------------
void VLQtWidget::EnableTrackballManipulator(bool enable)
{
  if (enable)
  {
    if (m_Trackball.get() == 0)
    {
      m_Trackball = new TrackballManipulator;
      m_Trackball->setEnabled(true);
      m_Trackball->setCamera(m_Camera.get());
      //m_Trackball->setTransform(m_CameraTransform.get());
      m_Trackball->setPivot(vl::vec3(0,0,0));
      vl::OpenGLContext::addEventListener(m_Trackball.get());
    }
  }
  else
  {
    if (m_Trackball.get() != 0)
    {
      vl::OpenGLContext::removeEventListener(m_Trackball.get());
      m_Trackball->setTransform(0);
      m_Trackball->setCamera(0);
      m_Trackball = 0;
    }
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::CreateAndUpdateFBOSizes(int width, int height)
{
  // sanity check: internal method, context should have been activated by caller.
  assert(this->context() == QGLContext::currentContext());

  // sanitise dimensions. depending on how windows are resized we can get zero here.
  // but that breaks on the ogl side.
  width  = std::max(1, width);
  height = std::max(1, height);

  vl::ref<vl::FramebufferObject> opaqueFBO = vl::OpenGLContext::createFramebufferObject(width, height);
  opaqueFBO->setObjectName("opaqueFBO");
  opaqueFBO->addDepthAttachment(new vl::FBODepthBufferAttachment(vl::DBF_DEPTH_COMPONENT24));
  opaqueFBO->addColorAttachment(vl::AP_COLOR_ATTACHMENT0, new vl::FBOColorBufferAttachment(vl::CBF_RGBA));   // this is a renderbuffer
  opaqueFBO->setDrawBuffer(vl::RDB_COLOR_ATTACHMENT0);

  m_BackgroundRendering->renderer()->setFramebuffer(opaqueFBO.get());
  m_OpaqueObjectsRendering->renderer()->setFramebuffer(opaqueFBO.get());
  m_VolumeRendering->renderer()->setFramebuffer(opaqueFBO.get());

  m_FinalBlit->setReadFramebuffer(opaqueFBO.get());
  m_FinalBlit->setReadBuffer(vl::RDB_COLOR_ATTACHMENT0);

#ifdef _USE_CUDA
  if (m_CUDAInteropPimpl)
  {
    delete m_CUDAInteropPimpl->m_FBOAdaptor;
    m_CUDAInteropPimpl->m_FBOAdaptor = new VLFramebufferAdaptor(opaqueFBO.get());
  }
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::resizeGL(int width, int height)
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());


  // dont do anything if window is zero size.
  // it's an opengl error to have a viewport like that!
  if ((width <= 0) || (height <= 0))
    return;

  // no idea if this is necessary...
  framebuffer()->setWidth(width);
  framebuffer()->setHeight(height);
  m_OpaqueObjectsRendering->renderer()->framebuffer()->setWidth(width);
  m_OpaqueObjectsRendering->renderer()->framebuffer()->setHeight(height);
  m_VolumeRendering->renderer()->framebuffer()->setWidth(width);
  m_VolumeRendering->renderer()->framebuffer()->setHeight(height);

  CreateAndUpdateFBOSizes(width, height);

  m_FinalBlit->setSrcRect(0, 0, width, height);
  m_FinalBlit->setDstRect(0, 0, width, height);

  UpdateViewportAndCameraAfterResize();

  vl::OpenGLContext::dispatchResizeEvent(width, height);
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateViewportAndCameraAfterResize()
{
  // some sane defaults
  m_Camera->viewport()->set(0, 0, QWidget::width(), QWidget::height());
  m_BackgroundCamera->viewport()->set(0, 0, QWidget::width(), QWidget::height());

  if (m_BackgroundNode.IsNotNull())
  {
    std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::iterator ni = m_NodeToActorMap.find(m_BackgroundNode);
    if (ni == m_NodeToActorMap.end())
    {
      // actor not ready yet, try again later.
      // this is getting messy... but stuffing our widget here into an editor causes various methods
      // to be called at the wrong time.
      QMetaObject::invokeMethod(this, "UpdateViewportAndCameraAfterResize", Qt::QueuedConnection);
    }
    else
    {
      vl::ref<vl::Actor> backgroundactor = ni->second;

      // this is based on my old araknes video-ar app.
      // FIXME: aspect ratio?
      float   width_scale  = (float) QWidget::width()  / (float) m_BackgroundWidth;
      float   height_scale = (float) QWidget::height() / (float) m_BackgroundHeight;
      int     vpw = QWidget::width();
      int     vph = QWidget::height();
      if (width_scale < height_scale)
        vph = (int) ((float) m_BackgroundHeight * width_scale);
      else
        vpw = (int) ((float) m_BackgroundWidth * height_scale);

      int   vpx = QWidget::width()  / 2 - vpw / 2;
      int   vpy = QWidget::height() / 2 - vph / 2;

      m_BackgroundCamera->viewport()->set(vpx, vpy, vpw, vph);
      // the main-scene-camera should conform to this viewport too!
      // otherwise geometry would never line up with the background (for overlays, etc).
      m_Camera->viewport()->set(vpx, vpy, vpw, vph);
    }
  }
  // this default perspective depends on the viewport!
  m_Camera->setProjectionPerspective();

  UpdateCameraParameters();
}


//-----------------------------------------------------------------------------
void VLQtWidget::paintGL()
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());

  RenderScene();

  vl::OpenGLContext::dispatchRunEvent();
}


//-----------------------------------------------------------------------------
void VLQtWidget::RenderScene()
{
  // caller of paintGL() (i.e. Qt's internals) should have activated our context!
  assert(this->context() == QGLContext::currentContext());

  // update vl-cache for nodes that have been modified since the last frame.
  for (std::set<mitk::DataNode::ConstPointer>::const_iterator i = m_NodesQueuedForUpdate.begin(); i != m_NodesQueuedForUpdate.end(); ++i)
  {
    UpdateDataNode(*i);
  }
  m_NodesQueuedForUpdate.clear();

  // UpdateTranslucentTriangles() is clever enough to do work only if necessary.
  UpdateTranslucentTriangles();


  // update scene graph.
  vl::mat4 cameraMatrix = m_Camera->modelingMatrix();
  // FIXME: light is lagging behind one frame
  m_LightTr->setLocalMatrix(cameraMatrix);


  // trigger execution of the renderer(s).
  vl::real now_time = vl::Time::currentTime();

  // simple fps stats
  {
    static int    counter = 0;
    ++counter;

    if ((counter % 100) == 0)
    {
      static vl::real prev = 0;

      std::cerr << "frame time: " << ((now_time - prev) / 10) << std::endl;
      prev = m_RenderingTree->frameClock();
    }
  }
  m_RenderingTree->setFrameClock(now_time);
  m_RenderingTree->render();

  if (vl::OpenGLContext::hasDoubleBuffer())
    swapBuffers();

  VL_CHECK_OGL();
}


//-----------------------------------------------------------------------------
void VLQtWidget::ClearScene()
{
  niftk::ScopedOGLContext  ctx(context());

  if (m_SceneManager)
  {
    if (m_SceneManager->tree())
      m_SceneManager->tree()->actors()->clear();
  }

  m_TranslucentActors.clear();
  m_TranslucentSurface = 0;
  m_TranslucentSurfaceActor = 0;

  m_BackgroundNode = 0;
  m_CameraNode = 0;

  m_NodeToActorMap.clear();
  m_ActorToRenderableMap.clear();
  m_NodesQueuedForUpdate.clear();
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateThresholdVal(int isoVal)
{
  niftk::ScopedOGLContext    ctx(context());

  float val_threshold = 0.0f;
  m_ThresholdVal->getUniform(&val_threshold);

  val_threshold = isoVal / 10000.0f;
  val_threshold = vl::clamp(val_threshold, 0.0f, 1.0f);

  m_ThresholdVal->setUniformF(val_threshold);
}


//-----------------------------------------------------------------------------
bool VLQtWidget::SetCameraTrackingNode(const mitk::DataNode::ConstPointer& node)
{
  m_CameraNode = node;

  if (m_CameraNode.IsNotNull())
  {
    EnableTrackballManipulator(false);
    UpdateCameraParameters();
  }
  else
    EnableTrackballManipulator(true);

  return true;
}


//-----------------------------------------------------------------------------
vl::mat4 VLQtWidget::GetVLMatrixFromData(const mitk::BaseData::ConstPointer& data)
{
  vl::mat4  mat;
  // intentionally not setIdentity()
  mat.setNull();

  if (data.IsNotNull())
  {
    mitk::BaseGeometry::Pointer   geom = data->GetGeometry();
    if (geom.IsNotNull())
    {
      if (geom->GetVtkTransform() != 0)
      {
        vtkSmartPointer<vtkMatrix4x4> vtkmat = vtkSmartPointer<vtkMatrix4x4>::New();
        geom->GetVtkTransform()->GetMatrix(vtkmat);
        if (vtkmat.GetPointer() != 0)
        {
          for (int i = 0; i < 4; i++)
          {
            for (int j = 0; j < 4; j++)
            {
              double val = vtkmat->GetElement(i, j);
              mat.e(i, j) = val;
            }
          }
        }
      }
    }
  }

  return mat;
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateTransformFromData(vl::ref<vl::Transform> txf, const mitk::BaseData::ConstPointer& data)
{
  vl::mat4  mat = GetVLMatrixFromData(data);

  if (!mat.isNull())
  {
    txf->setLocalMatrix(mat);
    txf->computeWorldMatrix();
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateActorTransformFromNode(vl::ref<vl::Actor> actor, const mitk::DataNode::ConstPointer& node)
{
  if (node.IsNotNull())
  {
    vl::ref<VLUserData>       userdata  = GetUserData(actor);
    mitk::BaseData::Pointer   data      = node->GetData();
    if (data.IsNotNull())
    {
      mitk::BaseGeometry::Pointer   geom = data->GetGeometry();
      if (geom.IsNotNull())
      {
        if (geom->GetMTime() > userdata->m_TransformLastModified)
        {
          UpdateTransformFromData(actor->transform(), data.GetPointer());
          userdata->m_TransformLastModified = geom->GetMTime();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
vl::ref<VLUserData> VLQtWidget::GetUserData(vl::ref<vl::Actor> actor)
{
  vl::ref<VLUserData>   userdata = actor->userData()->as<VLUserData>();
  if (userdata.get() == 0)
  {
    userdata = new VLUserData;
    actor->setUserData(userdata.get());
  }

  return userdata;
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateTransformFromNode(vl::ref<vl::Transform> txf, const mitk::DataNode::ConstPointer& node)
{
  if (node.IsNotNull())
  {
    UpdateTransformFromData(txf, node->GetData());
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateCameraParameters()
{
  // calibration parameters come from the background node.
  // so no background, no camera parameters.
  if (m_BackgroundNode.IsNotNull())
  {
#ifdef BUILD_IGI
    mitk::BaseProperty::Pointer       cambp = m_BackgroundNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
    if (cambp.IsNotNull())
    {
      mitk::CameraIntrinsicsProperty::Pointer cam = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cambp.GetPointer());
      if (cam.IsNotNull())
      {
        mitk::CameraIntrinsics::Pointer   nodeIntrinsic = cam->GetValue();

        if (nodeIntrinsic.IsNotNull())
        {
          // based on niftkCore/Rendering/vtkOpenGLMatrixDrivenCamera
          float   znear = 1;
          float   zfar  = 10000;
          float   pixelaspectratio = 1;   // FIXME: depends on background image

          vl::mat4  proj;
          proj.setNull();
          proj.e(0, 0) =  2 * nodeIntrinsic->GetFocalLengthX() / (float) m_BackgroundWidth;
          //proj.e(0, 1) = -2 * 0 / m_ImageWidthInPixels;
          proj.e(0, 2) = ((float) m_BackgroundWidth - 2 * nodeIntrinsic->GetPrincipalPointX()) / (float) m_BackgroundWidth;
          proj.e(1, 1) = 2 * (nodeIntrinsic->GetFocalLengthY() / pixelaspectratio) / ((float) m_BackgroundHeight / pixelaspectratio);
          proj.e(1, 2) = (-((float) m_BackgroundHeight / pixelaspectratio) + 2 * (nodeIntrinsic->GetPrincipalPointY() / pixelaspectratio)) / ((float) m_BackgroundHeight / pixelaspectratio);
          proj.e(2, 2) = (-zfar - znear) / (zfar - znear);
          proj.e(2, 3) = -2 * zfar * znear / (zfar - znear);
          proj.e(3, 2) = -1;

          m_Camera->setProjectionMatrix(proj.transpose(), vl::PMT_UserProjection);
        }
      }
    }
#endif
  }

  if (m_CameraNode.IsNotNull())
  {
    vl::mat4  mat = GetVLMatrixFromData(m_CameraNode->GetData());
    if (!mat.isNull())
      // beware: there is also a view-matrix! the inverse of modelling-matrix.
      m_Camera->setModelingMatrix(mat);
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::PrepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  // nasty
  mitk::Image::Pointer  imgp(const_cast<mitk::Image*>(img));
  vl::ref<vl::Actor>  actor = Add2DImageActor(imgp);


  // essentially copied from vl::makeGrid()
  vl::ref<vl::Geometry>         vlquad = new vl::Geometry;

  vl::ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  vl::ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  vl::ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls()->push_back(polys.get());

  // replace original quad with ours.
  actor->setLod(0, vlquad.get());
  actor->effect()->shader()->disable(vl::EN_LIGHTING);

  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeToActorMap[node] = actor;
}


//-----------------------------------------------------------------------------
void VLQtWidget::PrepareBackgroundActor(const niftk::LightweightCUDAImage* lwci, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_CUDA
  assert(lwci != 0);

  vl::mat4  mat;
  mat = mat.setIdentity();
  vl::ref<vl::Transform> tr     = new vl::Transform();
  tr->setLocalMatrix(mat);


  // essentially copied from vl::makeGrid()
  vl::ref<vl::Geometry>         vlquad = new vl::Geometry;

  vl::ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  vl::ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  vl::ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls()->push_back(polys.get());


  vl::ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  // UpdateDataNode() takes care of assigning colour etc.

  vl::ref<vl::Actor>    actor = m_SceneManager->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[actor] = vlquad;


  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeToActorMap[node] = actor;
  m_NodeToTextureMap[node] = TextureDataPOD();

#else
  throw std::runtime_error("No CUDA support enabled at compile time");
#endif
}


//-----------------------------------------------------------------------------
bool VLQtWidget::SetBackgroundNode(const mitk::DataNode::ConstPointer& node)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  niftk::ScopedOGLContext    ctx(context());

  // clear up after previous background node.
  if (m_BackgroundNode.IsNotNull())
  {
    const mitk::DataNode::ConstPointer    oldbackgroundnode = m_BackgroundNode;
    m_BackgroundNode = 0;
    RemoveDataNode(oldbackgroundnode);
    // add back as normal node.
    AddDataNode(oldbackgroundnode);
  }

  // default "no background" value.
  m_BackgroundWidth  = 0;
  m_BackgroundHeight = 0;

  bool    result = false;
  mitk::BaseData::Pointer   basedata;
  if (node.IsNotNull())
    basedata = node->GetData();
  if (basedata.IsNotNull())
  {
    // clear up whatever we had cached for the new background node.
    // it's very likely that it was a normal node before.
    RemoveDataNode(node);

    mitk::Image::Pointer      imgdata = dynamic_cast<mitk::Image*>(basedata.GetPointer());
    if (imgdata.IsNotNull())
    {
#ifdef _USE_CUDA
      niftk::CUDAImageProperty::Pointer    cudaimgprop = dynamic_cast<niftk::CUDAImageProperty*>(imgdata->GetProperty("CUDAImageProperty").GetPointer());
      if (cudaimgprop.IsNotNull())
      {
        niftk::LightweightCUDAImage    lwci = cudaimgprop->Get();

        // does the size of cuda-image have to match the mitk-image where it's attached to?
        // i think it does: it is supposed to be the same data living in cuda.
        assert(lwci.GetWidth()  == imgdata->GetDimension(0));
        assert(lwci.GetHeight() == imgdata->GetDimension(1));

        PrepareBackgroundActor(&lwci, imgdata->GetGeometry(), node);
        result = true;
      }
      else
#endif
      {
        PrepareBackgroundActor(imgdata.GetPointer(), imgdata->GetGeometry(), node);
        result = true;
      }

      m_BackgroundWidth  = imgdata->GetDimension(0);
      m_BackgroundHeight = imgdata->GetDimension(1);
    }
    else
    {
#ifdef _USE_CUDA
      niftk::CUDAImage::Pointer    cudaimgdata = dynamic_cast<niftk::CUDAImage*>(basedata.GetPointer());
      if (cudaimgdata.IsNotNull())
      {
        niftk::LightweightCUDAImage    lwci = cudaimgdata->GetLightweightCUDAImage();
        PrepareBackgroundActor(&lwci, cudaimgdata->GetGeometry(), node);
        result = true;

        m_BackgroundWidth  = lwci.GetWidth();
        m_BackgroundHeight = lwci.GetHeight();
      }
      // no else here
#endif
    }

    // UpdateDataNode() depends on m_BackgroundNode.
    m_BackgroundNode = node;
    UpdateDataNode(node);
  }


  UpdateViewportAndCameraAfterResize();

  // now that the camera may have changed, fit-view-to-scene again.
  if (m_CameraNode.IsNull())
  {
    if (m_Trackball.get() != 0)
      m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);
  }


  return result;
}


//-----------------------------------------------------------------------------
void VLQtWidget::AddAllNodesFromDataStorage()
{
  if (m_DataStorage.IsNull())
    return;

  typedef itk::VectorContainer<unsigned int, mitk::DataNode::Pointer>   NodesContainerType;
  NodesContainerType::ConstPointer vc = m_DataStorage->GetAll();

  for (unsigned int i = 0; i < vc->Size(); ++i)
  {
    mitk::DataNode::Pointer currentDataNode = vc->ElementAt(i);
    if (currentDataNode.IsNull() || currentDataNode->GetData()== 0)
      continue;

    AddDataNode(mitk::DataNode::ConstPointer(currentDataNode.GetPointer()));
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::AddDataNode(const mitk::DataNode::ConstPointer& node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  // only add node once.
  if (m_NodeToActorMap.find(node) != m_NodeToActorMap.end())
    return;

  // Propagate color and opacity down to basedata
  node->GetData()->SetProperty("color", node->GetProperty("color"));
  node->GetData()->SetProperty("opacity", node->GetProperty("opacity"));
  node->GetData()->SetProperty("visible", node->GetProperty("visible"));

  bool                    doMitkImageIfSuitable = true;
  mitk::Image::Pointer    mitkImg   = dynamic_cast<mitk::Image*>(node->GetData());
  mitk::Surface::Pointer  mitkSurf  = dynamic_cast<mitk::Surface*>(node->GetData());
  mitk::PointSet::Pointer mitkPS    = dynamic_cast<mitk::PointSet*>(node->GetData());
#ifdef _USE_PCL
  niftk::PCLData::Pointer  pclPS     = dynamic_cast<niftk::PCLData*>(node->GetData());
#endif
  mitk::CoordinateAxesData::Pointer   coords = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
#ifdef _USE_CUDA
  mitk::BaseData::Pointer cudaImg   = dynamic_cast<niftk::CUDAImage*>(node->GetData());
  // this check will prefer a CUDAImageProperty attached to the node's data object.
  // e.g. if there is mitk::Image and an attached CUDAImageProperty then CUDAImageProperty wins and
  // mitk::Image is ignored.
  doMitkImageIfSuitable = !(dynamic_cast<niftk::CUDAImageProperty*>(node->GetData()->GetProperty("CUDAImageProperty").GetPointer()) != 0);
  if (doMitkImageIfSuitable == false)
  {
    cudaImg = node->GetData();
  }
#endif


  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  niftk::ScopedOGLContext    ctx(context());


  vl::ref<vl::Actor>    newActor;
  std::string           namePostFix;
  if (mitkImg.IsNotNull() && doMitkImageIfSuitable)
  {
    newActor = AddImageActor(mitkImg);
    namePostFix = "_image";
  }
  else
  if (mitkSurf.IsNotNull())
  {
    newActor = AddSurfaceActor(mitkSurf);
    namePostFix = "_surface";
  }
  else
  if (mitkPS.IsNotNull())
  {
    newActor = AddPointsetActor(mitkPS);
    namePostFix = "_pointset";
  }
  else
  if (coords.IsNotNull())
  {
    newActor = AddCoordinateAxisActor(coords);
    namePostFix = "_coordinateaxisdata";
  }
#ifdef _USE_PCL
  else
  if (pclPS.IsNotNull())
  {
    newActor = AddPointCloudActor(pclPS);
    namePostFix = "_pcl";
  }
#endif
#ifdef _USE_CUDA
  else
  if (cudaImg.IsNotNull())
  {
    newActor = AddCUDAImageActor(cudaImg);
    namePostFix = "_cudaimage";

    m_NodeToTextureMap[node] = TextureDataPOD();
  }
#endif

  if (newActor.get() != 0)// && sceneManager()->tree()->actors()->find(newActor.get()) == -1)
  {
    std::string   objName = newActor->objectName();
    objName.append(namePostFix);
    newActor->setObjectName(objName.c_str());

    m_NodeToActorMap[node] = newActor;

    // update colour, etc.
    UpdateDataNode(node);
  }

  if (m_CameraNode.IsNull())
  {
    if (m_Trackball.get() != 0)
      m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::QueueUpdateDataNode(const mitk::DataNode::ConstPointer& node)
{
  m_NodesQueuedForUpdate.insert(node);
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateDataNode(const mitk::DataNode::ConstPointer& node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::iterator     it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;

  vl::ref<vl::Actor>    vlActor = it->second;
  if (vlActor.get() == 0)
    return;


  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  niftk::ScopedOGLContext    ctx(context());

  bool  isVisble = true;
  mitk::BoolProperty* visibleProp = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible"));
  if (visibleProp != 0)
    isVisble = visibleProp->GetValue();

  float opacity = 1.0f;
  mitk::FloatProperty* opacityProp = dynamic_cast<mitk::FloatProperty*>(node->GetProperty("opacity"));
  if (opacityProp != 0)
    opacity = opacityProp->GetValue();
  // if object is too translucent to not have a effect after blending then just skip it.
  if (opacity < (1.0f / 255.0f))
    isVisble = false;

  if (isVisble == false)
  {
    vlActor->setEnableMask(0);
  }
  else
  {
    vlActor->setEnableMask(0xFFFFFFFF);

    vl::fvec4 color(1, 1, 1, opacity);
    mitk::ColorProperty* colorProp = dynamic_cast<mitk::ColorProperty*>(node->GetProperty("color"));
    if (colorProp != 0)
    {
      mitk::Color mitkColor = colorProp->GetColor();
      color[0] = mitkColor[0];
      color[1] = mitkColor[1];
      color[2] = mitkColor[2];
    }

    vl::ref<vl::Effect> fx = vlActor->effect();
    fx->shader()->enable(vl::EN_DEPTH_TEST);
    fx->shader()->setRenderState(m_Light.get(), 0);
    fx->shader()->gocMaterial()->setDiffuse(vl::white);   // normal shading is done via tint colour.
    fx->shader()->gocRenderStateSet()->setRenderState(m_GenericGLSLShader.get(), -1);

    // the uniform tint colour is defined on the actor, instead of shader or effect.
    vlActor->gocUniformSet()->gocUniform("u_TintColour")->setUniform4f(1, color.ptr());

    // shader still needs to know whether to apply light-shading or not.
    float   disableLighting = fx->shader()->isEnabled(vl::EN_LIGHTING) ? 0.0f : 1.0f;
    vlActor->gocUniformSet()->gocUniform("u_DisableLighting")->setUniform1f(1, &disableLighting);

    // by convention, all meaningful texture maps are bound in slot 0.
    // slot 1 will have the default-empty-dummy.
    if (fx->shader()->getTextureSampler(0) != 0)
      vlActor->gocUniformSet()->gocUniform("u_TextureMap")->setUniformI(0);
    else
      vlActor->gocUniformSet()->gocUniform("u_TextureMap")->setUniformI(1);


    float   pointsize = 1;
    bool    haspointsize = node->GetFloatProperty("pointsize", pointsize);
    if (haspointsize)
    {
      vl::PointSize* ps = fx->shader()->getPointSize();
      if (ps != 0)
      {
        if (ps->pointSize() != pointsize)
          ps = 0;
      }

      if (ps == 0)
      {
        fx->shader()->setRenderState(new vl::PointSize(pointsize));
        if (pointsize > 1)
          fx->shader()->enable(vl::EN_POINT_SMOOTH);
        else
          fx->shader()->disable(vl::EN_POINT_SMOOTH);
      }
    }


    bool  isVolume = false;
    // special case for volumes: they'll have a certain event-callback bound.
    for (int i = 0; i < vlActor->actorEventCallbacks()->size(); ++i)
    {
      isVolume |= (dynamic_cast<vl::RaycastVolume*>(vlActor->actorEventCallbacks()->at(i)) != 0);
      if (isVolume)
        break;
    }

    if (isVolume)
    {
      vlActor->setRenderBlock(RENDERBLOCK_TRANSLUCENT);
      vlActor->setEnableMask(ENABLEMASK_VOLUME);
      fx->shader()->enable(vl::EN_BLEND);
      fx->shader()->enable(vl::EN_CULL_FACE);
      fx->shader()->gocMaterial()->setTransparency(1.0f);//opacity);
    }
    else
    {
      bool  isProbablyTranslucent = opacity <= (1.0f - (1.0f / 255.0f));
      if (isProbablyTranslucent)
      {
        vlActor->setRenderBlock(RENDERBLOCK_TRANSLUCENT);
        vlActor->setEnableMask(ENABLEMASK_TRANSLUCENT);
        fx->shader()->enable(vl::EN_BLEND);
        // no backface culling for translucent objects: you should be able to see the backside!
        fx->shader()->disable(vl::EN_CULL_FACE);
      }
      else
      {
        vlActor->setRenderBlock(RENDERBLOCK_OPAQUE);
        vlActor->setEnableMask(ENABLEMASK_OPAQUE);
        fx->shader()->disable(vl::EN_BLEND);
        //fx->shader()->enable(vl::EN_CULL_FACE);
        fx->shader()->disable(vl::EN_CULL_FACE);
      }
    }
  }

  UpdateActorTransformFromNode(vlActor, node);

  // does the right thing if node does not contain an mitk-image.
  UpdateTextureFromImage(node);

  // if we do have live-updating textures then we do need to refresh the vl-side of it!
  // even if the node is not visible.
#ifdef _USE_CUDA
  UpdateGLTexturesFromCUDA(node);
#endif

  // background is always visible, even if its datanode is not.
  if (node == m_BackgroundNode)
  {
    assert(vlActor->objectName().find("_background") != std::string::npos);
    vlActor->setEnableMask(ENABLEMASK_BACKGROUND);
    vlActor->effect()->shader()->disable(vl::EN_DEPTH_TEST);
    vlActor->effect()->shader()->enable(vl::EN_BLEND);
    vlActor->effect()->shader()->disable(vl::EN_CULL_FACE);
  }

  if (node == m_CameraNode)
  {
    UpdateCameraParameters();
  }
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::FindActorForNode(const mitk::DataNode::ConstPointer& node)
{
  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::iterator     it = m_NodeToActorMap.find(node);
  if (it != m_NodeToActorMap.end())
  {
    return it->second;
  }

  return vl::ref<vl::Actor>();
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateTextureFromImage(const mitk::DataNode::ConstPointer& node)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  if (node.IsNotNull())
  {
    mitk::Image::Pointer    img = dynamic_cast<mitk::Image*>(node->GetData());
    if (img.IsNotNull())
    {
      vl::ref<vl::Actor>  vlactor = FindActorForNode(node);
      if (vlactor.get() != 0)
      {
        assert(vlactor->effect());
        assert(vlactor->effect()->shader());

        vl::ref<VLUserData>   userdata = GetUserData(vlactor);
        if (img->GetVtkImageData()->GetMTime() > userdata->m_ImageVtkDataLastModified)
        {
          vl::ref<vl::Texture> tex = vlactor->effect()->shader()->gocTextureSampler(0)->texture();
          if (tex.get() != 0)
          {
            unsigned int*       dims    = img->GetDimensions();    // we do not own dims!
            mitk::PixelType     pixType = img->GetPixelType();
            vl::EImageType      type    = MapITKPixelTypeToVL(pixType.GetComponentType());
            vl::EImageFormat    format  = MapComponentsToVLColourFormat(pixType.GetNumberOfComponents());

            vl::ref<vl::Image>    vlimg = new vl::Image(dims[0], dims[1], 0, 1, format, type);
            // sanity check
            unsigned int  size = (dims[0] * dims[1] * dims[2]) * pixType.GetSize();
            assert(vlimg->requiredMemory() == size);

            try
            {
              mitk::ImageReadAccessor   readAccess(img);
              const void*               cPointer = readAccess.GetData();
              std::memcpy(vlimg->pixels(), cPointer, vlimg->requiredMemory());
            }
            catch (...)
            {
              // FIXME: error handling?
              MITK_ERROR << "Did not get pixel read access to 2D image.";
            }

            tex->setMipLevel(0, vlimg.get(), false);

            userdata->m_ImageVtkDataLastModified = img->GetVtkImageData()->GetMTime();
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateGLTexturesFromCUDA(const mitk::DataNode::ConstPointer& node)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_CUDA
  niftk::LightweightCUDAImage    lwcImage;

  niftk::CUDAImage::Pointer cudaimg = dynamic_cast<niftk::CUDAImage*>(node->GetData());
  if (cudaimg.IsNotNull())
  {
    lwcImage = cudaimg->GetLightweightCUDAImage();
  }
  else
  {
    mitk::Image::Pointer    img = dynamic_cast<mitk::Image*>(node->GetData());
    if (img.IsNotNull())
    {
      niftk::CUDAImageProperty::Pointer prop = dynamic_cast<niftk::CUDAImageProperty*>(img->GetProperty("CUDAImageProperty").GetPointer());
      if (prop.IsNotNull())
      {
        lwcImage = prop->Get();
      }
    }
  }

  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::iterator     it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;
  vl::ref<vl::Actor>    vlActor = it->second;
  if (vlActor.get() == 0)
    return;

  if (lwcImage.GetId() != 0)
  {
    // whatever we had cached from a previous frame.
    TextureDataPOD          texpod    = m_NodeToTextureMap[node];

    // only need to update the vl texture, if content in our cuda buffer has changed.
    // and the cuda buffer can change only when we have a different id.
    if (texpod.m_LastUpdatedID != lwcImage.GetId())
    {
      cudaError_t   err = cudaSuccess;
      bool          neednewvltexture = texpod.m_Texture.get() == 0;

      // check if vl-texture size needs to change
      if (texpod.m_Texture.get() != 0)
      {
        neednewvltexture |= lwcImage.GetWidth()  != texpod.m_Texture->width();
        neednewvltexture |= lwcImage.GetHeight() != texpod.m_Texture->height();
      }

      if (neednewvltexture)
      {
        if (texpod.m_CUDARes)
        {
          err = cudaGraphicsUnregisterResource(texpod.m_CUDARes);
          texpod.m_CUDARes = 0;
          if (err != cudaSuccess)
          {
            MITK_WARN << "Could not unregister VL texture from CUDA. This will likely leak GPU memory.";
          }
        }

        texpod.m_Texture = new vl::Texture(lwcImage.GetWidth(), lwcImage.GetHeight(), vl::TF_RGBA8, false);
        vlActor->effect()->shader()->gocTextureSampler(0)->setTexture(texpod.m_Texture.get());
        vlActor->effect()->shader()->gocTextureSampler(0)->setTexParameter(m_DefaultTextureParams.get());

        err = cudaGraphicsGLRegisterImage(&texpod.m_CUDARes, texpod.m_Texture->handle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess)
        {
          texpod.m_CUDARes = 0;
          MITK_WARN << "Registering VL texture into CUDA failed. Will not update (properly).";
        }
      }

      if (texpod.m_CUDARes)
      {
        assert(vlActor->effect()->shader()->getTextureSampler(0)->texture() == texpod.m_Texture);

        niftk::CUDAManager*  cudamng   = niftk::CUDAManager::GetInstance();
        cudaStream_t         mystream  = cudamng->GetStream("VLQtWidget vl-texture update");
        niftk::ReadAccessor  inputRA   = cudamng->RequestReadAccess(lwcImage);

        // make sure producer of the cuda-image finished.
        err = cudaStreamWaitEvent(mystream, inputRA.m_ReadyEvent, 0);
        if (err != cudaSuccess)
        {
          // flood the log
          MITK_WARN << "cudaStreamWaitEvent failed with error code " << err;
        }

        // this also guarantees that ogl will have finished doing its thing before mystream starts copying.
        err = cudaGraphicsMapResources(1, &texpod.m_CUDARes, mystream);
        if (err == cudaSuccess)
        {
          // normally we would need to flip image! ogl is left-bottom, whereas everywhere else is left-top origin.
          // but texture coordinates that we have assigned to the quads rendering the current image will do that for us.

          cudaArray_t   arr = 0;
          err = cudaGraphicsSubResourceGetMappedArray(&arr, texpod.m_CUDARes, 0, 0);
          if (err == cudaSuccess)
          {
            err = cudaMemcpy2DToArrayAsync(arr, 0, 0, inputRA.m_DevicePointer, inputRA.m_BytePitch, lwcImage.GetWidth() * 4, lwcImage.GetHeight(), cudaMemcpyDeviceToDevice, mystream);
            if (err == cudaSuccess)
            {
              texpod.m_LastUpdatedID = lwcImage.GetId();
            }
          }

          err = cudaGraphicsUnmapResources(1, &texpod.m_CUDARes, mystream);
          if (err != cudaSuccess)
          {
            MITK_WARN << "Cannot unmap VL texture from CUDA. This will probably kill the renderer. Error code: " << err;
          }
        }
        // make sure Autorelease() and Finalise() are always the last things to do for a stream!
        // otherwise the streamcallback will block subsequent work.
        // in this case here, the callback managed by CUDAManager that keeps track of refcounts could stall
        // the opengl driver if cudaGraphicsUnmapResources() came after Autorelease().
        cudamng->Autorelease(inputRA, mystream);
      }

      // update cache, even if something went wrong.
      m_NodeToTextureMap[node] = texpod;

      // helps with debugging
      vlActor->effect()->shader()->disable(vl::EN_CULL_FACE);
    }
  }
#else
  throw std::runtime_error("No CUDA-support enabled at compile time!");
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::RemoveDataNode(const mitk::DataNode::ConstPointer& node)
{
  // dont leave a dangling update behind.
  m_NodesQueuedForUpdate.erase(node);

  if (node.IsNull() || node->GetData() == 0)
    return;

  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  niftk::ScopedOGLContext    ctx(context());

  // recompute the big-fat-translucent-triangle-buffer.
  m_TranslucentStructuresMerged = false;

#ifdef _USE_CUDA
  {
    std::map<mitk::DataNode::ConstPointer, TextureDataPOD>::iterator i = m_NodeToTextureMap.find(node);
    if (i != m_NodeToTextureMap.end())
    {
      if (i->second.m_CUDARes != 0)
      {
        cudaError_t err = cudaGraphicsUnregisterResource(i->second.m_CUDARes);
        if (err != cudaSuccess)
        {
          MITK_WARN << "Failed to unregister VL texture from CUDA";
        }
      }

      m_NodeToTextureMap.erase(i);
    }
  }
#endif

  std::map<mitk::DataNode::ConstPointer, vl::ref<vl::Actor> >::iterator    it = m_NodeToActorMap.find(node);
  if (it != m_NodeToActorMap.end())
  {
    vl::ref<vl::Actor>    vlActor = it->second;
    if (vlActor.get() != 0)
    {
      m_ActorToRenderableMap.erase(vlActor);
      m_SceneManager->tree()->eraseActor(vlActor.get());
      m_NodeToActorMap.erase(it);
    }
  }
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddCoordinateAxisActor(const mitk::CoordinateAxesData::Pointer& coord)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, coord.GetPointer());

  vl::ref<vl::ArrayFloat3>      vlVerts  = new vl::ArrayFloat3;
  vl::ref<vl::ArrayFloat4>      vlColors = new vl::ArrayFloat4;
  vlVerts->resize(4);
  vlColors->resize(4);

  // x y z r g b a
  vlVerts->at(0).x() =  0;   vlVerts->at(0).y() =  0;   vlVerts->at(0).z() =  0;   vlColors->at(0).r() = 0;  vlColors->at(0).g() = 0;  vlColors->at(0).b() = 0;  vlColors->at(0).a() = 1;
  vlVerts->at(1).x() = 10;   vlVerts->at(1).y() =  0;   vlVerts->at(1).z() =  0;   vlColors->at(1).r() = 1;  vlColors->at(1).g() = 0;  vlColors->at(1).b() = 0;  vlColors->at(1).a() = 1;
  vlVerts->at(2).x() =  0;   vlVerts->at(2).y() = 10;   vlVerts->at(2).z() =  0;   vlColors->at(2).r() = 0;  vlColors->at(2).g() = 1;  vlColors->at(2).b() = 0;  vlColors->at(2).a() = 1;
  vlVerts->at(3).x() =  0;   vlVerts->at(3).y() =  0;   vlVerts->at(3).z() = 10;   vlColors->at(3).r() = 0;  vlColors->at(3).g() = 0;  vlColors->at(3).b() = 1;  vlColors->at(2).a() = 1;


  vl::ref<vl::DrawElementsUInt>   lines = new vl::DrawElementsUInt(vl::PT_LINES);
  lines->indexBuffer()->resize(3 * 2);
  lines->indexBuffer()->at(0) = 0;  lines->indexBuffer()->at(1) = 1;      // x
  lines->indexBuffer()->at(2) = 0;  lines->indexBuffer()->at(3) = 2;      // y
  lines->indexBuffer()->at(4) = 0;  lines->indexBuffer()->at(5) = 3;      // z

  vl::ref<vl::Geometry>         vlGeom   = new vl::Geometry;
  vlGeom->drawCalls()->push_back(lines.get());
  vlGeom->setVertexArray(vlVerts.get());
  vlGeom->setColorArray(vlColors.get());

  vl::ref<vl::Effect>   fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  fx->shader()->setRenderState(new vl::ShadeModel(vl::SM_FLAT));    // important! otherwise colour is wrong.
  fx->shader()->setRenderState(new vl::LineWidth(5));               // arbitrary
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());


  vl::ref<vl::Actor>    actor = m_SceneManager->tree()->addActor(vlGeom.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[actor] = vlGeom;

  return actor;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddPointCloudActor(niftk::PCLData* pcl)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_PCL
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr   cloud = pcl->GetCloud();

  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, pcl);

  vl::ref<vl::ArrayFloat3>      vlVerts  = new vl::ArrayFloat3;
  vl::ref<vl::ArrayFloat4>      vlColors = new vl::ArrayFloat4;
  vlVerts->resize(cloud->size());
  vlColors->resize(cloud->size());
  int   j = 0;
  for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator i = cloud->begin(); i != cloud->end(); ++i, ++j)
  {
    const pcl::PointXYZRGB& p = *i;
    vlVerts->at(j).x() = p.x;
    vlVerts->at(j).y() = p.y;
    vlVerts->at(j).z() = p.z;
    // would be nice if we could interleave the vl arrays...
    vlColors->at(j).r() = (float)p.r / 255.0f;
    vlColors->at(j).g() = (float)p.g / 255.0f;
    vlColors->at(j).b() = (float)p.b / 255.0f;
    vlColors->at(j).a() = 1;
  }

  vl::ref<vl::DrawArrays>       vlPoints = new vl::DrawArrays(vl::PT_POINTS, 0, vlVerts->size());
  vl::ref<vl::Geometry>         vlGeom   = new vl::Geometry;
  vlGeom->drawCalls()->push_back(vlPoints.get());
  vlGeom->setVertexArray(vlVerts.get());
  vlGeom->setColorArray(vlColors.get());

  vl::ref<vl::Effect>   fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  // FIXME: currently nothing assigns a pointsize property for PCLData nodes. so set an arbitrary fixed size.
  fx->shader()->setRenderState(new vl::PointSize(5));
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());


  vl::ref<vl::Actor>    psActor = m_SceneManager->tree()->addActor(vlGeom.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[psActor] = vlGeom;

  return psActor;
#else
  throw std::runtime_error("No PCL-support enabled at compile time!");
#endif
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddPointsetActor(const mitk::PointSet::Pointer& mitkPS)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());


  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, mitkPS.GetPointer());

  vl::ref<vl::ArrayFloat3>      vlVerts  = new vl::ArrayFloat3;
  vlVerts->resize(mitkPS->GetSize());
  int   j = 0;
  for (mitk::PointSet::PointsConstIterator i = mitkPS->Begin(); i != mitkPS->End(); ++i, ++j)
  {
    mitk::PointSet::PointType p = i->Value();
    vlVerts->at(j).x() = p[0];
    vlVerts->at(j).y() = p[1];
    vlVerts->at(j).z() = p[2];
  }

  vl::ref<vl::DrawArrays>       vlPoints = new vl::DrawArrays(vl::PT_POINTS, 0, vlVerts->size());
  vl::ref<vl::Geometry>         vlGeom   = new vl::Geometry;
  vlGeom->drawCalls()->push_back(vlPoints.get());
  vlGeom->setVertexArray(vlVerts.get());

  vl::ref<vl::Effect>   fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());


  vl::ref<vl::Actor>    psActor = m_SceneManager->tree()->addActor(vlGeom.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[psActor] = vlGeom;

  return psActor;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddSurfaceActor(const mitk::Surface::Pointer& mitkSurf)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());


  vl::ref<vl::Geometry>  vlSurf = new vl::Geometry();
  ConvertVTKPolyData(mitkSurf->GetVtkPolyData(), vlSurf);

  //MITK_INFO <<"Num of vertices: " << vlSurf->vertexArray()->size()/3;
  //ArrayAbstract* posarr = vertexArray() ? vertexArray() : vertexAttribArray(vl::VA_Position) ? vertexAttribArray(vl::VA_Position)->data() : NULL;
  if (!vlSurf->normalArray())
    vlSurf->computeNormals();

  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, mitkSurf.GetPointer());

  vl::ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->enable(vl::EN_LIGHTING);
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());
  // UpdateDataNode() takes care of assigning colour etc.

  vl::ref<vl::Actor>    surfActor = m_SceneManager->tree()->addActor(vlSurf.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[surfActor] = vlSurf;

  return surfActor;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Geometry> VLQtWidget::CreateGeometryFor2DImage(int width, int height)
{
  vl::ref<vl::Geometry>         vlquad = new vl::Geometry;
  vl::ref<vl::ArrayFloat3>      vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  vl::ref<vl::ArrayFloat2>      text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = 0;     vert3->at(0).y() = 0;      vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = 0;     vert3->at(1).y() = height; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() = width; vert3->at(2).y() = height; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() = width; vert3->at(3).y() = 0;      vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  vl::ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls()->push_back(polys.get());

  return vlquad;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddCUDAImageActor(const mitk::BaseData* _cudaImg)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_CUDA
  niftk::LightweightCUDAImage lwci;
  const niftk::CUDAImage* cudaImg = dynamic_cast<const niftk::CUDAImage*>(_cudaImg);
  if (cudaImg != 0)
  {
    lwci = cudaImg->GetLightweightCUDAImage();
  }
  else
  {
    niftk::CUDAImageProperty::Pointer prop = dynamic_cast<niftk::CUDAImageProperty*>(_cudaImg->GetProperty("CUDAImageProperty").GetPointer());
    if (prop.IsNotNull())
    {
      lwci = prop->Get();
    }
  }
  assert(lwci.GetId() != 0);

  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, cudaImg);

  vl::ref<vl::Geometry>         vlquad    = CreateGeometryFor2DImage(lwci.GetWidth(), lwci.GetHeight());

  vl::ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());
  // UpdateDataNode() takes care of assigning colour etc.

  vl::ref<vl::Actor>    actor = m_SceneManager->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[actor] = vlquad;

  return actor;

#else
  throw std::runtime_error("No CUDA-support enabled at compile time!");
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::ConvertVTKPolyData(vtkPolyData* vtkPoly, vl::ref<vl::Geometry> vlPoly)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  if (vtkPoly == 0)
    return;

  /// \brief Buffer in host memory to store cell info
  unsigned int * m_IndexBuffer = 0;
  
  /// \brief Buffer in host memory to store vertex points
  float * m_PointBuffer = 0;
  
  /// \brief Buffer in host memory to store normals associated with vertices
  float * m_NormalBuffer = 0;

  /// \brief Buffer in host memory to store scalar info associated with vertices
  char * m_ScalarBuffer = 0;


  unsigned int numOfvtkPolyPoints = vtkPoly->GetNumberOfPoints();

  // A polydata will always have point data
  int pointArrayNum = vtkPoly->GetPointData()->GetNumberOfArrays();

  if (pointArrayNum == 0 && numOfvtkPolyPoints == 0)
  {
    MITK_ERROR <<"No points detected in the vtkPoly data!\n";
    return;
  }

  // We'll have to build the cell data if not present already
  int cellArrayNum  = vtkPoly->GetCellData()->GetNumberOfArrays();
  if (cellArrayNum == 0)
    vtkPoly->BuildCells();

  vtkSmartPointer<vtkCellArray> verts;

  // Try to get access to cells
  if (vtkPoly->GetVerts() != 0 && vtkPoly->GetVerts()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetVerts();
  else if (vtkPoly->GetLines() != 0 && vtkPoly->GetLines()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetLines();
  else if (vtkPoly->GetPolys() != 0 && vtkPoly->GetPolys()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetPolys();
  else if (vtkPoly->GetStrips() != 0 && vtkPoly->GetStrips()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetStrips();

  if (verts->GetMaxCellSize() > 3)
  {
    // Panic and return
    MITK_ERROR <<"More than three vertices / cell detected, can't handle this data type!\n";
    return;
  }
  
  vtkSmartPointer<vtkPoints>     points = vtkPoly->GetPoints();

  if (points == 0)
  {
    MITK_ERROR <<"Corrupt vtkPoly, returning! \n";
    return;
  }

  // Deal with normals
  vtkSmartPointer<vtkDataArray> normals = vtkPoly->GetPointData()->GetNormals();

  if (normals == 0)
  {
    MITK_INFO <<"Generating normals for the vtkPoly data (mitk::OclSurface)";
    
    vtkSmartPointer<vtkPolyDataNormals> normalGen = vtkSmartPointer<vtkPolyDataNormals>::New();
    normalGen->SetInputData(vtkPoly);
    normalGen->AutoOrientNormalsOn();
    normalGen->Update();

    normals = normalGen->GetOutput()->GetPointData()->GetNormals();

    if (normals == 0)
    {
      MITK_ERROR <<"Couldn't generate normals, returning! \n";
      return;
    }

    vtkPoly->GetPointData()->SetNormals(normals);
    vtkPoly->GetPointData()->GetNormals()->Modified();
    vtkPoly->GetPointData()->Modified();
  }
 
  // Check if we have scalars
  vtkSmartPointer<vtkDataArray> scalars = vtkPoly->GetPointData()->GetScalars();

  bool pointsValid  = (points.GetPointer() == 0) ? false : true;
  bool normalsValid = (normals.GetPointer() == 0) ? false : true;
  bool scalarsValid = (scalars.GetPointer() == 0) ? false : true;
  
  unsigned int pointBufferSize = 0;
  unsigned int numOfPoints = static_cast<unsigned int> (points->GetNumberOfPoints());
  pointBufferSize = numOfPoints * sizeof(float) *3;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with points


  // Allocate memory
  m_PointBuffer = new float[numOfPoints*3];

  // Copy data to buffer
  memcpy(m_PointBuffer, points->GetVoidPointer(0), pointBufferSize);

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with normals

  if (normalsValid)
  {
    // Get the number of normals we have to deal with
    int m_NormalCount = static_cast<unsigned int> (normals->GetNumberOfTuples());
    assert(m_NormalCount == numOfPoints);

    // Size of the buffer that is required to store all the normals
    unsigned int normalBufferSize = numOfPoints * sizeof(float) * 3;

    // Allocate memory
    m_NormalBuffer = new float[numOfPoints*3];

    // Copy data to buffer
    memcpy(m_NormalBuffer, normals->GetVoidPointer(0), normalBufferSize);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with scalars (colors or texture coordinates)
  if (scalarsValid)
  {

    // Get the number of scalars we have to deal with
    int m_ScalarCount = static_cast<unsigned int> (scalars->GetNumberOfTuples());

    // Size of the buffer that is required to store all the scalars
    unsigned int scalarBufferSize = numOfPoints * sizeof(char) * 1;

    // Allocate memory
    m_ScalarBuffer = new char[numOfPoints];

    // Copy data to buffer
    memcpy(m_ScalarBuffer, scalars->GetVoidPointer(0), scalarBufferSize);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with cells - initialize index buffer
  vtkIdType npts;
  vtkIdType *pts;

  // Get the number of indices we have to deal with
  unsigned int m_IndexCount = static_cast<unsigned int> (verts->GetNumberOfCells());

  // Get the max number of vertices / cell
  int maxPointsPerCell = verts->GetMaxCellSize();

  // Get the number of indices we have to deal with
  unsigned int numOfTriangles = static_cast<unsigned int> (verts->GetNumberOfCells());

  // Allocate memory for the index buffer
  m_IndexBuffer = new unsigned int[numOfTriangles*3];
  memset(m_IndexBuffer, 0, numOfTriangles*3*sizeof(unsigned int));

  verts->InitTraversal();

  unsigned int cellIndex = 0;
  // Iterating through all the cells
  while (cellIndex < numOfTriangles)
  {
    verts->GetNextCell(npts, pts);

    // Copy the indices into the index buffer
    for (size_t i = 0; i < static_cast<size_t>(npts); i++)
      m_IndexBuffer[cellIndex*3 +i] = pts[i];

    cellIndex++;
  }
  MITK_INFO <<"Surface data initialized. Num of Points: " <<points->GetNumberOfPoints() <<" Num of Cells: " <<verts->GetNumberOfCells() <<"\n";

  vl::ref<vl::ArrayFloat3>  vlVerts   = new vl::ArrayFloat3;
  vl::ref<vl::ArrayFloat3>  vlNormals = new vl::ArrayFloat3;
  vl::ref<vl::DrawElementsUInt> vlTriangles = new vl::DrawElementsUInt(vl::PT_TRIANGLES);

  vlVerts->resize(numOfPoints *3);
  vlNormals->resize(numOfPoints *3);
   
  vlPoly->drawCalls()->push_back(vlTriangles.get());
  vlTriangles->indexBuffer()->resize(numOfTriangles*3);
  
  vlPoly->setVertexArray(vlVerts.get());
  vlPoly->setNormalArray(vlNormals.get());

  float * vertBufFlotPtr = reinterpret_cast<float *>( vlVerts->ptr());
  float * normBufFlotPtr = reinterpret_cast<float *>( vlNormals->ptr());

  // Vertices and normals
  for (unsigned int i=0; i<numOfPoints; ++i)
  {
    vertBufFlotPtr[3*i + 0] = m_PointBuffer[i*3 +0];
    vertBufFlotPtr[3*i + 1] = m_PointBuffer[i*3 +1];
    vertBufFlotPtr[3*i + 2] = m_PointBuffer[i*3 +2];

    normBufFlotPtr[3*i + 0] = m_NormalBuffer[i*3 +0];
    normBufFlotPtr[3*i + 1] = m_NormalBuffer[i*3 +1];
    normBufFlotPtr[3*i + 2] = m_NormalBuffer[i*3 +2];
  }

  // Make sure that the values are copied onto GPU memory
  //vlPoly->vertexArray()->updateBufferObject();
  //glFinish();

  // Read triangles
  for(unsigned int i=0; i<numOfTriangles; ++i)
  {
    vlTriangles->indexBuffer()->at(i*3+0) = m_IndexBuffer[i*3 +0];
    vlTriangles->indexBuffer()->at(i*3+1) = m_IndexBuffer[i*3 +1];
    vlTriangles->indexBuffer()->at(i*3+2) = m_IndexBuffer[i*3 +2];
  }

  // Make sure that the values are copied onto GPU memory
  vlVerts->updateBufferObject();
  vlNormals->updateBufferObject();
  vlTriangles->indexBuffer()->updateBufferObject();
  glFinish();

  /// \brief Buffer in host memory to store cell info
  if (m_IndexBuffer != 0)
    delete m_IndexBuffer;
  
  /// \brief Buffer in host memory to store vertex points
  if (m_PointBuffer != 0)
    delete m_PointBuffer;
  
  /// \brief Buffer in host memory to store normals associated with vertices
  if (m_NormalBuffer != 0)
    delete m_NormalBuffer;

  /// \brief Buffer in host memory to store scalar info associated with vertices
  if (m_ScalarBuffer != 0)
    delete m_ScalarBuffer;

  //MITK_INFO <<"Num of VL vertices: " <<vlPoly->vertexArray()->size()/3;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::AddImageActor(const mitk::Image::Pointer& mitkImg)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  unsigned int* dims = 0;
  dims = mitkImg->GetDimensions();
  // we do not own dims!

  if (dims[2] <= 1)
  {
    return Add2DImageActor(mitkImg);
  }
  else
  {
    return Add3DImageActor(mitkImg);
  }
}


//-----------------------------------------------------------------------------
vl::EImageType VLQtWidget::MapITKPixelTypeToVL(int itkComponentType)
{
  static const vl::EImageType     typeMap[] =
  {
    vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::UNKNOWNCOMPONENTTYPE = 0
    vl::IT_UNSIGNED_BYTE,   // itk::ImageIOBase::UCHAR = 1
    vl::IT_BYTE,            // itk::ImageIOBase::CHAR = 2
    vl::IT_UNSIGNED_SHORT,  // itk::ImageIOBase::USHORT = 3
    vl::IT_SHORT,           // itk::ImageIOBase::SHORT = 4
    vl::IT_UNSIGNED_INT,    // itk::ImageIOBase::UINT = 5
    vl::IT_INT,             // itk::ImageIOBase::INT = 6
    vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::ULONG = 7
    vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::LONG = 8
    vl::IT_FLOAT,           // itk::ImageIOBase::FLOAT = 9
    vl::IT_IMPLICIT_TYPE    // itk::ImageIOBase::DOUBLE = 10
  };

  return typeMap[itkComponentType];
}


//-----------------------------------------------------------------------------
vl::EImageFormat VLQtWidget::MapComponentsToVLColourFormat(int components)
{
  // this assumes the image data is a normal colour image, not encoding pointers or
  // indices, or similar stuff.

  switch (components)
  {
    default:
    case 1:
      return vl::IF_LUMINANCE;
    case 2:
      return vl::IF_RG;
    case 3:
      return vl::IF_RGB;
    case 4:
      return vl::IF_RGBA;
  }
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::Add2DImageActor(const mitk::Image::Pointer& mitkImg)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

  unsigned int*       dims    = mitkImg->GetDimensions();    // we do not own dims!
  mitk::PixelType     pixType = mitkImg->GetPixelType();
  vl::EImageType      type    = MapITKPixelTypeToVL(pixType.GetComponentType());
  vl::EImageFormat    format  = MapComponentsToVLColourFormat(pixType.GetNumberOfComponents());

  vl::ref<vl::Image>    vlImg = new vl::Image(dims[0], dims[1], 0, 1, format, type);


  // sanity check
  unsigned int  size = (dims[0] * dims[1] * dims[2]) * pixType.GetSize();
  assert(vlImg->requiredMemory() == size);

  try
  {
    mitk::ImageReadAccessor   readAccess(mitkImg, mitkImg->GetVolumeData(0));
    const void*               cPointer = readAccess.GetData();
    std::memcpy(vlImg->pixels(), cPointer, vlImg->requiredMemory());

  }
  catch (...)
  {
    // FIXME: error handling?
    MITK_ERROR << "Did not get pixel read access to 2D image.";
  }


  vl::ref<vl::Transform> tr     = new vl::Transform;
  UpdateTransformFromData(tr, mitkImg.GetPointer());

  vl::ref<vl::Geometry>         vlquad = CreateGeometryFor2DImage(dims[0], dims[1]);

  vl::ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  fx->shader()->gocTextureSampler(0)->setTexture(new vl::Texture(vlImg.get(), vl::TF_UNKNOWN, false));
  fx->shader()->gocTextureSampler(0)->setTexParameter(m_DefaultTextureParams.get());
  fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
  fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());
  // UpdateDataNode() takes care of assigning colour etc.
  // FIXME: alpha-blending? independent of opacity prop!

  vl::ref<vl::Actor>    actor = m_SceneManager->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[actor] = vlquad;

  return actor;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQtWidget::Add3DImageActor(const mitk::Image::Pointer& mitkImg)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());


  mitk::PixelType pixType = mitkImg->GetPixelType();
  size_t numOfComponents = pixType.GetNumberOfComponents();

  if (false)
  {
    std::cout << "Original pixel type:" << std::endl;
    std::cout << " PixelType: " <<pixType.GetTypeAsString() << std::endl;
    std::cout << " BitsPerElement: " <<pixType.GetBpe() << std::endl;
    std::cout << " NumberOfComponents: " << numOfComponents << std::endl;
    std::cout << " BitsPerComponent: " <<pixType.GetBitsPerComponent() << std::endl;
  }


  vl::ref<vl::Image>     vlImg;

  try
  {
    mitk::ImageReadAccessor   readAccess(mitkImg, mitkImg->GetVolumeData(0));
    const void*               cPointer = readAccess.GetData();


    vl::EImageType     type = MapITKPixelTypeToVL(pixType.GetComponentType());
    vl::EImageFormat   format;

    if (type != vl::IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = vl::IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = vl::IF_RG_INTEGER;
      else if (numOfComponents == 3)
        format = vl::IF_RGB_INTEGER;
      else if (numOfComponents == 4)
        // FIXME: not sure whether we really want integer formats here!
        //        for now, dont do integer for rgba, we have quite a few rgba images.
        format = vl::IF_RGBA;//_INTEGER;
    }
    else if (type == vl::IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = vl::IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = vl::IF_RG;
      else if (numOfComponents == 3)
        format = vl::IF_RGB;
      else if (numOfComponents == 4)
        format = vl::IF_RGBA;
    }

    unsigned int* dims = 0;
    dims = mitkImg->GetDimensions();
    // we do not own dims!

    int bytealign = 1;
    if (dims[2] <= 1)
      vlImg = new vl::Image(dims[0], dims[1], 0, bytealign, format, type);
    else
      vlImg = new vl::Image(dims[0], dims[1], dims[2], bytealign, format, type);

    // sanity check
    unsigned int size = (dims[0] * dims[1] * dims[2]) * pixType.GetSize();
    assert(vlImg->requiredMemory() == size);
    std::memcpy(vlImg->pixels(), cPointer, vlImg->requiredMemory());

    vlImg = vlImg->convertFormat(vl::IF_LUMINANCE)->convertType(vl::IT_UNSIGNED_SHORT);
/*
    ref<KeyValues> tags = new KeyValues;
    tags->set("Origin")    = Say("%n %n %n") << mitkImg->GetGeometry()->GetOrigin()[0]  << mitkImg->GetGeometry()->GetOrigin()[1]  << mitkImg->GetGeometry()->GetOrigin()[2];
    tags->set("Spacing")   = Say("%n %n %n") << mitkImg->GetGeometry()->GetSpacing()[0] << mitkImg->GetGeometry()->GetSpacing()[1] << mitkImg->GetGeometry()->GetSpacing()[2];
    vlImg->setTags(tags.get());
*/
  }
  catch(mitk::Exception& e)
  {
    // deal with the situation not to have access
    assert(false);
  }


  float opacity;
  mitkImg->GetPropertyList()->GetFloatProperty("opacity", opacity);

  mitk::BaseProperty::Pointer   colourProp = mitkImg->GetProperty("color");
  mitk::Color                   mitkColor;
  if (colourProp.IsNotNull())
    mitkColor = dynamic_cast<mitk::ColorProperty*>(colourProp.GetPointer())->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];
  color[3] = opacity;

  vl::ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->enable(vl::EN_DEPTH_TEST);
  fx->shader()->enable(vl::EN_BLEND);
  fx->shader()->setRenderState(m_Light.get(), 0);
  fx->shader()->enable(vl::EN_LIGHTING);
  fx->shader()->gocMaterial()->setDiffuse(color);
  fx->shader()->gocMaterial()->setTransparency(opacity);

  vl::String fragmentShaderSource   = LoadGLSLSourceFromResources("volume_raycast_isosurface_transp.fs");
  vl::String vertexShaderSource     = LoadGLSLSourceFromResources("volume_luminance_light.vs");

  // The GLSL program used to perform the actual rendering.
  // The \a volume_luminance_light.fs fragment shader allows you to specify how many 
  // lights to use (up to 4) and can optionally take advantage of a precomputed normals texture.
  vl::ref<vl::GLSLProgram>    glslShader = fx->shader()->gocGLSLProgram();
  glslShader->attachShader(new vl::GLSLFragmentShader(fragmentShaderSource));
  glslShader->attachShader(new vl::GLSLVertexShader(vertexShaderSource));

  vl::ref<vl::Actor>    imageActor = new vl::Actor;
  imageActor->setEffect(fx.get());
  imageActor->setUniform(m_ThresholdVal.get());

  vl::ref<vl::Transform>    tr = new vl::Transform;
  //UpdateTransfromFromData(tr, cudaImg);       // FIXME: needs proper thinking through
  imageActor->setTransform(tr.get());
  m_SceneManager->tree()->addActor(imageActor.get());

  // this is a callback: gets triggered everytime its bound actor is to be rendered.
  // during that callback it updates the uniforms of our glsl shader to match fixed-function state.
  vl::ref<vl::RaycastVolume>    raycastVolume = new vl::RaycastVolume;
  // this stuffs the proxy geometry onto our actor, as lod-slot zero.
  raycastVolume->bindActor(imageActor.get());


  // we do not own dims!
  unsigned int*   dims    = mitkImg->GetDimensions();
  mitk::Vector3D  spacing = mitkImg->GetGeometry()->GetSpacing();

  float dimX = (float) dims[0] * spacing[0] / 2.0f;
  float dimY = (float) dims[1] * spacing[1] / 2.0f;
  float dimZ = (float) dims[2] * spacing[2] / 2.0f;
  float shiftX = 0.0f;//0.5f * spacing[0];
  float shiftY = 0.0f;//0.5f * spacing[1];
  float shiftZ = 0.0f;//0.5f * spacing[2];

  vl::AABB    volume_box(vl::vec3(-dimX + shiftX, -dimY + shiftY, -dimZ + shiftZ)
                       , vl::vec3( dimX + shiftX,  dimY + shiftY,  dimZ + shiftZ));
  raycastVolume->setBox(volume_box);
  raycastVolume->generateTextureCoordinates(vl::ivec3(vlImg->width(), vlImg->height(), vlImg->depth()));


  // note img has been converted unconditionally to IT_UNSIGNED_SHORT above!
  fx->shader()->gocTextureSampler(0)->setTexture(new vl::Texture(vlImg.get(), vl::TF_LUMINANCE16, false, false));
  fx->shader()->gocUniform("volume_texunit")->setUniformI(0);

  // generate a simple colored transfer function
  vl::ref<vl::Image>  trfunc = vl::makeColorSpectrum(1024, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson);
  // installs the transfer function as texture #1
  fx->shader()->gocTextureSampler(1)->setTexture(new vl::Texture(trfunc.get()));
  fx->shader()->gocUniform("trfunc_texunit")->setUniformI(1);
/*
  ref<Image> gradient;
  // note that this can take a while...
  gradient = vl::genGradientNormals( vlImg.get() );
  fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 1);
  fx->shader()->gocTextureSampler( 2 )->setTexture( new Texture( gradient.get(), TF_RGBA, false, false ) );
  fx->shader()->gocUniform( "gradient_texunit" )->setUniformI( 2 );
*/
  fx->shader()->gocUniform("precomputed_gradient")->setUniformI(0);
  // used to compute on the fly the normals based on the volume's gradient
  fx->shader()->gocUniform("gradient_delta")->setUniform(vl::fvec3(0.5f / vlImg->width(), 0.5f / vlImg->height(), 0.5f / vlImg->depth()));

  fx->shader()->gocUniform( "sample_step" )->setUniformF(1.0f / 512.0f);

  vtkLinearTransform * nodeVtkTr = mitkImg->GetGeometry()->GetVtkTransform();
  vtkSmartPointer<vtkMatrix4x4> geometryTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  nodeVtkTr->GetMatrix(geometryTransformMatrix);

  float vals[16];
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double val = geometryTransformMatrix->GetElement(i, j);
      vals[i*4+j] = val;
    }
  }
  vl::mat4 mat(vals);
  tr->setLocalMatrix(mat);

  // refresh window
  //openglContext()->update();

  m_ActorToRenderableMap[imageActor] = dynamic_cast<vl::Renderable*>( vlImg.get());
  return imageActor;
}


//-----------------------------------------------------------------------------
vl::String VLQtWidget::LoadGLSLSourceFromResources(const char* filename)
{
  QString   sourceFilename(filename);
  sourceFilename.prepend(":/NewVisualization/");
  QFile   sourceFile;
  sourceFile.setFileName(sourceFilename);

  if (sourceFile.exists() && sourceFile.open(QIODevice::ReadOnly))
  {
    QTextStream   textStream(&sourceFile);

    QString   qContents = textStream.readAll();
    return vl::String(qContents.toStdString().c_str());
  }
  else
  {
    MITK_ERROR << "Failed to open GLSL source file: " << filename << std::endl;
    throw std::runtime_error("Failed to open GLSL source file");
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::UpdateTranslucentTriangles()
{
  bool    thereIsSomethingTranslucent = true;
  //if (!m_TranslucentStructuresMerged)
  {
    thereIsSomethingTranslucent = MergeTranslucentTriangles();
  }
#if 0
  else
  {
    vl::ref<vl::ActorCollection> actors = m_SceneManager->tree()->actors();
    int numOfActors = actors->size();
  
    unsigned int summedNumOfVerts = 0;
    for (int i = 0; i < numOfActors; i++)
    {
      vl::ref<vl::Actor> act = actors->at(i);
      std::string objName = act->objectName();
      int renderBlock = act->renderBlock();
    
      size_t found =objName.find("_surface");
      if ((found != std::string::npos) && (renderBlock == RENDERBLOCK_TRANSLUCENT))
      {
        vl::ref<vl::Renderable> ren = m_ActorToRenderableMap[act];
        vl::ref<vl::Geometry> surface = dynamic_cast<vl::Geometry*>(ren.get());
        if (surface == 0)
          continue;

        // Update vertex counter
        unsigned int numOfVertices = surface->vertexArray()->size() /3;
        summedNumOfVerts += numOfVertices;
      }
    }

    if (summedNumOfVerts != m_TotalNumOfTranslucentVertices && summedNumOfVerts != 0)
    {
      thereIsSomethingTranslucent = MergeTranslucentTriangles();
    }
    else if (summedNumOfVerts == 0)
    {
      thereIsSomethingTranslucent = false;
    }
  }
#endif
  // m_TotalNumOfTranslucentVertices is set by MergeTranslucentTriangles().

  bool    mergedok = false;
  if ((m_TotalNumOfTranslucentVertices > 0) && (m_TranslucentActors.size() > 0) && thereIsSomethingTranslucent)
    mergedok = true;

  bool  sortedok = false;
  if (mergedok)
    sortedok = SortTranslucentTriangles();

  // the sorted-translucent-all-in-one-actor is only visible if merging and sorting actually worked.
  // otherwise, fall back to unsorted.
  if (mergedok && sortedok)
  {
    vl::ref<vl::Effect>    fx;
    vl::ref<vl::Transform> tr;

    if (m_TranslucentSurfaceActor == 0)
    {
      // Add the new merged geometry actor
      tr = new vl::Transform();
      fx = new vl::Effect;
      m_TranslucentSurfaceActor = m_SceneManager->tree()->addActor(m_TranslucentSurface.get(), fx.get(), tr.get());
      m_TranslucentSurfaceActor->setObjectName("m_TranslucentSurfaceActor");
    }
    else
    {
      fx = m_TranslucentSurfaceActor->effect();
      tr = m_TranslucentSurfaceActor->transform();
    }

    m_TranslucentSurfaceActor->setRenderBlock(RENDERBLOCK_SORTEDTRANSLUCENT);
    m_TranslucentSurfaceActor->setEnableMask(ENABLEMASK_SORTEDTRANSLUCENT);
    fx->shader()->gocMaterial()->setColorMaterialEnabled(true);

    // no backface culling for translucent objects: you should be able to see the backside!
    fx->shader()->disable(vl::EN_CULL_FACE);

    fx->shader()->enable(vl::EN_BLEND);
    fx->shader()->enable(vl::EN_DEPTH_TEST);
    fx->shader()->enable(vl::EN_LIGHTING);
    fx->shader()->setRenderState(m_Light.get(), 0 );

    // dont render unsorted translucent triangles by simply disabling that part of the pipeline.
    m_OpaqueObjectsRendering->setEnableMask(m_OpaqueObjectsRendering->enableMask() & ~ENABLEMASK_TRANSLUCENT | ENABLEMASK_SORTEDTRANSLUCENT);
  }
  else
  {
    // if there is no sorted-translucent-geometry just disable that part of the pipeline.
    m_OpaqueObjectsRendering->setEnableMask(m_OpaqueObjectsRendering->enableMask() & ~ENABLEMASK_SORTEDTRANSLUCENT);
    // also re-enable unsorted possibly-translucent geometry.
    m_OpaqueObjectsRendering->setEnableMask(m_OpaqueObjectsRendering->enableMask() | ENABLEMASK_TRANSLUCENT);
  }
}


//-----------------------------------------------------------------------------
bool VLQtWidget::MergeTranslucentTriangles()
{
  // sanity check: internal method, context should have been activated by caller.
  assert(this->context() == QGLContext::currentContext());

  vl::ref<vl::ActorCollection> actors = m_SceneManager->tree()->actors();
  int numOfActors = actors->size();

  if (m_OclService == 0)
    return false;

  // hopefully the buffers wrapping vbos will have finished doing stuff.
  glFinish();

  // Get context 
  cl_context clContext = m_OclService->GetContext();
  cl_command_queue clCmdQue = m_OclService->GetCommandQueue();

  if (clContext == 0 || clCmdQue == 0)
    return false;

  std::vector<vl::ref<vl::Geometry> >  translucentSurfaces;
  std::vector<vl::fvec4>               translucentColors;

  cl_int clStatus = 0;

  // Instantiate TriangleSorter
  if (m_OclTriangleSorter == 0)
    m_OclTriangleSorter = new mitk::OclTriangleSorter();

  // Make sure previous values are cleared
  m_OclTriangleSorter->Reset();

  m_TotalNumOfTranslucentTriangles = 0;
  m_TotalNumOfTranslucentVertices  = 0;

  m_TranslucentActors.clear();

  if (m_MergedTranslucentIndexBuf != 0)
  {
    clStatus = clReleaseMemObject(m_MergedTranslucentIndexBuf);
    CHECK_OCL_ERR(clStatus);
    m_MergedTranslucentIndexBuf = 0;
  }

  if (m_MergedTranslucentVertexBuf != 0)
  {
    clStatus = clReleaseMemObject(m_MergedTranslucentVertexBuf);
    CHECK_OCL_ERR(clStatus);
    m_MergedTranslucentVertexBuf = 0;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Find all the translucent geometries and stuff their actor/geometry/transform to lists

  for (int i = 0; i < numOfActors; i++)
  {
    vl::ref<vl::Actor>  act           = actors->at(i);
    const std::string&  objName       = act->objectName();
    int                 renderBlock   = act->renderBlock();
    bool                enabled       = act->enableMask() != 0;
    bool                nameOk        = objName.find("_surface") != std::string::npos;

    if (nameOk && (renderBlock == RENDERBLOCK_TRANSLUCENT) && enabled)
    {
      vl::ref<vl::Renderable> ren = m_ActorToRenderableMap[act];
      vl::ref<vl::Geometry> surface = dynamic_cast<vl::Geometry*>(ren.get());
      if (surface == 0)
        continue;

      m_TranslucentActors.insert(act);

      vl::ref<vl::Effect> fx = act->effect();
      vl::fvec4 color = fx->shader()->gocMaterial()->frontDiffuse();
      translucentColors.push_back(color);
      translucentSurfaces.push_back(surface);
    }
  }

  // Return if there's nothing to do 
  if (translucentSurfaces.size() == 0)
  {
    // dont call again, there is nothing to merge.
    m_TranslucentStructuresMerged = true;
    return false;
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Acquire the VBO/IBO handles of the translucent objects
  // and pass them to the triangle sorter as GLuint objects

  for (int i = 0; i < translucentSurfaces.size(); i++)
  {
    // Get pointer to the index buffer
    size_t numOfDrawcalls = translucentSurfaces.at(i)->drawCalls()->size();
    vl::DrawCall * dc = translucentSurfaces.at(i)->drawCalls()->at(numOfDrawcalls-1);
    vl::ref<vl::DrawElementsUInt> vlTriangles = dynamic_cast<vl::DrawElementsUInt *>(dc);

    // Update triangle counter
    unsigned int numOfTriangles = vlTriangles->countTriangles();
    m_TotalNumOfTranslucentTriangles += numOfTriangles;
    
    // Update buffer, get handle and push it to TriangleSorter
    vlTriangles->indexBuffer()->updateBufferObject();
    GLuint indexBufferHandle = vlTriangles->indexBuffer()->bufferObject()->handle();
    m_OclTriangleSorter->AddGLIndexBuffer(indexBufferHandle, numOfTriangles);

    // Update vertex counter
    unsigned int numOfVertices = translucentSurfaces.at(i)->vertexArray()->size() /3;
    m_TotalNumOfTranslucentVertices += numOfVertices;

    // Update buffer, get handle and push it to TriangleSorter
    translucentSurfaces.at(i)->vertexArray()->updateBufferObject();
    GLuint vertexArrayHandle = translucentSurfaces.at(i)->vertexArray()->bufferObject()->handle();
    m_OclTriangleSorter->AddGLVertexBuffer(vertexArrayHandle, numOfVertices);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Allocate VL arrays for the merged object's vertices and normals

  vl::ref<vl::ArrayFloat3>      vlVerts;
  vl::ref<vl::ArrayFloat3>      vlNormals;
  vl::ref<vl::ArrayUByte4>      vlColors;
  vl::ref<vl::DrawElementsUInt> vlTriangles;

  // Acquire or create the merged VL geometry object that holds all translucent triangles
  if (m_TranslucentSurface == 0)
  {
    m_TranslucentSurface = new vl::Geometry();

    vlVerts     = new vl::ArrayFloat3;
    vlNormals   = new vl::ArrayFloat3;
    vlTriangles = new vl::DrawElementsUInt(vl::PT_TRIANGLES);
    vlColors    = new vl::ArrayUByte4;

    m_TranslucentSurface->setVertexArray(vlVerts.get());
    m_TranslucentSurface->setNormalArray(vlNormals.get());
    m_TranslucentSurface->setColorArray(vlColors.get());
    m_TranslucentSurface->drawCalls()->push_back(vlTriangles.get());
  }
  else
  {
    vlVerts   = dynamic_cast<vl::ArrayFloat3 *>(m_TranslucentSurface->vertexArray());
    vlNormals = dynamic_cast<vl::ArrayFloat3 *>(m_TranslucentSurface->normalArray());
    vlColors  = dynamic_cast<vl::ArrayUByte4 *>(m_TranslucentSurface->colorArray());

    size_t numOfDrawcalls = m_TranslucentSurface->drawCalls()->size();
    vl::DrawCall * dc = m_TranslucentSurface->drawCalls()->at(numOfDrawcalls-1);
    vlTriangles = dynamic_cast<vl::DrawElementsUInt *>(dc);

    if (vlVerts == 0 || vlNormals == 0 || vlTriangles == 0)
    {
      MITK_ERROR <<"Failed to acquire buffer objects from the VL geometry.";
      return false;
    }
  }

  // hack bounding box. vl uses it for scene culling.
  vl::AABB    mergedbb;
  vl::Sphere  mergedbs;
  for (int i = 0; i < translucentSurfaces.size(); ++i)
  {
    mergedbb += translucentSurfaces[i]->boundingBox();
    mergedbs += translucentSurfaces[i]->boundingSphere();
  }
  m_TranslucentSurface->setBoundingBox(mergedbb);
  m_TranslucentSurface->setBoundingSphere(mergedbs);
  m_TranslucentSurface->setBoundsDirty(false);

  // Resize buffer objects
  vlVerts->resize(m_TotalNumOfTranslucentVertices *3);
  vlNormals->resize(m_TotalNumOfTranslucentVertices *3);
  vlColors->resize(m_TotalNumOfTranslucentVertices);
  vlTriangles->indexBuffer()->resize(m_TotalNumOfTranslucentTriangles*3);

  // Make sure that the buffers are allocated in GPU memory
  m_TranslucentSurface->vertexArray()->updateBufferObject();
  m_TranslucentSurface->normalArray()->updateBufferObject();
  m_TranslucentSurface->colorArray()->updateBufferObject();
  vlTriangles->indexBuffer()->updateBufferObject();

  // this is good here! do not remove.
  glFinish();


  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Get hold of the Index buffer of the merged object a'la OpenCL mem
  GLuint mergedIndexBufferHandle = vlTriangles->indexBuffer()->bufferObject()->handle();
  m_MergedTranslucentIndexBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, mergedIndexBufferHandle, &clStatus);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }
  // note: m_MergedTranslucentIndexBuf is normally released at the beginning of this method.

  clStatus = clEnqueueAcquireGLObjects(clCmdQue, 1, &m_MergedTranslucentIndexBuf, 0, NULL, NULL);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    clStatus = clReleaseMemObject(m_MergedTranslucentIndexBuf);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
    }
    m_MergedTranslucentIndexBuf = 0;
    return false;
  }

  // Here we retrieve the merged and sorted index buffer
  cl_uint totalNumOfVertices;
  bool mergedok = m_OclTriangleSorter->MergeIndexBuffers(m_MergedTranslucentIndexBuf, totalNumOfVertices);
  if (!mergedok)
    return false;

  if (totalNumOfVertices != m_TotalNumOfTranslucentVertices)
  {
    MITK_ERROR <<"Index buffer merge error, returning!";
    return false;
  }

  clStatus = clEnqueueReleaseGLObjects(clCmdQue, 1, &m_MergedTranslucentIndexBuf, 0, NULL, NULL);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  clStatus = clFinish(clCmdQue);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  // Get hold of the Vertex/Normal buffers of the merged object a'la OpenCL mem
  GLuint mergedVertexArrayHandle = vlVerts->bufferObject()->handle();
  m_MergedTranslucentVertexBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, mergedVertexArrayHandle, &clStatus);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  clStatus = clEnqueueAcquireGLObjects(clCmdQue, 1, &m_MergedTranslucentVertexBuf, 0, NULL, NULL);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  // Get normal array
  GLuint mergedNormalArrayHandle = vlNormals->bufferObject()->handle();
  cl_mem clMergedNormalBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, mergedNormalArrayHandle, &clStatus);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  clStatus = clEnqueueAcquireGLObjects(clCmdQue, 1, &clMergedNormalBuf, 0, NULL, NULL);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  size_t vertexBufferOffset = 0;
  size_t normalBufferOffset = 0;
  size_t colorBufferOffset = 0;

  std::vector<cl_mem> clVertexBufs;
  std::vector<cl_mem> clNormalBufs;

  // Here we merge the vertices and normals and colors
  for (size_t i = 0; i < translucentSurfaces.size(); i++)
  {
    // get number of vertices
    unsigned int numOfVertices2 = translucentSurfaces.at(i)->vertexArray()->size() /3;
    //MITK_INFO <<"Copying vertices of structure " <<i <<": " <<numOfVertices2;
    unsigned int computedSize = numOfVertices2 * sizeof(GLfloat) * 3;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Get vertex array
    GLuint vertexArrayHandle = translucentSurfaces.at(i)->vertexArray()->bufferObject()->handle();
    cl_mem clVertexBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, vertexArrayHandle, &clStatus);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }
    clVertexBufs.push_back(clVertexBuf);

    clStatus = clEnqueueAcquireGLObjects(clCmdQue, 1, &clVertexBuf, 0, NULL, NULL);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }

    // Copy to merged buffer
    clStatus = clEnqueueCopyBuffer(clCmdQue, clVertexBuf, m_MergedTranslucentVertexBuf, 0, vertexBufferOffset, computedSize, 0, 0, 0);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }
    vertexBufferOffset += computedSize;

    clStatus = clEnqueueReleaseGLObjects(clCmdQue, 1, &clVertexBuf, 0, NULL, NULL);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Get normal array
    GLuint normalArrayHandle = translucentSurfaces.at(i)->normalArray()->bufferObject()->handle();
    cl_mem clNormalBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, normalArrayHandle, &clStatus);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }
    clNormalBufs.push_back(clNormalBuf);

    clStatus = clEnqueueAcquireGLObjects(clCmdQue, 1, &clNormalBuf, 0, NULL, NULL);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }

    // Copy to merged buffer
    clStatus = clEnqueueCopyBuffer(clCmdQue, clNormalBuf, clMergedNormalBuf, 0, normalBufferOffset, computedSize, 0, 0, 0);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }
    normalBufferOffset += computedSize;

    clStatus = clEnqueueReleaseGLObjects(clCmdQue, 1, &clNormalBuf, 0, NULL, NULL);
    if (clStatus)
    {
      CHECK_OCL_ERR(clStatus);
      return false;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Get color array
    size_t colorBufSize = numOfVertices2*sizeof(unsigned int);
    vl::fvec4 color = translucentColors.at(i);

    unsigned int * colorData = new unsigned int[numOfVertices2];
    for (unsigned int bla = 0; bla < numOfVertices2; bla++)
    {
      // Color format: AABBGGRR
      unsigned char a = color[3] * 255;
      unsigned char b = color[2] * 255;
      unsigned char g = color[1] * 255;
      unsigned char r = color[0] * 255;
      colorData[bla] = r | (g << 8) | (b << 16) | (a << 24);
    }

    vlColors->bufferObject()->setBufferSubData(colorBufferOffset, colorBufSize, colorData);
    glFinish();
    colorBufferOffset += colorBufSize;
    delete colorData;
 
  }

/*
  MITK_INFO <<"Total num of triangles: " <<totalNumOfTriangles <<" Total num of vertices: " <<totalNumOfVertices;

  //range = 1024;
  float step = 255.0f/range;

  unsigned int * colorData = new unsigned int[totalNumOfVertices];
  float * vertDistData = new float[totalNumOfVertices];
  memset(vertDistData, 0, totalNumOfVertices*sizeof(float));

  for (unsigned int bla = 0; bla < totalNumOfTriangles; bla++)
  {
    float distVal  = mitk::OclTriangleSorter::IFloatFlip(mergedDistances[bla]);
    //if (distVal >= 1024.0f)
    //  distVal = 1023.0f;
    
    // Color format: AABBGGRR
    unsigned char a = 255;
    unsigned char b = 0;
    unsigned char g = (distVal-minDist)*step;
    unsigned char r = 255 - (distVal-minDist)*step;
    unsigned int colorVal = r | (g << 8) | (b << 16) | (a << 24);
  
//    if (bla < 1000)
//      std::cout <<"Index: " <<bla <<" dist: " <<std::setprecision(10) <<distVal <<" color: " <<(int)r <<" " <<(int)g <<" " <<(int)b <<"\n";

    cl_uint vertIndex0 = mergedIBO[bla*3 +0];
    cl_uint vertIndex1 = mergedIBO[bla*3 +1];
    cl_uint vertIndex2 = mergedIBO[bla*3 +2];

    if (vertIndex0 >= totalNumOfVertices)
    {
      MITK_INFO <<"vertIndex0: " <<vertIndex0 <<" Total num of vertices: " <<totalNumOfVertices;
      break;
    }

    if (vertIndex1 >= totalNumOfVertices)
    {
      MITK_INFO <<"vertIndex1: " <<vertIndex1 <<" Total num of vertices: " <<totalNumOfVertices;
      break;
    }

    if (vertIndex2 >= totalNumOfVertices)
    {
      MITK_INFO <<"vertIndex2: " <<vertIndex2 <<" Total num of vertices: " <<totalNumOfVertices;
      break;
    }

    if (distVal > vertDistData[vertIndex0])
    {
      vertDistData[vertIndex0] = distVal;
      colorData[vertIndex0]    = colorVal;
    }

    if (distVal > vertDistData[vertIndex1])
    {
      vertDistData[vertIndex1] = distVal;
      colorData[vertIndex1]    = colorVal;
    }

    if (distVal > vertDistData[vertIndex2])
    {
      vertDistData[vertIndex2] = distVal;
      colorData[vertIndex2]    = colorVal;
    }
  }

  colorBufferOffset = 0;
  vlColors->bufferObject()->setBufferSubData(colorBufferOffset, totalNumOfVertices*sizeof(unsigned int), colorData);

  delete colorData;
  delete vertDistData;
*/
  clStatus |= clEnqueueReleaseGLObjects(clCmdQue, 1, &m_MergedTranslucentVertexBuf, 0, NULL, NULL);
  clStatus |= clEnqueueReleaseGLObjects(clCmdQue, 1, &clMergedNormalBuf, 0, NULL, NULL);
  clStatus |= clFinish(clCmdQue);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }

  for (size_t ii = 0; ii < clVertexBufs.size(); ii++)
  {
    clStatus = clReleaseMemObject(clVertexBufs.at(ii));
    CHECK_OCL_ERR(clStatus);
    clVertexBufs.at(ii) = 0;
  }
  
  for (size_t ii = 0; ii < clNormalBufs.size(); ii++)
  {
    clStatus = clReleaseMemObject(clNormalBufs.at(ii));
    CHECK_OCL_ERR(clStatus);
    clNormalBufs.at(ii) = 0;
  }

  clStatus = clReleaseMemObject(clMergedNormalBuf);
  if (clStatus)
  {
    CHECK_OCL_ERR(clStatus);
    return false;
  }
  clMergedNormalBuf = 0;

  m_TranslucentStructuresMerged = true;
  return true;
}


//-----------------------------------------------------------------------------
bool VLQtWidget::SortTranslucentTriangles()
{
  // Get context 
  cl_context clContext = m_OclService->GetContext();
  cl_command_queue clCmdQue = m_OclService->GetCommandQueue();

  if (clContext == 0 || clCmdQue == 0 || !m_TranslucentStructuresMerged)
    return false;

  // Get camera position
  vl::vec3 cameraPos = m_Camera->modelingMatrix().getT();
  cl_float4 clCameraPos;
  clCameraPos.s[0] = cameraPos[0];
  clCameraPos.s[1] = cameraPos[1];
  clCameraPos.s[2] = cameraPos[2];
  clCameraPos.s[3] = 1.0f;

  cl_int clStatus = 0;

  // Pass on the camera position to the sorter
  m_OclTriangleSorter->SetViewPoint(clCameraPos);

  // Compute trinagle distances and sort the triangles
  bool sortok = m_OclTriangleSorter->SortIndexBufferByDist(m_MergedTranslucentIndexBuf, m_MergedTranslucentVertexBuf, m_TotalNumOfTranslucentTriangles, m_TotalNumOfTranslucentVertices);
  if (!sortok)
    return false;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // This code is only for debugging. It colors the translucent object's triangles based on distance from camera.
/*
  vl::ref<vl::ArrayUByte4>      vlColors;
  vlColors  = dynamic_cast<vl::ArrayUByte4 *>(m_TranslucentSurface->colorArray());

  cl_mem mergedDistBufOutput = clCreateBuffer(clContext, CL_MEM_READ_WRITE, m_TotalNumOfTranslucentTriangles*sizeof(cl_uint), 0, 0);
  
  // Here we retrieve the merged and sorted distance buffer
  m_OclTriangleSorter->GetTriangleDistOutput(mergedDistBufOutput, m_TotalNumOfTranslucentTriangles);

  cl_uint * mergedDistances = new cl_uint[m_TotalNumOfTranslucentTriangles];
  clStatus = clEnqueueReadBuffer(clCmdQue, mergedDistBufOutput, true, 0, m_TotalNumOfTranslucentTriangles*sizeof(cl_uint), mergedDistances, 0, 0, 0);
  CHECK_OCL_ERR(clStatus);

  cl_uint * mergedIBO = new cl_uint[m_TotalNumOfTranslucentTriangles*3];
  clStatus = clEnqueueReadBuffer(clCmdQue, m_MergedTranslucentIndexBuf, true, 0, m_TotalNumOfTranslucentTriangles*3*sizeof(cl_uint), mergedIBO, 0, 0, 0);
  CHECK_OCL_ERR(clStatus);

  //std::ofstream outfileA;
  //outfileA.open ("d://triangleDists.txt", std::ios::out);

  float maxDist = -FLT_MAX;
  for (int kk = 0; kk < m_TotalNumOfTranslucentTriangles; kk++)
  {
    float val  = mitk::OclTriangleSorter::IFloatFlip(mergedDistances[kk]);

    if (val > maxDist)
      maxDist = val;

    //outfileA <<"Index: " <<kk <<" s: " <<mergedDistances[kk] <<" Dist: " <<std::setprecision(10) <<val <<"\n";
  }

  //outfileA.close();

  float minDist = FLT_MAX;
  for (int kk = 0; kk < m_TotalNumOfTranslucentTriangles; kk++)
  {
    float val  = mitk::OclTriangleSorter::IFloatFlip(mergedDistances[kk]);
    if (val < minDist)
      minDist = val;
  }

  float range = (maxDist-minDist);
  float step = 255.0f/range;

  unsigned int * colorData = new unsigned int[m_TotalNumOfTranslucentVertices];
  float * vertDistData = new float[m_TotalNumOfTranslucentVertices];
  memset(vertDistData, 0, m_TotalNumOfTranslucentVertices*sizeof(float));

  for (unsigned int bla = 0; bla < m_TotalNumOfTranslucentTriangles; bla++)
  {
    float distVal  = mitk::OclTriangleSorter::IFloatFlip(mergedDistances[bla]);
    //if (distVal >= 1024.0f)
    //  distVal = 1023.0f;
    
    // Color format: AABBGGRR
    unsigned char a = 255;
    unsigned char b = 0;
    unsigned char g = (distVal-minDist)*step;
    unsigned char r = 255 - (distVal-minDist)*step;
    unsigned int colorVal = r | (g << 8) | (b << 16) | (a << 24);
  
//    if (bla < 1000)
//      std::cout <<"Index: " <<bla <<" dist: " <<std::setprecision(10) <<distVal <<" color: " <<(int)r <<" " <<(int)g <<" " <<(int)b <<"\n";

    cl_uint vertIndex0 = mergedIBO[bla*3 +0];
    cl_uint vertIndex1 = mergedIBO[bla*3 +1];
    cl_uint vertIndex2 = mergedIBO[bla*3 +2];

    if (vertIndex0 >= m_TotalNumOfTranslucentVertices)
    {
      MITK_INFO <<"vertIndex0: " <<vertIndex0 <<" Total num of vertices: " <<m_TotalNumOfTranslucentVertices;
      break;
    }

    if (vertIndex1 >= m_TotalNumOfTranslucentVertices)
    {
      MITK_INFO <<"vertIndex1: " <<vertIndex1 <<" Total num of vertices: " <<m_TotalNumOfTranslucentVertices;
      break;
    }

    if (vertIndex2 >= m_TotalNumOfTranslucentVertices)
    {
      MITK_INFO <<"vertIndex2: " <<vertIndex2 <<" Total num of vertices: " <<m_TotalNumOfTranslucentVertices;
      break;
    }

    if (distVal > vertDistData[vertIndex0])
    {
      vertDistData[vertIndex0] = distVal;
      colorData[vertIndex0]    = colorVal;
    }

    if (distVal > vertDistData[vertIndex1])
    {
      vertDistData[vertIndex1] = distVal;
      colorData[vertIndex1]    = colorVal;
    }

    if (distVal > vertDistData[vertIndex2])
    {
      vertDistData[vertIndex2] = distVal;
      colorData[vertIndex2]    = colorVal;
    }
  }

  vlColors->bufferObject()->setBufferSubData(0, m_TotalNumOfTranslucentVertices*sizeof(unsigned int), colorData);

  delete mergedIBO;
  delete mergedDistances;
  delete colorData;
  delete vertDistData;

  clReleaseMemObject(mergedDistBufOutput);
  mergedDistBufOutput = 0;
*/

  return true;
}


//-----------------------------------------------------------------------------
void VLQtWidget::setContinuousUpdate(bool continuous)
{
  vl::OpenGLContext::setContinuousUpdate(continuous);

  if (continuous)
  {
    disconnect(&m_UpdateTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
    connect(&m_UpdateTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
    m_UpdateTimer.setSingleShot(false);
    m_UpdateTimer.setInterval(m_Refresh);
    m_UpdateTimer.start(0);
  }
  else
  {
    disconnect(&m_UpdateTimer, SIGNAL(timeout()), this, SLOT(updateGL()));
    m_UpdateTimer.stop();
  }
}


//-----------------------------------------------------------------------------
void VLQtWidget::setWindowTitle(const vl::String& title)
{
  QGLWidget::setWindowTitle( QString::fromStdString(title.toStdString()) );
}


//-----------------------------------------------------------------------------
bool VLQtWidget::setFullscreen(bool fullscreen)
{
  // fullscreen not allowed (yet)!
  fullscreen = false;

  mFullscreen = fullscreen;
  if (fullscreen)
    QGLWidget::setWindowState(QGLWidget::windowState() | Qt::WindowFullScreen);
  else
    QGLWidget::setWindowState(QGLWidget::windowState() & (~Qt::WindowFullScreen));
  return true;
}


//-----------------------------------------------------------------------------
void VLQtWidget::show()
{
  QGLWidget::show();
}


//-----------------------------------------------------------------------------
void VLQtWidget::hide()
{
  QGLWidget::hide();
}


//-----------------------------------------------------------------------------
void VLQtWidget::setPosition(int x, int y)
{
  QGLWidget::move(x,y);
}


//-----------------------------------------------------------------------------
vl::ivec2 VLQtWidget::position() const
{
  return vl::ivec2(QGLWidget::pos().x(), QGLWidget::pos().y());
}


//-----------------------------------------------------------------------------
void VLQtWidget::update()
{
  // schedules a repaint, will eventually call into paintGL()
  QGLWidget::update();
}


//-----------------------------------------------------------------------------
void VLQtWidget::setSize(int w, int h)
{
  // this already excludes the window's frame so it's ok for Visualization Library standards
  QGLWidget::resize(w,h);
}


//-----------------------------------------------------------------------------
vl::ivec2 VLQtWidget::size() const
{
  // this already excludes the window's frame so it's ok for Visualization Library standards
  return vl::ivec2(QGLWidget::size().width(), QGLWidget::size().height());
}


//-----------------------------------------------------------------------------
void VLQtWidget::swapBuffers()
{
  // on windows, swapBuffers() does not depend on the opengl rendering context.
  // instead it is initiated on the device context, which is not implicitly bound to the calling thread.
  niftk::ScopedOGLContext    ctx(context());

  QGLWidget::swapBuffers();

#ifdef _USE_CUDA
  if (m_CUDAInteropPimpl)
  {
    cudaError_t          err         = cudaSuccess;
    niftk::CUDAManager*  cudamanager = niftk::CUDAManager::GetInstance();
    cudaStream_t         mystream    = cudamanager->GetStream(m_CUDAInteropPimpl->m_NodeName);
    niftk::WriteAccessor outputWA    = cudamanager->RequestOutputImage(QWidget::width(), QWidget::height(), 4);
    cudaArray_t          fboarr      = m_CUDAInteropPimpl->m_FBOAdaptor->Map(mystream);

    // side note: cuda-arrays are always measured in bytes, never in pixels.
    err = cudaMemcpy2DFromArrayAsync(outputWA.m_DevicePointer, outputWA.m_BytePitch, fboarr, 0, 0, outputWA.m_PixelWidth * 4, outputWA.m_PixelHeight, cudaMemcpyDeviceToDevice, mystream);
    // not sure what to do if it fails. do not throw an exception, that's for sure.
    if (err != cudaSuccess)
    {
      assert(false);
    }

    // the opengl-interop side is done, renderer can continue from now on.
    m_CUDAInteropPimpl->m_FBOAdaptor->Unmap(mystream);

    // need to flip the image! ogl is left-bottom, but everywhere else is left-top origin!
    niftk::WriteAccessor  flippedWA   = cudamanager->RequestOutputImage(outputWA.m_PixelWidth, outputWA.m_PixelHeight, 4);
    // FIXME: instead of explicitly flipping we could bind the fboarr to a texture, and do a single write out.
    niftk::FlipImageLauncher(outputWA, flippedWA, mystream);

    niftk::LightweightCUDAImage lwciFlipped = cudamanager->Finalise(flippedWA, mystream);
    // Finalise() needs to come before Autorelease(), for performance reasons.
    cudamanager->Autorelease(outputWA, mystream);

    bool    isNewNode = false;
    mitk::DataNode::Pointer node = m_CUDAInteropPimpl->m_DataStorage->GetNamedNode(m_CUDAInteropPimpl->m_NodeName);
    if (node.IsNull())
    {
      isNewNode = true;
      node = mitk::DataNode::New();
      node->SetName(m_CUDAInteropPimpl->m_NodeName);
      node->SetVisibility(false);
      //node->SetBoolProperty("helper object", true);
    }
    niftk::CUDAImage::Pointer  img = dynamic_cast<niftk::CUDAImage*>(node->GetData());
    if (img.IsNull())
      img = niftk::CUDAImage::New();
    img->SetLightweightCUDAImage(lwciFlipped);
    node->SetData(img);
    if (isNewNode)
      m_CUDAInteropPimpl->m_DataStorage->Add(node);
    else
      node->Modified();
  }
#endif
}


//-----------------------------------------------------------------------------
void VLQtWidget::makeCurrent()
{
  QGLWidget::makeCurrent();
  // sanity check
  assert(QGLContext::currentContext() == QGLWidget::context());
}


//-----------------------------------------------------------------------------
void VLQtWidget::setMousePosition(int x, int y)
{
  QCursor::setPos(mapToGlobal(QPoint(x,y)));
}


//-----------------------------------------------------------------------------
void VLQtWidget::setMouseVisible(bool visible)
{
  vl::OpenGLContext::setMouseVisible(visible);

  if (visible)
    QGLWidget::setCursor(Qt::ArrowCursor);
  else
    QGLWidget::setCursor(Qt::BlankCursor);
}


//-----------------------------------------------------------------------------
void VLQtWidget::getFocus()
{
  QGLWidget::setFocus(Qt::OtherFocusReason);
}


//-----------------------------------------------------------------------------
void VLQtWidget::setRefreshRate(int msec)
{
  m_Refresh = msec;
  m_UpdateTimer.setInterval(m_Refresh);
}


//-----------------------------------------------------------------------------
int VLQtWidget::refreshRate()
{
  return m_Refresh;
}


#if 0
//-----------------------------------------------------------------------------
void VLQtWidget::dragEnterEvent(QDragEnterEvent *ev)
{
  if (ev->mimeData()->hasUrls())
    ev->acceptProposedAction();
}


//-----------------------------------------------------------------------------
void VLQtWidget::dropEvent(QDropEvent* ev)
{
  if ( ev->mimeData()->hasUrls() )
  {
    std::vector<vl::String> files;
    QList<QUrl> list = ev->mimeData()->urls();
    for(int i=0; i<list.size(); ++i)
    {
      if (list[i].path().isEmpty())
        continue;
      #ifdef WIN32
        if (list[i].path()[0] == '/')
          files.push_back( list[i].path().toStdString().c_str()+1 );
        else
          files.push_back( list[i].path().toStdString().c_str() );
      #else
        files.push_back( list[i].path().toStdString().c_str() );
      #endif
    }
    dispatchFileDroppedEvent(files);
  }
}
#endif


//-----------------------------------------------------------------------------
void VLQtWidget::mouseMoveEvent(QMouseEvent* ev)
{
  if (!vl::OpenGLContext::mIgnoreNextMouseMoveEvent)
    dispatchMouseMoveEvent(ev->x(), ev->y());
  vl::OpenGLContext::mIgnoreNextMouseMoveEvent = false;
}


//-----------------------------------------------------------------------------
void VLQtWidget::mousePressEvent(QMouseEvent* ev)
{
  vl::EMouseButton bt = vl::NoButton;
  switch(ev->button())
  {
  case Qt::LeftButton:  bt = vl::LeftButton; break;
  case Qt::RightButton: bt = vl::RightButton; break;
  case Qt::MidButton:   bt = vl::MiddleButton; break;
  default:
    bt = vl::UnknownButton; break;
  }
  vl::OpenGLContext::dispatchMouseDownEvent(bt, ev->x(), ev->y());
}


//-----------------------------------------------------------------------------
void VLQtWidget::mouseReleaseEvent(QMouseEvent* ev)
{
  vl::EMouseButton bt = vl::NoButton;
  switch(ev->button())
  {
  case Qt::LeftButton:  bt = vl::LeftButton; break;
  case Qt::RightButton: bt = vl::RightButton; break;
  case Qt::MidButton:   bt = vl::MiddleButton; break;
  default:
    bt = vl::UnknownButton; break;
  }
  vl::OpenGLContext::dispatchMouseUpEvent(bt, ev->x(), ev->y());
}


//-----------------------------------------------------------------------------
void VLQtWidget::wheelEvent(QWheelEvent* ev)
{
  vl::OpenGLContext::dispatchMouseWheelEvent(ev->delta() / 120);
}


//-----------------------------------------------------------------------------
void VLQtWidget::keyPressEvent(QKeyEvent* ev)
{
  unsigned short unicode_ch = 0;
  vl::EKey key = vl::Key_None;
  translateKeyEvent(ev, unicode_ch, key);
  vl::OpenGLContext::dispatchKeyPressEvent(unicode_ch, key);
}


//-----------------------------------------------------------------------------
void VLQtWidget::keyReleaseEvent(QKeyEvent* ev)
{
  unsigned short unicode_ch = 0;
  vl::EKey key = vl::Key_None;
  translateKeyEvent(ev, unicode_ch, key);
  vl::OpenGLContext::dispatchKeyReleaseEvent(unicode_ch, key);
}


//-----------------------------------------------------------------------------
void VLQtWidget::translateKeyEvent(QKeyEvent* ev, unsigned short& unicode_out, vl::EKey& key_out)
{
  // translate non unicode characters
  key_out     = vl::Key_None;
  unicode_out = 0;

  switch(ev->key())
  {
    case Qt::Key_Clear:    key_out = vl::Key_Clear; break;
    case Qt::Key_Control:  key_out = vl::Key_Ctrl; break;
    // case Qt::Key_LCONTROL: key_out = Key_LeftCtrl; break;
    // case Qt::Key_RCONTROL: key_out = Key_RightCtrl; break;
    case Qt::Key_Alt:     key_out = vl::Key_Alt; break;
    // case Qt::Key_LMENU:    key_out = Key_LeftAlt; break;
    // case Qt::Key_RMENU:    key_out = Key_RightAlt; break;
    case Qt::Key_Shift:    key_out = vl::Key_Shift; break;
    // case Qt::Key_LSHIFT:   key_out = Key_LeftShift; break;
    // case Qt::Key_RSHIFT:   key_out = Key_RightShift; break;
    case Qt::Key_Insert:   key_out = vl::Key_Insert; break;
    case Qt::Key_Delete:   key_out = vl::Key_Delete; break;
    case Qt::Key_Home:     key_out = vl::Key_Home; break;
    case Qt::Key_End:      key_out = vl::Key_End; break;
    case Qt::Key_Print:    key_out = vl::Key_Print; break;
    case Qt::Key_Pause:    key_out = vl::Key_Pause; break;
    case Qt::Key_PageUp:   key_out = vl::Key_PageUp; break;
    case Qt::Key_PageDown: key_out = vl::Key_PageDown; break;
    case Qt::Key_Left:     key_out = vl::Key_Left; break;
    case Qt::Key_Right:    key_out = vl::Key_Right; break;
    case Qt::Key_Up:       key_out = vl::Key_Up; break;
    case Qt::Key_Down:     key_out = vl::Key_Down; break;
    case Qt::Key_F1:       key_out = vl::Key_F1; break;
    case Qt::Key_F2:       key_out = vl::Key_F2; break;
    case Qt::Key_F3:       key_out = vl::Key_F3; break;
    case Qt::Key_F4:       key_out = vl::Key_F4; break;
    case Qt::Key_F5:       key_out = vl::Key_F5; break;
    case Qt::Key_F6:       key_out = vl::Key_F6; break;
    case Qt::Key_F7:       key_out = vl::Key_F7; break;
    case Qt::Key_F8:       key_out = vl::Key_F8; break;
    case Qt::Key_F9:       key_out = vl::Key_F9; break;
    case Qt::Key_F10:      key_out = vl::Key_F10; break;
    case Qt::Key_F11:      key_out = vl::Key_F11; break;
    case Qt::Key_F12:      key_out = vl::Key_F12; break;

    case Qt::Key_0: key_out = vl::Key_0; break;
    case Qt::Key_1: key_out = vl::Key_1; break;
    case Qt::Key_2: key_out = vl::Key_2; break;
    case Qt::Key_3: key_out = vl::Key_3; break;
    case Qt::Key_4: key_out = vl::Key_4; break;
    case Qt::Key_5: key_out = vl::Key_5; break;
    case Qt::Key_6: key_out = vl::Key_6; break;
    case Qt::Key_7: key_out = vl::Key_7; break;
    case Qt::Key_8: key_out = vl::Key_8; break;
    case Qt::Key_9: key_out = vl::Key_9; break;

    case Qt::Key_A: key_out = vl::Key_A; break;
    case Qt::Key_B: key_out = vl::Key_B; break;
    case Qt::Key_C: key_out = vl::Key_C; break;
    case Qt::Key_D: key_out = vl::Key_D; break;
    case Qt::Key_E: key_out = vl::Key_E; break;
    case Qt::Key_F: key_out = vl::Key_F; break;
    case Qt::Key_G: key_out = vl::Key_G; break;
    case Qt::Key_H: key_out = vl::Key_H; break;
    case Qt::Key_I: key_out = vl::Key_I; break;
    case Qt::Key_J: key_out = vl::Key_J; break;
    case Qt::Key_K: key_out = vl::Key_K; break;
    case Qt::Key_L: key_out = vl::Key_L; break;
    case Qt::Key_M: key_out = vl::Key_M; break;
    case Qt::Key_N: key_out = vl::Key_N; break;
    case Qt::Key_O: key_out = vl::Key_O; break;
    case Qt::Key_P: key_out = vl::Key_P; break;
    case Qt::Key_Q: key_out = vl::Key_Q; break;
    case Qt::Key_R: key_out = vl::Key_R; break;
    case Qt::Key_S: key_out = vl::Key_S; break;
    case Qt::Key_T: key_out = vl::Key_T; break;
    case Qt::Key_U: key_out = vl::Key_U; break;
    case Qt::Key_V: key_out = vl::Key_V; break;
    case Qt::Key_W: key_out = vl::Key_W; break;
    case Qt::Key_X: key_out = vl::Key_X; break;
    case Qt::Key_Y: key_out = vl::Key_Y; break;
    case Qt::Key_Z: key_out = vl::Key_Z; break;
  }

  // fill unicode
  QString ustring = ev->text();
  if ( ustring.length() == 1 )
  {
    unicode_out = ustring[0].unicode();

    // fill key
    switch(unicode_out)
    {
      case L'0': key_out = vl::Key_0; break;
      case L'1': key_out = vl::Key_1; break;
      case L'2': key_out = vl::Key_2; break;
      case L'3': key_out = vl::Key_3; break;
      case L'4': key_out = vl::Key_4; break;
      case L'5': key_out = vl::Key_5; break;
      case L'6': key_out = vl::Key_6; break;
      case L'7': key_out = vl::Key_7; break;
      case L'8': key_out = vl::Key_8; break;
      case L'9': key_out = vl::Key_9; break;

      case L'A': key_out = vl::Key_A; break;
      case L'B': key_out = vl::Key_B; break;
      case L'C': key_out = vl::Key_C; break;
      case L'D': key_out = vl::Key_D; break;
      case L'E': key_out = vl::Key_E; break;
      case L'F': key_out = vl::Key_F; break;
      case L'G': key_out = vl::Key_G; break;
      case L'H': key_out = vl::Key_H; break;
      case L'I': key_out = vl::Key_I; break;
      case L'J': key_out = vl::Key_J; break;
      case L'K': key_out = vl::Key_K; break;
      case L'L': key_out = vl::Key_L; break;
      case L'M': key_out = vl::Key_M; break;
      case L'N': key_out = vl::Key_N; break;
      case L'O': key_out = vl::Key_O; break;
      case L'P': key_out = vl::Key_P; break;
      case L'Q': key_out = vl::Key_Q; break;
      case L'R': key_out = vl::Key_R; break;
      case L'S': key_out = vl::Key_S; break;
      case L'T': key_out = vl::Key_T; break;
      case L'U': key_out = vl::Key_U; break;
      case L'V': key_out = vl::Key_V; break;
      case L'W': key_out = vl::Key_W; break;
      case L'X': key_out = vl::Key_X; break;
      case L'Y': key_out = vl::Key_Y; break;
      case L'Z': key_out = vl::Key_Z; break;

      case L'a': key_out = vl::Key_A; break;
      case L'b': key_out = vl::Key_B; break;
      case L'c': key_out = vl::Key_C; break;
      case L'd': key_out = vl::Key_D; break;
      case L'e': key_out = vl::Key_E; break;
      case L'f': key_out = vl::Key_F; break;
      case L'g': key_out = vl::Key_G; break;
      case L'h': key_out = vl::Key_H; break;
      case L'i': key_out = vl::Key_I; break;
      case L'j': key_out = vl::Key_J; break;
      case L'k': key_out = vl::Key_K; break;
      case L'l': key_out = vl::Key_L; break;
      case L'm': key_out = vl::Key_M; break;
      case L'n': key_out = vl::Key_N; break;
      case L'o': key_out = vl::Key_O; break;
      case L'p': key_out = vl::Key_P; break;
      case L'q': key_out = vl::Key_Q; break;
      case L'r': key_out = vl::Key_R; break;
      case L's': key_out = vl::Key_S; break;
      case L't': key_out = vl::Key_T; break;
      case L'u': key_out = vl::Key_U; break;
      case L'v': key_out = vl::Key_V; break;
      case L'w': key_out = vl::Key_W; break;
      case L'x': key_out = vl::Key_X; break;
      case L'y': key_out = vl::Key_Y; break;
      case L'z': key_out = vl::Key_Z; break;

      case 13: key_out = vl::Key_Return; break;
      case 8: key_out = vl::Key_BackSpace; break;
      case 9: key_out = vl::Key_Tab; break;
      case L' ': key_out = vl::Key_Space; break;

      case 27: key_out = vl::Key_Escape; break;
      case L'!': key_out = vl::Key_Exclam; break;
      case L'"': key_out = vl::Key_QuoteDbl; break;
      case L'#': key_out = vl::Key_Hash; break;
      case L'$': key_out = vl::Key_Dollar; break;
      case L'&': key_out = vl::Key_Ampersand; break;
      case L'\'': key_out = vl::Key_Quote; break;
      case L'(': key_out = vl::Key_LeftParen; break;
      case L')': key_out = vl::Key_RightParen; break;
      case L'*': key_out = vl::Key_Asterisk; break;
      case L'+': key_out = vl::Key_Plus; break;
      case L',': key_out = vl::Key_Comma; break;
      case L'-': key_out = vl::Key_Minus; break;
      case L'.': key_out = vl::Key_Period; break;
      case L'\\': key_out = vl::Key_Slash; break;
      case L':': key_out = vl::Key_Colon; break;
      case L';': key_out = vl::Key_Semicolon; break;
      case L'<': key_out = vl::Key_Less; break;
      case L'=': key_out = vl::Key_Equal; break;
      case L'>': key_out = vl::Key_Greater; break;
      case L'?': key_out = vl::Key_Question; break;
      case L'@': key_out = vl::Key_At; break;
      case L'[': key_out = vl::Key_LeftBracket; break;
      case L'/': key_out = vl::Key_BackSlash; break;
      case L']': key_out = vl::Key_RightBracket; break;
      case L'|': key_out = vl::Key_Caret; break;
      case L'_': key_out = vl::Key_Underscore; break;
      case L'`': key_out = vl::Key_QuoteLeft; break;
    }
  }
}


//-----------------------------------------------------------------------------
QGLContext* VLQtWidget::context()
{
  return const_cast<QGLContext*>(QGLWidget::context());
}