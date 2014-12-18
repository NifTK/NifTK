/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


#include "VLQt4Widget.h"
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
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorage.h>
#include <QFile>
#include <QTextStream>
#include <stdexcept>
#include <sstream>
#include "ScopedOGLContext.h"

#ifdef _USE_CUDA
#include <Rendering/VLFramebufferToCUDA.h>
#include <CUDAManager/CUDAManager.h>
#include <CUDAImage/CUDAImage.h>
#include <CUDAImage/LightweightCUDAImage.h>
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
VLQt4Widget::TextureDataPOD::TextureDataPOD()
  : m_LastUpdatedID(0)
  , m_CUDARes(0)
{
}


#else
// empty dummy, in case we have no cuda
struct CUDAInterop { };
#endif // _USE_CUDA


//-----------------------------------------------------------------------------
VLQt4Widget::VLQt4Widget(QWidget* parent, const QGLWidget* shareWidget, Qt::WindowFlags f)
  : QGLWidget(parent, shareWidget, f)
  , m_Refresh(10) // 100 fps
  , m_OclService(0)
  , m_CUDAInteropPimpl(0)
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
VLQt4Widget::~VLQt4Widget()
{
  ScopedOGLContext  ctx(this->context());

  dispatchDestroyEvent();

  delete m_CUDAInteropPimpl;
}


#if 0
//-----------------------------------------------------------------------------
bool VLQt4Widget::initQt4Widget(const vl::String& title/*, const vl::OpenGLContextFormat& info, const QGLContext* shareContext=0*/, int x=0, int y=0, int width=640, int height=480)
{
#if 0
  // setFormat(fmt) is marked as deprecated so we use this other method
  QGLContext* glctx = new QGLContext(context()->format(), this);
  QGLFormat fmt = context()->format();

  // double buffer
  fmt.setDoubleBuffer( info.doubleBuffer() );

  // color buffer
  fmt.setRedBufferSize( info.rgbaBits().r() );
  fmt.setGreenBufferSize( info.rgbaBits().g() );
  fmt.setBlueBufferSize( info.rgbaBits().b() );
  // setAlpha == true makes the create() function alway fail
  // even if the returned format has the requested alpha channel
  fmt.setAlphaBufferSize( info.rgbaBits().a() );
  fmt.setAlpha( info.rgbaBits().a() != 0 );

  // accumulation buffer
  int accum = vl::max( info.accumRGBABits().r(), info.accumRGBABits().g() );
  accum = vl::max( accum, info.accumRGBABits().b() );
  accum = vl::max( accum, info.accumRGBABits().a() );
  fmt.setAccumBufferSize( accum );
  fmt.setAccum( accum != 0 );

  // multisampling
  if (info.multisample())
    fmt.setSamples( info.multisampleSamples() );
  fmt.setSampleBuffers( info.multisample() );

  // depth buffer
  fmt.setDepthBufferSize( info.depthBufferBits() );
  fmt.setDepth( info.depthBufferBits() != 0 );

  // stencil buffer
  fmt.setStencilBufferSize( info.stencilBufferBits() );
  fmt.setStencil( info.stencilBufferBits() != 0 );

  // stereo
  fmt.setStereo( info.stereo() );

  // swap interval / v-sync
  fmt.setSwapInterval( info.vSync() ? 1 : 0 );

  glctx->setFormat(fmt);
  // this function returns false when we request an alpha buffer
  // even if the created context seem to have the alpha buffer
  /*bool ok = */glctx->create(shareContext);
  setContext(glctx);
#endif

  initGLContext();

  framebuffer()->setWidth(width);
  framebuffer()->setHeight(height);

#if 0//ndef NDEBUG
  printf("--------------------------------------------\n");
  printf("REQUESTED OpenGL Format:\n");
  printf("--------------------------------------------\n");
  printf("rgba = %d %d %d %d\n", fmt.redBufferSize(), fmt.greenBufferSize(), fmt.blueBufferSize(), fmt.alphaBufferSize() );
  printf("double buffer = %d\n", (int)fmt.doubleBuffer() );
  printf("depth buffer size = %d\n", fmt.depthBufferSize() );
  printf("depth buffer = %d\n", fmt.depth() );
  printf("stencil buffer size = %d\n", fmt.stencilBufferSize() );
  printf("stencil buffer = %d\n", fmt.stencil() );
  printf("accum buffer size %d\n", fmt.accumBufferSize() );
  printf("accum buffer %d\n", fmt.accum() );
  printf("stereo = %d\n", (int)fmt.stereo() );
  printf("swap interval = %d\n", fmt.swapInterval() );
  printf("multisample = %d\n", (int)fmt.sampleBuffers() );
  printf("multisample samples = %d\n", (int)fmt.samples() );

  fmt = format();

  printf("--------------------------------------------\n");
  printf("OBTAINED OpenGL Format:\n");
  printf("--------------------------------------------\n");
  printf("rgba = %d %d %d %d\n", fmt.redBufferSize(), fmt.greenBufferSize(), fmt.blueBufferSize(), fmt.alphaBufferSize() );
  printf("double buffer = %d\n", (int)fmt.doubleBuffer() );
  printf("depth buffer size = %d\n", fmt.depthBufferSize() );
  printf("depth buffer = %d\n", fmt.depth() );
  printf("stencil buffer size = %d\n", fmt.stencilBufferSize() );
  printf("stencil buffer = %d\n", fmt.stencil() );
  printf("accum buffer size %d\n", fmt.accumBufferSize() );
  printf("accum buffer %d\n", fmt.accum() );
  printf("stereo = %d\n", (int)fmt.stereo() );
  printf("swap interval = %d\n", fmt.swapInterval() );
  printf("multisample = %d\n", (int)fmt.sampleBuffers() );
  printf("multisample samples = %d\n", (int)fmt.samples() );
  printf("--------------------------------------------\n");
#endif

  setWindowTitle(title);
  move(x,y);
  resize(width,height);

#if 0
  if (info.fullscreen())
    setFullscreen(true);
#endif

  return true;
}
#endif


//-----------------------------------------------------------------------------
void VLQt4Widget::setOclResourceService(OclResourceService* oclserv)
{
  m_OclService = oclserv;
}


//-----------------------------------------------------------------------------
vl::FramebufferObject* VLQt4Widget::GetFBO()
{
  // createAndUpdateFBOSizes() where we always stuff a proper fbo into the blit.
  return dynamic_cast<vl::FramebufferObject*>(m_FinalBlit->readFramebuffer());
}


//-----------------------------------------------------------------------------
void VLQt4Widget::EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage, const std::string& nodename)
{
#ifdef _USE_CUDA
  ScopedOGLContext  ctx(this->context());

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
void VLQt4Widget::initializeGL()
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());

  vl::OpenGLContext::initGLContext();

  // use the device that is running our opengl context as the compute-device
  // for sorting triangles in the correct order.
  if (m_OclService)
  {
    m_OclService->SpecifyPlatformAndDevice(0, 0, true);
  }


  m_Camera = new vl::Camera;
  vl::vec3 eye    = vl::vec3(0,10,35);
  vl::vec3 center = vl::vec3(0,0,0);
  vl::vec3 up     = vl::vec3(0,1,0);
  vl::mat4 view_mat = vl::mat4::getLookAt(eye, center, up);
  m_Camera->setViewMatrix(view_mat);

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

  // opaque objects dont need any sorting (in theory).
  // but they have to happen before anything else.
  m_OpaqueObjectsRendering = new vl::Rendering;
  m_OpaqueObjectsRendering->setEnableMask(ENABLEMASK_OPAQUE | ENABLEMASK_TRANSLUCENT);
  m_OpaqueObjectsRendering->setObjectName("m_OpaqueObjectsRendering");
  m_OpaqueObjectsRendering->setCamera(m_Camera.get());
  m_OpaqueObjectsRendering->sceneManagers()->push_back(m_SceneManager.get());
  // we sort them anyway, front-to-back so that early-fragment rejection can work its magic.
  m_OpaqueObjectsRendering->setRenderQueueSorter(new vl::RenderQueueSorterAggressive);

  // volume rendering is a separate stage, after opaque.
  // it needs access to the depth-buffer of the opaque geometry so that raycast can clip properly.
  m_VolumeRendering = new vl::Rendering;
  m_VolumeRendering->setEnableMask(ENABLEMASK_VOLUME);
  m_VolumeRendering->setObjectName("m_VolumeRendering");
  m_VolumeRendering->setCamera(m_Camera.get());
  m_VolumeRendering->sceneManagers()->push_back(m_SceneManager.get());
  // FIXME: only single volume supported for now, so no queue sorting.

  m_RenderingTree = new vl::RenderingTree;
  m_RenderingTree->setObjectName("m_RenderingTree");
  m_RenderingTree->subRenderings()->push_back(m_OpaqueObjectsRendering.get());
  //m_RenderingTree->subRenderings()->push_back(m_VolumeRendering.get());

  // once rendering to fbo has finished, blit it to the screen's backbuffer.
  // a final swapbuffers in renderScene() and/or paintGL() will show it on screen.
  m_FinalBlit = new vl::BlitFramebuffer;
  m_FinalBlit->setObjectName("m_FinalBlit");
  m_FinalBlit->setLinearFilteringEnabled(false);
  m_FinalBlit->setBufferMask(vl::BB_COLOR_BUFFER_BIT | vl::BB_DEPTH_BUFFER_BIT);
  m_FinalBlit->setDrawFramebuffer(vl::OpenGLContext::framebuffer());
  m_RenderingTree->onFinishedCallbacks()->push_back(m_FinalBlit.get());

  // updating the size of our fbo is a bit of a pain.
  createAndUpdateFBOSizes(QGLWidget::width(), QGLWidget::height());

  m_Camera->viewport()->setClearColor(vl::fuchsia);

  // ???
  m_OpaqueObjectsRendering->transform()->addChild(m_LightTr.get());


  m_Trackball = new vl::TrackballManipulator;
  m_Trackball->setEnabled(true);
  m_Trackball->setCamera(m_Camera.get());
  m_Trackball->setTransform(NULL);
  m_Trackball->setPivot(vl::vec3(0,0,0));
  vl::OpenGLContext::addEventListener(m_Trackball.get());

  m_ThresholdVal = new vl::Uniform("val_threshold");
  m_ThresholdVal->setUniformF(0.5f);

  vl::OpenGLContext::dispatchInitEvent();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::createAndUpdateFBOSizes(int width, int height)
{
  // sanity check: internal method, context should have been activated by caller.
  assert(this->context() == QGLContext::currentContext());

  vl::ref<vl::FramebufferObject> opaqueFBO = vl::OpenGLContext::createFramebufferObject(width, height);
  opaqueFBO->setObjectName("opaqueFBO");
  opaqueFBO->addDepthAttachment(new vl::FBODepthBufferAttachment(vl::DBF_DEPTH_COMPONENT24));
  opaqueFBO->addColorAttachment(vl::AP_COLOR_ATTACHMENT0, new vl::FBOColorBufferAttachment(vl::CBF_RGBA));   // this is a renderbuffer
  opaqueFBO->setDrawBuffer(vl::RDB_COLOR_ATTACHMENT0);

  m_OpaqueObjectsRendering->renderer()->setFramebuffer(opaqueFBO.get());

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
void VLQt4Widget::resizeGL(int width, int height)
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());


  // dont do anything if window is zero size.
  // it's an opengl error to have a viewport like that!
  if ((width <= 0) || (height <= 0))
    return;

  framebuffer()->setWidth(width);
  framebuffer()->setHeight(height);
  m_OpaqueObjectsRendering->renderer()->framebuffer()->setWidth(width);
  m_OpaqueObjectsRendering->renderer()->framebuffer()->setHeight(height);

  createAndUpdateFBOSizes(width, height);

  //m_VolumeRendering->renderer()->framebuffer()->setWidth(width);
  //m_VolumeRendering->renderer()->framebuffer()->setHeight(height);

  m_FinalBlit->setSrcRect(0, 0, width, height);
  m_FinalBlit->setDstRect(0, 0, width, height);

  m_Camera->viewport()->setWidth(width);
  m_Camera->viewport()->setHeight(height);
  m_Camera->setProjectionPerspective();

  vl::OpenGLContext::dispatchResizeEvent(width, height);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::paintGL()
{
  // sanity check: context is initialised by Qt
  assert(this->context() == QGLContext::currentContext());

  renderScene();

  vl::OpenGLContext::dispatchRunEvent();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::renderScene()
{
  // caller of paintGL() (i.e. Qt's internals) should have activated our context!
  assert(this->context() == QGLContext::currentContext());


  // update scene graph.
  vl::mat4 cameraMatrix = m_Camera->modelingMatrix();
  m_LightTr->setLocalMatrix(cameraMatrix);


  // trigger execution of the renderer(s).
  vl::real now_time = vl::Time::currentTime();
  m_RenderingTree->setFrameClock(now_time);
  m_RenderingTree->render();

  if (vl::OpenGLContext::hasDoubleBuffer())
    swapBuffers();

  VL_CHECK_OGL();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::ClearScene()
{
  ScopedOGLContext  ctx(context());

  m_SceneManager->tree()->actors()->clear();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::UpdateThresholdVal(int isoVal)
{
  ScopedOGLContext    ctx(context());

  float val_threshold = 0.0f;
  m_ThresholdVal->getUniform(&val_threshold);

  val_threshold = isoVal / 10000.0f;
  val_threshold = vl::clamp(val_threshold, 0.0f, 1.0f);

  m_ThresholdVal->setUniformF(val_threshold);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::AddDataNode(const mitk::DataNode::Pointer& node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  // Propagate color and opacity down to basedata
  node->GetData()->SetProperty("color", node->GetProperty("color"));
  node->GetData()->SetProperty("opacity", node->GetProperty("opacity"));
  node->GetData()->SetProperty("visible", node->GetProperty("visible"));

  mitk::Image::Pointer    mitkImg   = dynamic_cast<mitk::Image*>(node->GetData());
  mitk::Surface::Pointer  mitkSurf  = dynamic_cast<mitk::Surface*>(node->GetData());
#ifdef _USE_CUDA
  CUDAImage::Pointer      cudaImg   = dynamic_cast<CUDAImage*>(node->GetData());
#endif


  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  ScopedOGLContext    ctx(context());


  vl::ref<vl::Actor>    newActor;
  std::string           namePostFix;
  if (mitkImg.IsNotNull())
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
}


//-----------------------------------------------------------------------------
void VLQt4Widget::UpdateDataNode(const mitk::DataNode::Pointer& node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  std::map< mitk::DataNode::Pointer, vl::ref<vl::Actor> >::iterator     it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;

  vl::ref<vl::Actor>    vlActor = it->second;
  if (vlActor.get() == 0)
    return;


  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  ScopedOGLContext    ctx(context());


  mitk::BoolProperty* visibleProp = dynamic_cast<mitk::BoolProperty*>(node->GetProperty("visible"));
  bool  isVisble = visibleProp->GetValue();

  mitk::FloatProperty* opacityProp = dynamic_cast<mitk::FloatProperty*>(node->GetProperty("opacity"));
  float opacity = opacityProp->GetValue();
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

    mitk::ColorProperty* colorProp = dynamic_cast<mitk::ColorProperty*>(node->GetProperty("color"));
    mitk::Color mitkColor = colorProp->GetColor();

    vl::fvec4 color;
    color[0] = mitkColor[0];
    color[1] = mitkColor[1];
    color[2] = mitkColor[2];
    color[3] = opacity;


    vl::ref<vl::Effect> fx = vlActor->effect();
    fx->shader()->enable(vl::EN_DEPTH_TEST);
    fx->shader()->enable(vl::EN_LIGHTING);
    fx->shader()->setRenderState(m_Light.get(), 0 );
    fx->shader()->gocMaterial()->setDiffuse(color);

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
        fx->shader()->gocMaterial()->setTransparency(opacity);
      }
      else
      {
        vlActor->setRenderBlock(RENDERBLOCK_OPAQUE);
        vlActor->setEnableMask(ENABLEMASK_OPAQUE);
        fx->shader()->disable(vl::EN_BLEND);
        fx->shader()->enable(vl::EN_CULL_FACE);
        fx->shader()->gocMaterial()->setTransparency(1);
      }
    }

#ifdef _USE_CUDA
    bool    isCUDAImage = false;
    CUDAImage::Pointer cudaimg = dynamic_cast<CUDAImage*>(node->GetData());
    if (cudaimg.IsNotNull())
    {
      isCUDAImage = cudaimg->GetLightweightCUDAImage().GetId() != 0;
    }
    if (isCUDAImage)
    {
      // whatever we had cached from a previous frame.
      TextureDataPOD          texpod    = m_NodeToTextureMap[node];
      LightweightCUDAImage    cudaImage = cudaimg->GetLightweightCUDAImage();
      // only need to update the vl texture, if content in our cuda buffer has changed.
      // and the cuda buffer can change only when we have a different id.
      if (texpod.m_LastUpdatedID != cudaImage.GetId())
      {
        cudaError_t   err = cudaSuccess;
        bool          neednewvltexture = texpod.m_Texture.get() == 0;

        // FIXME: check if vl-texture size needs to change

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

          texpod.m_Texture = new vl::Texture(cudaImage.GetWidth(), cudaImage.GetHeight(), vl::TF_RGBA8, false);

          assert(m_NodeToActorMap.find(node) != m_NodeToActorMap.end());
          m_NodeToActorMap[node]->effect()->shader()->gocTextureSampler(0)->setTexture(texpod.m_Texture.get());


          err = cudaGraphicsGLRegisterImage(&texpod.m_CUDARes, texpod.m_Texture->handle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
          if (err != cudaSuccess)
          {
            texpod.m_CUDARes = 0;
            MITK_WARN << "Registering VL texture into CUDA failed. Will not update (properly).";
          }
        }

        // FIXME: map vl texture


        // everything good, update cache.
        m_NodeToTextureMap[node] = texpod;
      }
    }
#endif
  }
}


//-----------------------------------------------------------------------------
void VLQt4Widget::RemoveDataNode(const mitk::DataNode::Pointer& node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  std::map<mitk::DataNode::Pointer, vl::ref<vl::Actor> >::iterator    it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;

  vl::ref<vl::Actor>    vlActor = it->second;
  if (vlActor.get() == 0)
    return;


  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  ScopedOGLContext    ctx(context());


  m_SceneManager->tree()->eraseActor(vlActor.get());
  m_NodeToActorMap.erase(it);
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQt4Widget::AddSurfaceActor(const mitk::Surface::Pointer& mitkSurf)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());


  vl::ref<vl::Geometry>  vlSurf = new vl::Geometry();
  ConvertVTKPolyData(mitkSurf->GetVtkPolyData(), vlSurf);

  MITK_INFO <<"Num of vertices: " << vlSurf->vertexArray()->size();
  //ArrayAbstract* posarr = vertexArray() ? vertexArray() : vertexAttribArray(vl::VA_Position) ? vertexAttribArray(vl::VA_Position)->data() : NULL;
  if (!vlSurf->normalArray())
    vlSurf->computeNormals();

  vtkSmartPointer<vtkMatrix4x4> geometryTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  mitkSurf->GetGeometry()->GetVtkTransform()->GetMatrix(geometryTransformMatrix);

  vl::mat4  mat;
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double val = geometryTransformMatrix->GetElement(i, j);
      mat.e(i, j) = val;
    }
  }

  vl::ref<vl::Transform> tr     = new vl::Transform();
  tr->setLocalMatrix(mat);

  vl::ref<vl::Effect>    fx = new vl::Effect;
  // UpdateDataNode() takes care of assigning colour etc.

  vl::ref<vl::Actor>    surfActor = m_SceneManager->tree()->addActor(vlSurf.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[surfActor] = vlSurf;

  // FIXME: should go somewhere else
  m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);

  return surfActor;
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQt4Widget::AddCUDAImageActor(const CUDAImage* cudaImg)
{
  // beware: vl does not draw a clean boundary between what is client and what is server side state.
  // so we always need our opengl context current.
  // internal method, so sanity check.
  assert(QGLContext::currentContext() == QGLWidget::context());

#ifdef _USE_CUDA
  vtkSmartPointer<vtkMatrix4x4> geometryTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  cudaImg->GetGeometry()->GetVtkTransform()->GetMatrix(geometryTransformMatrix);
  vl::mat4  mat;
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double val = geometryTransformMatrix->GetElement(i, j);
      mat.e(i, j) = val;
    }
  }
  vl::ref<vl::Transform> tr     = new vl::Transform();
  tr->setLocalMatrix(mat);


  //vl::ref<vl::ArrayFloat3>      vlVerts   = new vl::ArrayFloat3;
  //vl::ref<vl::DrawElementsUInt> vlde      = new vl::DrawElementsUInt(vl::PT_TRIANGLES);//,0,numOfTriangles*3);
  vl::ref<vl::Geometry>         vlquad    = //new vl::Geometry();
    vl::makeGrid(vl::vec3(0, 0, 0), 1, 1, 1, 1, true);
  //vlquad->setVertexArray(vlVerts.get());

  vl::ref<vl::Effect>    fx = new vl::Effect;
  // UpdateDataNode() takes care of assigning colour etc.

  vl::ref<vl::Actor>    actor = m_SceneManager->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  m_ActorToRenderableMap[actor] = vlquad;

  // FIXME: should go somewhere else
  m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);

  return actor;

#else
  throw std::runtime_error("No CUDA-support enabled at compile time!");
#endif
}


//-----------------------------------------------------------------------------
void VLQt4Widget::ConvertVTKPolyData(vtkPolyData* vtkPoly, vl::ref<vl::Geometry> vlPoly)
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

  if (verts->GetMaxCellSize() > 4)
  {
    // Panic and return
    MITK_ERROR <<"More than four vertices / cell detected, can't handle this data type!\n";
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

  //vlVerts->resize(numOfPoints *3);
  //vlNormals->resize(numOfPoints *3);
  //ref<DrawArrays> de = new DrawArrays(PT_TRIANGLES,0,numOfPoints*3);

  vlVerts->resize(numOfTriangles *3);
  vlNormals->resize(numOfTriangles *3);
  vl::ref<vl::DrawArrays> de = new vl::DrawArrays(vl::PT_TRIANGLES,0,numOfTriangles*3);
   
  vlPoly->drawCalls()->push_back(de.get());
  vlPoly->setVertexArray(vlVerts.get());
  vlPoly->setNormalArray(vlNormals.get());

/*
    // read triangles
  for(unsigned int i=0; i<numOfPoints; ++i)
  {
    fvec3 n0, n1, n2, v1,v2,v0;
    n0.x() = m_NormalBuffer[i*3 +0];
    n0.y() = m_NormalBuffer[i*3 +1];
    n0.z() = m_NormalBuffer[i*3 +2];
    v0.x() = m_PointBuffer[i*3 +0];
    v0.y() = m_PointBuffer[i*3 +1];
    v0.z() = m_PointBuffer[i*3 +2];

    vlNormals->at(i*3+0) = n0;
    vlVerts->at(i*3+0) = v0;
  }
*/

  // read triangles
  for(unsigned int i=0; i<numOfTriangles; ++i)
  {
    vl::fvec3 n0, n1, n2, v1,v2,v0;
    unsigned int vertIndex = m_IndexBuffer[i*3 +0];
    n0.x() = m_NormalBuffer[vertIndex*3 +0];
    n0.y() = m_NormalBuffer[vertIndex*3 +1];
    n0.z() = m_NormalBuffer[vertIndex*3 +2];
    v0.x() = m_PointBuffer[vertIndex*3 +0];
    v0.y() = m_PointBuffer[vertIndex*3 +1];
    v0.z() = m_PointBuffer[vertIndex*3 +2];

    vertIndex = m_IndexBuffer[i*3 +1];
    n1.x() = m_NormalBuffer[vertIndex*3 +0];
    n1.y() = m_NormalBuffer[vertIndex*3 +1];
    n1.z() = m_NormalBuffer[vertIndex*3 +2];
    v1.x() = m_PointBuffer[vertIndex*3 +0];
    v1.y() = m_PointBuffer[vertIndex*3 +1];
    v1.z() = m_PointBuffer[vertIndex*3 +2];

    vertIndex = m_IndexBuffer[i*3 +2];
    n2.x() = m_NormalBuffer[vertIndex*3 +0];
    n2.y() = m_NormalBuffer[vertIndex*3 +1];
    n2.z() = m_NormalBuffer[vertIndex*3 +2];
    v2.x() = m_PointBuffer[vertIndex*3 +0];
    v2.y() = m_PointBuffer[vertIndex*3 +1];
    v2.z() = m_PointBuffer[vertIndex*3 +2];

    vlNormals->at(i*3+0) = n0;
    vlVerts->at(i*3+0) = v0;
    vlNormals->at(i*3+1) = n1;
    vlVerts->at(i*3+1) = v1;
    vlNormals->at(i*3+2) = n2;
    vlVerts->at(i*3+2) = v2;
  }

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


  MITK_INFO <<"Num of VL vertices: " <<vlPoly->vertexArray()->size();
}


//-----------------------------------------------------------------------------
vl::ref<vl::Actor> VLQt4Widget::AddImageActor(const mitk::Image::Pointer& mitkImg)
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

    vl::EImageType     typeMap[] =
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

    vl::EImageType     type = typeMap[pixType.GetComponentType()];
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
        format = vl::IF_RGBA_INTEGER;
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
  imageActor->setTransform(tr.get());
  m_SceneManager->tree()->addActor(imageActor.get());

  // this is a callback: gets triggered everytime its bound actor is to be rendered.
  // during that callback it updates the uniforms of our glsl shader to match fixed-function state.
  vl::ref<vl::RaycastVolume>    raycastVolume = new vl::RaycastVolume;
  // this stuffs the proxy geometry onto our actor, as lod-slot zero.
  raycastVolume->bindActor(imageActor.get());


  // we do not own dims!
  unsigned int*   dims    = mitkImg->GetDimensions();
  const float*    spacing = /*const_cast<float*>*/(mitkImg->GetGeometry()->GetFloatSpacing());

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
vl::String VLQt4Widget::LoadGLSLSourceFromResources(const char* filename)
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
void VLQt4Widget::setContinuousUpdate(bool continuous)
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
void VLQt4Widget::setWindowTitle(const vl::String& title)
{
  QGLWidget::setWindowTitle( QString::fromStdString(title.toStdString()) );
}


//-----------------------------------------------------------------------------
bool VLQt4Widget::setFullscreen(bool fullscreen)
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
void VLQt4Widget::show()
{
  QGLWidget::show();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::hide()
{
  QGLWidget::hide();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::setPosition(int x, int y)
{
  QGLWidget::move(x,y);
}


//-----------------------------------------------------------------------------
vl::ivec2 VLQt4Widget::position() const
{
  return vl::ivec2(QGLWidget::pos().x(), QGLWidget::pos().y());
}


//-----------------------------------------------------------------------------
void VLQt4Widget::update()
{
  // schedules a repaint, will eventually call into paintGL()
  QGLWidget::update();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::setSize(int w, int h)
{
  // this already excludes the window's frame so it's ok for Visualization Library standards
  QGLWidget::resize(w,h);
}


//-----------------------------------------------------------------------------
vl::ivec2 VLQt4Widget::size() const
{
  // this already excludes the window's frame so it's ok for Visualization Library standards
  return vl::ivec2(QGLWidget::size().width(), QGLWidget::size().height());
}


//-----------------------------------------------------------------------------
void VLQt4Widget::swapBuffers()
{
  // on windows, swapBuffers() does not depend on the opengl rendering context.
  // instead it is initiated on the device context, which is not implicitly bound to the calling thread.
  ScopedOGLContext    ctx(context());

  QGLWidget::swapBuffers();

#ifdef _USE_CUDA
  if (m_CUDAInteropPimpl)
  {
    cudaError_t     err         = cudaSuccess;
    CUDAManager*    cudamanager = CUDAManager::GetInstance();
    cudaStream_t    mystream    = cudamanager->GetStream(m_CUDAInteropPimpl->m_NodeName);
    WriteAccessor   outputWA    = cudamanager->RequestOutputImage(QWidget::width(), QWidget::height(), 4);
    cudaArray_t     fboarr      = m_CUDAInteropPimpl->m_FBOAdaptor->Map(mystream);

    // side note: cuda-arrays are always measured in bytes, never in pixels.
    err = cudaMemcpyFromArrayAsync(outputWA.m_DevicePointer, fboarr, 0, 0, QWidget::width() * QWidget::height() * 4, cudaMemcpyDeviceToDevice, mystream);
    // not sure what to do if it fails. do not throw and exception, that's for sure.
    assert(err == cudaSuccess);

    m_CUDAInteropPimpl->m_FBOAdaptor->Unmap(mystream);

    LightweightCUDAImage lwci = cudamanager->Finalise(outputWA, mystream);

    bool    isNewNode = false;
    mitk::DataNode::Pointer node = m_CUDAInteropPimpl->m_DataStorage->GetNamedNode(m_CUDAInteropPimpl->m_NodeName);
    if (node.IsNull())
    {
      isNewNode = true;
      node = mitk::DataNode::New();
      node->SetName(m_CUDAInteropPimpl->m_NodeName);
      node->SetVisibility(false);
      node->SetBoolProperty("helper object", true);
    }
    CUDAImage::Pointer  img = dynamic_cast<CUDAImage*>(node->GetData());
    if (img.IsNull())
      img = CUDAImage::New();
    img->SetLightweightCUDAImage(lwci);
    node->SetData(img);
    if (isNewNode)
      m_CUDAInteropPimpl->m_DataStorage->Add(node);
  }
#endif
}


//-----------------------------------------------------------------------------
void VLQt4Widget::makeCurrent()
{
  QGLWidget::makeCurrent();
  // sanity check
  assert(QGLContext::currentContext() == QGLWidget::context());
}


//-----------------------------------------------------------------------------
void VLQt4Widget::setMousePosition(int x, int y)
{
  QCursor::setPos(mapToGlobal(QPoint(x,y)));
}


//-----------------------------------------------------------------------------
void VLQt4Widget::setMouseVisible(bool visible)
{
  vl::OpenGLContext::setMouseVisible(visible);

  if (visible)
    QGLWidget::setCursor(Qt::ArrowCursor);
  else
    QGLWidget::setCursor(Qt::BlankCursor);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::getFocus()
{
  QGLWidget::setFocus(Qt::OtherFocusReason);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::setRefreshRate(int msec)
{
  m_Refresh = msec;
  m_UpdateTimer.setInterval(m_Refresh);
}


//-----------------------------------------------------------------------------
int VLQt4Widget::refreshRate()
{
  return m_Refresh;
}


#if 0
//-----------------------------------------------------------------------------
void VLQt4Widget::dragEnterEvent(QDragEnterEvent *ev)
{
  if (ev->mimeData()->hasUrls())
    ev->acceptProposedAction();
}


//-----------------------------------------------------------------------------
void VLQt4Widget::dropEvent(QDropEvent* ev)
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
void VLQt4Widget::mouseMoveEvent(QMouseEvent* ev)
{
  if (!vl::OpenGLContext::mIgnoreNextMouseMoveEvent)
    dispatchMouseMoveEvent(ev->x(), ev->y());
  vl::OpenGLContext::mIgnoreNextMouseMoveEvent = false;
}


//-----------------------------------------------------------------------------
void VLQt4Widget::mousePressEvent(QMouseEvent* ev)
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
void VLQt4Widget::mouseReleaseEvent(QMouseEvent* ev)
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
void VLQt4Widget::wheelEvent(QWheelEvent* ev)
{
  vl::OpenGLContext::dispatchMouseWheelEvent(ev->delta() / 120);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::keyPressEvent(QKeyEvent* ev)
{
  unsigned short unicode_ch = 0;
  vl::EKey key = vl::Key_None;
  translateKeyEvent(ev, unicode_ch, key);
  vl::OpenGLContext::dispatchKeyPressEvent(unicode_ch, key);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::keyReleaseEvent(QKeyEvent* ev)
{
  unsigned short unicode_ch = 0;
  vl::EKey key = vl::Key_None;
  translateKeyEvent(ev, unicode_ch, key);
  vl::OpenGLContext::dispatchKeyReleaseEvent(unicode_ch, key);
}


//-----------------------------------------------------------------------------
void VLQt4Widget::translateKeyEvent(QKeyEvent* ev, unsigned short& unicode_out, vl::EKey& key_out)
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
QGLContext* VLQt4Widget::context()
{
  return const_cast<QGLContext*>(QGLWidget::context());
}
