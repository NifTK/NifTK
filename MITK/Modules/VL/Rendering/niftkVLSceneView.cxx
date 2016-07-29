/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLWidget.h"
#include "niftkVLUtils.h"
#include "niftkVLMapper.h"
#include "niftkVLSceneView.h"
#include "niftkVLGlobalSettingsDataNode.h"

//#include <vlGraphics/plugins/ioVLX.hpp> // debugging

#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
#include <vlQt5/QtDirectory.hpp>
#include <vlQt5/QtFile.hpp>
#else
#include <vlQt4/QtDirectory.hpp>
#include <vlQt4/QtFile.hpp>
#endif
#include <vlCore/Colors.hpp>
#include <vlCore/FileSystem.hpp>
#include <vlCore/ResourceDatabase.hpp>
#include <vlCore/Time.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlVivid/VividVolume.hpp>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkImageData.h>
#include <mitkDataStorage.h>
#include <mitkProperties.h>
#include <mitkEnumerationProperty.h>
#include <mitkImage.h>
#include <mitkCoordinateAxesData.h>
#include <mitkImageReadAccessor.h>
#include <niftkScopedOGLContext.h>
#include <niftkVTKFunctions.h>
#include <stdexcept>

#ifdef BUILD_IGI
  #include <CameraCalibration/niftkUndistortion.h>
  #include <mitkCameraIntrinsicsProperty.h>
  #include <mitkCameraIntrinsics.h>
#endif

#ifdef _USE_PCL
  #include <niftkPCLData.h>
#endif

#ifdef _MSC_VER
  #ifdef _USE_NVAPI
    #include <nvapi.h>
  #endif
#endif

using namespace niftk;
using namespace vl;

//-----------------------------------------------------------------------------
// Init and shutdown VL
//-----------------------------------------------------------------------------

namespace
{
  class VLInit
  {
  public:
    VLInit() { vl::VisualizationLibrary::init(); }
    ~VLInit() { vl::VisualizationLibrary::shutdown(); }
  };

  VLInit s_ModuleInit;
}

//-----------------------------------------------------------------------------
// CUDA
//-----------------------------------------------------------------------------

#ifdef _USE_CUDA
  #include <niftkCUDAManager.h>
  #include <niftkCUDAImage.h>
  #include <niftkLightweightCUDAImage.h>
  #include <niftkCUDAImageProperty.h>
  #include <niftkFlipImageLauncher.h>
  #include <cuda_gl_interop.h>

  // #define VL_CUDA_TEST
namespace niftk
{
  class CudaTest {
    cudaGraphicsResource_t m_CudaResource;
    GLuint m_FramebufferId;
    GLuint m_TextureId;
    GLuint m_TextureTarget;
    mitk::DataNode::Pointer m_DataNode;
    niftk::CUDAImage::Pointer m_CUDAImage;

  public:
    CudaTest() {
      m_CudaResource = NULL;
      GLuint m_FramebufferId = 0;
      GLuint m_TextureId = 0;
      GLuint m_TextureTarget = 0;
    }

    mitk::DataNode* init(int w, int h) {
      m_CudaResource = NULL;
      m_TextureTarget = GL_TEXTURE_2D;
      glGenFramebuffers(1, &m_FramebufferId);
      glGenTextures(1, &m_TextureId);

      glBindTexture(m_TextureTarget, m_TextureId);
      glTexParameteri(m_TextureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
      glTexParameteri(m_TextureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
      glTexParameteri(m_TextureTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(m_TextureTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexImage2D(m_TextureTarget, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_FLOAT, 0);

      glBindFramebuffer(GL_FRAMEBUFFER, m_FramebufferId);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_TextureTarget, m_TextureId, 0);

      // Get CUDA representation for our GL texture

      cudaError_t err = cudaSuccess;
      err = cudaGraphicsGLRegisterImage( &m_CudaResource, m_TextureId, m_TextureTarget, cudaGraphicsRegisterFlagsNone );
      VIVID_CHECK( err == cudaSuccess );

      // Init CUDAImage

      // node
      m_DataNode = mitk::DataNode::New();
      // CUDAImage
      m_CUDAImage = niftk::CUDAImage::New();
      niftk::CUDAManager* cm = niftk::CUDAManager::GetInstance();
      cudaStream_t mystream = cm->GetStream(VL_CUDA_STREAM_NAME);
      niftk::WriteAccessor wa = cm->RequestOutputImage(w, h, 4);
      niftk::LightweightCUDAImage lwci = cm->Finalise(wa, mystream);
      // cm->Autorelease(wa, mystream);
      m_CUDAImage->SetLightweightCUDAImage(lwci);
      m_DataNode->SetData(m_CUDAImage);
      m_DataNode->SetName("CUDAImage VL Test");
      m_DataNode->SetVisibility(true);

      return m_DataNode.GetPointer();
    }

    void renderTriangle( int w, int h ) {
      glBindFramebuffer( GL_FRAMEBUFFER, m_FramebufferId );
      glDrawBuffer( GL_COLOR_ATTACHMENT0 );
      glViewport( 0, 0, w, h );
      glScissor( 0, 0, w, h );
      glEnable( GL_SCISSOR_TEST );
      glClearColor( 1.0f, 1.0f, 0.0f, 1.0f );
      glClear( GL_COLOR_BUFFER_BIT );
      glMatrixMode( GL_MODELVIEW );
      float zrot = vl::fract( vl::Time::currentTime() ) * 360.0f;
      glLoadMatrixf( mat4::getRotationXYZ( 0, 0, zrot ).ptr() );
      glMatrixMode( GL_PROJECTION );
      glLoadIdentity();
      glOrtho(-1, 1, -1, 1, -1, 1);
      glDisable( GL_LIGHTING );
      glDisable( GL_CULL_FACE );
      glDisable( GL_DEPTH_TEST );

      glBegin( GL_TRIANGLES );
        glColor3f( 1, 0, 0 );
        glVertex3f( -1, -1, 0 );
        glColor3f( 0, 1, 0 );
        glVertex3f( 0, 1, 0 );
        glColor3f( 0, 0, 1 );
        glVertex3f( 1, -1, 0 );
      glEnd();

      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
      glDrawBuffer( GL_BACK );
      glDisable( GL_SCISSOR_TEST );

      // Copy texture to CUDAImage
      niftk::CUDAManager* cm = niftk::CUDAManager::GetInstance();
      cudaStream_t mystream = cm->GetStream(VL_CUDA_STREAM_NAME);
      niftk::WriteAccessor wa = cm->RequestOutputImage(w, h, 4);

      cudaError_t err = cudaSuccess;
      cudaArray_t arr = 0;

      err = cudaGraphicsMapResources(1, &m_CudaResource, mystream);
      VIVID_CHECK(err == cudaSuccess);

      err = cudaGraphicsSubResourceGetMappedArray(&arr, m_CudaResource, 0, 0);
      VIVID_CHECK(err == cudaSuccess);

      err = cudaMemcpy2DFromArrayAsync(wa.m_DevicePointer, wa.m_BytePitch, arr, 0, 0, wa.m_PixelWidth * 4, wa.m_PixelHeight, cudaMemcpyDeviceToDevice, mystream);
      VIVID_CHECK(err == cudaSuccess);

      err = cudaGraphicsUnmapResources(1, &m_CudaResource, mystream);
      VIVID_CHECK(err == cudaSuccess);

      niftk::LightweightCUDAImage lwci = cm->Finalise(wa, mystream);
      // cm->Autorelease(wa, mystream);
      m_CUDAImage->SetLightweightCUDAImage(lwci);

      m_DataNode->Modified();
      m_DataNode->Update();
    }

    void renderQuad(int w, int h) {
      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
      glDrawBuffer( GL_BACK );
      glViewport( 0, 0, w, h );
      glScissor( 0, 0, w, h );
      glEnable( GL_SCISSOR_TEST );
      glClearColor( 1.0f, 0, 0, 1.0 );
      glClear( GL_COLOR_BUFFER_BIT );
      glMatrixMode( GL_MODELVIEW );
      glLoadIdentity();
      glMatrixMode( GL_PROJECTION );
      glOrtho(-1.1f, 1.1f, -1.1f, 1.1f, -1.1f, 1.1f);
      glDisable( GL_LIGHTING );
      glDisable( GLU_CULLING );

      glEnable( m_TextureTarget );
      VL_glActiveTexture( GL_TEXTURE0 );
      glBindTexture( m_TextureTarget, m_TextureId );

      glBegin( GL_QUADS );
        glColor3f( 1, 1, 1 );
        glTexCoord2f( 0, 0 );
        glVertex3f( -1, -1, 0 );

        glColor3f( 1, 1, 1 );
        glTexCoord2f( 1, 0 );
        glVertex3f( 1, -1, 0 );

        glColor3f( 1, 1, 1 );
        glTexCoord2f( 1, 1 );
        glVertex3f( 1, 1, 0 );

        glColor3f( 1, 1, 1 );
        glTexCoord2f( 0, 1 );
        glVertex3f( -1, 1, 0 );
      glEnd();

      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
      glDrawBuffer( GL_BACK );
      glDisable( GL_TEXTURE_2D );
      VL_glActiveTexture( GL_TEXTURE0 );
      glBindTexture( GL_TEXTURE_2D, 0 );
      glDisable( GL_SCISSOR_TEST );
    }
  };

} // namespace niftk

#endif

//-----------------------------------------------------------------------------
// VLSceneView
//-----------------------------------------------------------------------------

VLSceneView::VLSceneView( VLWidget* vlwidget ) :
  m_ScheduleTrackballAdjustView( true ),
  m_ScheduleInitScene ( true ),
  m_RenderingInProgressGuard ( false ),
  m_VLWidget( vlwidget )
{
#ifdef _USE_CUDA
  m_CudaTest = new CudaTest;
#endif

  m_EyeHandMatrix.setNull();

  // Note: here we don't have yet access to openglContext(), ie it's NULL

  // Interface VL with Qt's resource system to load GLSL shaders.
  vl::defFileSystem()->directories().clear();
  vl::defFileSystem()->directories().push_back( new vl::QtDirectory( ":/VL/" ) );

  // Create our VividRendering!
  m_VividRendering = new vl::VividRendering;
  m_VividRendering->setRenderingMode( vl::Vivid::DepthPeeling ); /* (default) */
  m_VividRendering->setCullingEnabled( false );
  // This creates some flickering on the skin for some reason
  m_VividRendering->setNearFarClippingPlanesOptimized( false );
  // Tries to accelerate rendering when no translucent objects are in the scene
  m_VividRendering->setDepthPeelingAutoThrottleEnabled( true );

  // VividRendering nicely prepares for us all the structures we need to use ;)
  m_VividRenderer = m_VividRendering->vividRenderer();
  m_SceneManager = m_VividRendering->sceneManager();

  // In the future Camera (and Trackball) should belong in VLView and be set upon rendering.
  m_Camera = m_VividRendering->calibratedCamera();

  // Initialize the trackball manipulator
  m_Trackball = new VLTrackballManipulator;
  m_Trackball->setEnabled( true );
  m_Trackball->setCamera( m_Camera.get() );
  m_Trackball->setTransform( NULL );
  m_Trackball->setPivot( vl::vec3(0,0,0) );
}

VLSceneView::~VLSceneView() {
#ifdef _USE_CUDA
  delete m_CudaTest;
  m_CudaTest = NULL;
#endif
}

//-----------------------------------------------------------------------------

 void VLSceneView::destroyEvent()
{
  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  removeDataStorageListeners();

  clearScene();

  // It's important do delete the VividRenderer here while the GL context
  // is still available as it disposes all the internal GL objects used.

  m_VividRendering = NULL;
  m_VividRenderer = NULL;
  m_SceneManager = NULL;
  m_Camera = NULL;
  m_Trackball = NULL;

  m_DataStorage = NULL;
  m_NodeVisibilityListener = NULL;
  m_NodeColorPropertyListener = NULL;
  m_NodeOpacityPropertyListener = NULL;

  m_DataNodeVLMapperMap.clear();
  m_NodesToUpdate.clear();
  m_NodesToAdd.clear();
  m_NodesToRemove.clear();
  m_CameraNode = NULL;
  m_BackgroundNode = NULL;
}

//-----------------------------------------------------------------------------

void VLSceneView::addDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.    AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeAdd));
    m_DataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeUpdate));
    m_DataStorage->RemoveNodeEvent. AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeRemove));
    m_DataStorage->DeleteNodeEvent. AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeRemove));
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::removeDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.    RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeAdd));
    m_DataStorage->ChangedNodeEvent.RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeUpdate));
    m_DataStorage->RemoveNodeEvent. RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeRemove));
    m_DataStorage->DeleteNodeEvent. RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::scheduleNodeRemove));
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::setDataStorage(mitk::DataStorage* ds)
{
  if ( ds == m_DataStorage ) {
    return;
  }

  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  removeDataStorageListeners();

  m_DataStorage = ds;
  addDataStorageListeners();

  clearScene();

  // Initialize VL Global Settings if not present
  if ( ! ds->GetNamedNode( VLGlobalSettingsDataNode::VLGlobalSettingsName() ) ) {
    VLGlobalSettingsDataNode::Pointer node = VLGlobalSettingsDataNode::New();
    ds->Add( node.GetPointer() );
  }

  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::scheduleSceneRebuild()
{
  clearScene();
  m_ScheduleInitScene = true;
  m_ScheduleTrackballAdjustView = true;
  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::scheduleTrackballAdjustView(bool schedule)
{
  m_ScheduleTrackballAdjustView = schedule;
  if ( schedule ) {
    openglContext()->update();
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::scheduleNodeAdd( const mitk::DataNode* node )
{
  if ( ! node || ! node->GetData() ) {
    return;
  }

  // m_NodesToRemove.erase( node ); // remove it first
  m_NodesToAdd.insert( mitk::DataNode::ConstPointer ( node ) ); // then add
  // m_NodesToUpdate.erase( node ); // then update
  openglContext()->update();

#if 0
  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<name-of-class>";
  printf("ScheduleNodeAdd: %s (%s)\n", node->GetName().c_str(), noc );
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::scheduleNodeUpdate( const mitk::DataNode* node )
{
  if ( ! node || ! node->GetData() ) {
    return;
  }

  m_NodesToRemove.erase( node ); // abort the removal
  // m_NodesToAdd.erase( node ); // let it add it first
  m_NodesToUpdate.insert( mitk::DataNode::ConstPointer ( node ) ); // then update
  openglContext()->update();

#if 0
  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<unknown-class>";
  printf("ScheduleNodeUpdate: %s (%s)\n", node->GetName().c_str(), noc );
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::scheduleNodeRemove( const mitk::DataNode* node )
{
  // If this fails most probably someone is calling this from another thread.
  VIVID_CHECK( ! m_RenderingInProgressGuard );

  if ( ! node /* || ! node->GetData() */ ) {
    return;
  }

#if 0
  m_NodesToRemove.insert( mitk::DataNode::ConstPointer ( node ) ); // remove it
#else
  // deal with it immediately
  removeDataNode( node );
#endif

  m_NodesToAdd.erase( node );    // abort the addition
  m_NodesToUpdate.erase( node ); // abort the update
  openglContext()->update();
#if 0
  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<name-of-class>";
  printf("ScheduleNodeRemove: %s (%s)\n", node->GetName().c_str(), noc );
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::initSceneFromDataStorage()
{
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );

  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  clearScene();

  if ( m_DataStorage.IsNull() ) {
    return;
  }

  typedef itk::VectorContainer<unsigned int, mitk::DataNode::Pointer> NodesContainerType;
  NodesContainerType::ConstPointer vc = m_DataStorage->GetAll();

  for (unsigned int i = 0; i < vc->Size(); ++i)
  {
    mitk::DataNode::Pointer node = vc->ElementAt(i);
    if ( ! node || ! node->GetData() ) {
      continue;
    } else {
      addDataNode( node.GetPointer() );
    }
  }

  #if 0
    // dump scene to VLB/VLT format for debugging
    ref< vl::ResourceDatabase > db = new vl::ResourceDatabase;
    for( int i = 0; i < m_SceneManager->tree()->actors()->size(); ++i ) {
      vl::Actor* act = m_SceneManager->tree()->actors()->at(i);
      if ( act->enableMask() ) {
        // db->resources().push_back( act );
        // vl::String fname = filename( files[i] );
        db->resources().push_back( act );
        vl::String fname = "niftk-liver";
        vl::saveVLT( "C:/git-ucl/VisualizationLibrary/data/tmp/" + fname + ".vlt", db.get() );
        vl::saveVLB( "C:/git-ucl/VisualizationLibrary/data/tmp/" + fname + ".vlb", db.get() );
      }
    }
  #endif
}

//-----------------------------------------------------------------------------

VLMapper* VLSceneView::addDataNode(const mitk::DataNode* node)
{
  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  // Add only once and only if valid
  if ( ! node || ! node->GetData() || getVLMapper( node ) != NULL ) {
    return NULL;
  }

  #if 0
    dumpNodeInfo( "addDataNode()", node );
    dumpNodeInfo( "addDataNode()->GetData()", node->GetData() );
  #endif

  ref<VLMapper> vl_node = VLMapper::create( node, this );
  if ( vl_node ) {
    if ( vl_node->init() ) {
      m_DataNodeVLMapperMap[ node ] = vl_node;
      vl_node->update();
    }
  }

  return vl_node.get();
}

//-----------------------------------------------------------------------------

void VLSceneView::removeDataNode(const mitk::DataNode* node)
{
  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  if ( node == m_BackgroundNode ) {
    setBackgroundNode( NULL );
  }

  // dont leave a dangling update behind.
  m_NodesToUpdate.erase(node);
  m_NodesToAdd.erase(node);

  // Remove VLMapper and VL data
  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  if ( it != m_DataNodeVLMapperMap.end() ) {
    VLMapper* vl_node = it->second.get();
    VIVID_CHECK( vl_node );
    if ( vl_node ) {
      vl_node->remove();
      m_DataNodeVLMapperMap.erase(it);
    }
  }
}

void VLSceneView::updateDataNode(const mitk::DataNode* node)
{
  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  #if 0
    dumpNodeInfo( "updateDataNode()", node );
    dumpNodeInfo( "updateDataNode()->GetData()", node->GetData() );
  #endif

  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  if ( it != m_DataNodeVLMapperMap.end() ) {
    // this might recreate new Actors
    it->second->update();
  }

  // The camera node contains the camera position information
  // The background node contains the camera intrinsics info
  // BTW, we also call updateCameraParameters() on resize.
  // update camera
  if ( node == m_CameraNode || node == m_BackgroundNode ) {
    updateCameraParameters();
  }
}

//-----------------------------------------------------------------------------

VLMapper* VLSceneView::getVLMapper( const mitk::DataNode* node )
{
  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  return it == m_DataNodeVLMapperMap.end() ? NULL : it->second.get();
}

//-----------------------------------------------------------------------------

void VLSceneView::setBackgroundColour(float r, float g, float b)
{
  VIVID_CHECK( m_VividRendering );
  m_VividRendering->setBackgroundColor( fvec4(r, g, b, 1) );
  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::initEvent()
{
  VIVID_CHECK( m_VLWidget->contextIsCurrent() );

#if 0
  // Mic: this seems to be failing for me.
  // use the device that is running our opengl context as the compute-device
  // for sorting triangles in the correct order.
  if (m_OclService)
  {
    // Force tests to run on the first GPU with shared context
    m_OclService->SpecifyPlatformAndDevice(0, 0, true);
    // Calling this to make sure that the context is created right at startup
    cl_context clContext = m_OclService->GetContext();
  }
#endif

  MITK_INFO << "OpenGL Context Info:\n";
  MITK_INFO << "GL_VERSION: " << glGetString(GL_VERSION) << "\n";
  MITK_INFO << "GL_VENDOR: " << glGetString(GL_VENDOR) << "\n";
  MITK_INFO << "GL_RENDERER: " << glGetString(GL_RENDERER) << "\n";
  MITK_INFO << "GL_SHADING_LANGUAGE_VERSION: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
  MITK_INFO << "\n";

  openglContext()->addEventListener( m_Trackball.get() );
  // Schedule reset of the camera based on the scene content
  scheduleTrackballAdjustView();

#ifdef VL_CUDA_TEST // CUDA test
  {
    mitk::DataNode::Pointer node = m_CudaTest->init( 100, 100 );
    m_DataStorage->Add(node);
  }
#endif

#if 0 // PointSet test
  {
    mitk::DataNode::Pointer node = new mitk::DataNode::New();
    mitk::PointSet::Pointer pointset = new mitk::PointSet::New();
    ... left as an exercise for the student ...
  }
#endif

#if 0 // PCL test
  {
    // Point cloud data test
    mitk::DataNode::Pointer node = mitk::DataNode::New();
    niftk::PCLData::Pointer pcl = niftk::PCLData::New();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  c(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (int i = 0; i < 1000; ++i) {
      pcl::PointXYZRGB  q(std::rand() % 256, std::rand() % 256, std::rand() % 256);
      q.x = std::rand() % 256;
      q.y = std::rand() % 256;
      q.z = std::rand() % 256;
      c->push_back(q);
    }
    pcl->SetCloud(c);
    node->SetData(pcl);

    node->SetName("PCL Test");
    node->SetVisibility(true);

    m_DataStorage->Add(node);
  }
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::resizeEvent( int w, int h )
{
   VIVID_CHECK( m_VLWidget->contextIsCurrent() );

  // dont do anything if window is zero size.
  // it's an opengl error to have a viewport like that!
  if ( w <= 0 || h <= 0 ) {
    return;
  }

  updateCameraParameters();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateEvent()
{
  VIVID_CHECK( m_VLWidget->contextIsCurrent() );

  renderScene();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateScene() {
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );
  VIVID_CHECK( m_VLWidget->contextIsCurrent() );

  if ( m_ScheduleInitScene ) {
    initSceneFromDataStorage();
    m_ScheduleInitScene = false;
  } else {
#if 0
    // Execute scheduled removals
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToRemove.begin(); it != m_NodesToRemove.end(); ++it)
    {
      removeDataNode(*it);
    }
    m_NodesToRemove.clear();
#else
     VIVID_CHECK( m_NodesToRemove.empty() );
#endif

    // Execute scheduled additions
    m_ScheduleTrackballAdjustView |= m_NodesToAdd.size() > 0;
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToAdd.begin(); it != m_NodesToAdd.end(); ++it)
    {
      addDataNode(*it);
    }
    m_NodesToAdd.clear();

    // Execute scheduled updates
    if (m_NodesToUpdate.size() > 0)
    {
      openglContext()->update();
    }
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToUpdate.begin(); it != m_NodesToUpdate.end(); ++it)
    {
      updateDataNode(*it);
    }
    m_NodesToUpdate.clear();
  }

  // Reset trackball view on demand

  if ( m_ScheduleTrackballAdjustView && m_Trackball->isEnabled() ) {
    m_Trackball->adjustView( m_VividRendering.get(), vl::vec3(0,0,1), vl::vec3(0,1,0), 1.0f );
    m_ScheduleTrackballAdjustView = false;
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::renderScene()
{
  m_RenderingInProgressGuard = true;

  VIVID_CHECK( m_VLWidget->contextIsCurrent() );

  updateScene();

  // Set frame time for all the rendering
  vl::real now_time = vl::Time::currentTime();
  m_VividRendering->setFrameClock( now_time );

  // Execute rendering
  m_VividRendering->render( openglContext()->framebuffer() );

#ifdef VL_CUDA_TEST // Cuda test
  m_CudaTest->renderTriangle( 100, 100 );
#endif

  // Show rendering
  if ( openglContext()->hasDoubleBuffer() ) {
    openglContext()->swapBuffers();
  }

  m_RenderingInProgressGuard = false;

  VL_CHECK_OGL();
}

//-----------------------------------------------------------------------------

void VLSceneView::clearScene()
{
  if ( ! m_VividRendering ) {
    return;
  }

  niftk::ScopedOGLContext glctx( const_cast<QGLContext*>(m_VLWidget->context()) );

  // Shut down VLMappers
  for ( DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.begin(); it != m_DataNodeVLMapperMap.end(); ++it ) {
    it->second->remove();
  }

  m_VividRendering->stencilActors().clear();
  m_SceneManager->tree()->actors()->clear();
  m_SceneManager->tree()->eraseAllChildren();

  m_DataNodeVLMapperMap.clear();
  m_NodesToUpdate.clear();
  m_NodesToAdd.clear();
  m_NodesToRemove.clear();
  m_CameraNode = 0;
  m_BackgroundNode = 0;
  m_VividRendering->setBackgroundImageEnabled( false );

  m_ScheduleInitScene = true;
  m_ScheduleTrackballAdjustView = true;
}

//-----------------------------------------------------------------------------

void VLSceneView::setOpacity( float opacity )
{
  m_VividRendering->setOpacity( opacity );
  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::setDepthPeelingPasses( int passes ) {
  m_VividRenderer->setNumPasses( passes );
}

//-----------------------------------------------------------------------------

void VLSceneView::reInit(const vl::vec3& dir, const vl::vec3& up, float bias) {
  AABB aabb;
  for ( DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.begin();
        it != m_DataNodeVLMapperMap.end();
        ++it ) {
    if ( VLUtils::getBoolProp( it->first.GetPointer(), "selected", false ) ) {
      aabb = it->second->actor()->boundingBox();
      break;
    }
  }
  if ( ! aabb.isNull() ) {
    m_Trackball->adjustView( aabb, dir, up, bias );
    openglContext()->update();
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::globalReInit(const vl::vec3& dir, const vl::vec3& up, float bias) {
  m_Trackball->adjustView( m_VividRendering.get(), dir, up, bias );
  openglContext()->update();
}

//-----------------------------------------------------------------------------

bool VLSceneView::setBackgroundNode(const mitk::DataNode* node)
{
  VIVID_CHECK( m_VividRendering );
  m_BackgroundNode = node;
  m_BackgroundImage = NULL;
#ifdef _USE_CUDA
  m_BackgroundCUDAImage = NULL;
#endif

  if ( ! node ) {
    m_VividRendering->setBackgroundImageEnabled( false );
    updateCameraParameters();
    return true;
  } else {
    updateCameraParameters();
  }

  vl::Texture* tex = NULL;
  mitk::Vector3D img_spacing;
  int width  = 0;
  int height = 0;

  // Wire up background texture
#ifdef _USE_CUDA
  VLMapperCUDAImage* imgCu_mapper = dynamic_cast<VLMapperCUDAImage*>( getVLMapper( node ) );
#endif

  VLMapper2DImage* img2d_mapper = dynamic_cast<VLMapper2DImage*>( getVLMapper( node ) );
  if ( img2d_mapper )
  {
    // assign texture
    tex = img2d_mapper->texture();
    // image size and pixel aspect ratio
    m_BackgroundImage = dynamic_cast<mitk::Image*>( node->GetData() );
    img_spacing = m_BackgroundImage->GetGeometry()->GetSpacing();
    width = m_BackgroundImage->GetDimension(0);
    height = m_BackgroundImage->GetDimension(1);
  }
#ifdef _USE_CUDA
  else if ( imgCu_mapper )
  {
    // assign texture
    tex = imgCu_mapper->texture();
    // image size and pixel aspect ratio
    m_BackgroundCUDAImage = dynamic_cast<niftk::CUDAImage*>( node->GetData() ); VIVID_CHECK(m_BackgroundCUDAImage);
    img_spacing = m_BackgroundCUDAImage->GetGeometry()->GetSpacing();
    niftk::LightweightCUDAImage lwci = m_BackgroundCUDAImage->GetLightweightCUDAImage();
    width = lwci.GetWidth();
    height = lwci.GetHeight();
  }
#endif
  else
  {
    return false;
  }
  // set background texture
  m_VividRendering->backgroundTexSampler()->setTexture( tex );
  // set background aspect ratio
  VIVID_CHECK(img_spacing[0]);
  VIVID_CHECK(img_spacing[1]);
  VIVID_CHECK(width);
  VIVID_CHECK(height);
  m_Camera->setCalibratedImageSize(width, height, img_spacing[0] / img_spacing[1]);

  // Hide 3D plane with 2D image on it
  VLUtils::setBoolProp( const_cast<mitk::DataNode*>(node), "visible", false );

  // Enable background rendering
  m_VividRendering->setBackgroundImageEnabled( true );

  openglContext()->update();

  return true;
}

//-----------------------------------------------------------------------------

bool VLSceneView::setCameraTrackingNode(const mitk::DataNode* node)
{
  VIVID_CHECK( m_VividRendering );
  VIVID_CHECK( m_Trackball );

  // Whenever we set the camera node to NULL we recenter the scene using the trackball

  m_CameraNode = node;

  if (m_CameraNode.IsNull())
  {
    m_Trackball->setEnabled( true );
    scheduleTrackballAdjustView( true );
  } else {
    VLUtils::dumpNodeInfo( "CameraNode():", node );
    VLUtils::dumpNodeInfo( "node->GetData()", node->GetData() );
    m_Trackball->setEnabled( false );
    scheduleTrackballAdjustView( false );
    // update camera position
    updateCameraParameters();
  }

  openglContext()->update();

  return true;
}

//-----------------------------------------------------------------------------

void VLSceneView::setEyeHandFileName(const std::string& fileName) {
  m_EyeHandMatrix.setNull();

  if ( ! fileName.empty() )
  {
    // Note: Currently doesn't do error handling properly.
    // i.e no return code, no exception.
    vtkSmartPointer<vtkMatrix4x4> vtkmat = niftk::LoadMatrix4x4FromFile(fileName);
    m_EyeHandMatrix = VLUtils::getVLMatrix(vtkmat);
    if ( m_EyeHandMatrix.isNull() )
    {
      mitkThrow() << "Failed to niftk::LoadMatrix4x4FromFile(" << fileName << ")";
    }
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::updateCameraParameters()
{
  int win_w = openglContext()->width();
  int win_h = openglContext()->height();

  // default perspective projection
  m_Camera->viewport()->set( 0, 0, win_w, win_h );
  m_Camera->setProjectionPerspective();
  VIVID_CHECK( m_Camera->viewport()->isScissorEnabled() );

  // update camera viewport and projecton

  if ( m_VividRendering->backgroundImageEnabled() )
  {
    // Calibration parameters come from the background node.
    VIVID_CHECK( m_BackgroundNode );

    #ifdef BUILD_IGI
      mitk::BaseProperty::Pointer cam_cal_prop = m_BackgroundNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
      if ( cam_cal_prop ) {
        mitk::CameraIntrinsicsProperty::Pointer cam_intr_prop = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cam_cal_prop.GetPointer());
        if ( cam_intr_prop ) {
          mitk::CameraIntrinsics::Pointer intrinsics = cam_intr_prop->GetValue();
          if ( intrinsics ) {
            // set up calibration parameters

            // screen size
            m_Camera->setScreenSize( win_w, win_h );

            // image size and pixel aspect ratio
            mitk::Image* image = dynamic_cast<mitk::Image*>( m_BackgroundNode->GetData() );
            mitk::Vector3D  imgScaling = image->GetGeometry()->GetSpacing();
            int width  = image->GetDimension(0);
            int height = image->GetDimension(1);
            m_Camera->setCalibratedImageSize(width, height, imgScaling[0] / imgScaling[1]);

            // intrinsic parameters
            m_Camera->setIntrinsicParameters(
              intrinsics->GetFocalLengthX(),
              intrinsics->GetFocalLengthY(),
              intrinsics->GetPrincipalPointX(),
              intrinsics->GetPrincipalPointY()
            );

            // updates projection and viewport based on the given parameters
            m_Camera->updateCalibration();
          }
        }
      }
    #endif
  }

  // update camera position

  if ( m_CameraNode ) {
    // This implies a right handed coordinate system.
    // By default, assume camera position is at origin, looking down the world +ve z-axis.
    vec3 origin(0, 0, 0);
    vec3 focalPoint(0, 0, 1000);
    vec3 viewUp(0, -1000, 0);

    // If the stereo right to left matrix exists, we must be doing the right hand image.
    // So, in this case, we have an extra transformation to consider.
    if ( m_BackgroundImage )
    {
      niftk::Undistortion::MatrixProperty::Pointer prop =
        dynamic_cast<niftk::Undistortion::MatrixProperty*>(
          m_BackgroundImage->GetProperty( niftk::Undistortion::s_StereoRigTransformationPropertyName ).GetPointer() );

      if ( prop.IsNotNull() )
      {
        mat4 rig_txf = VLUtils::getVLMatrix( prop->GetValue() );
        origin = rig_txf * origin;
        focalPoint = rig_txf * focalPoint;
        viewUp = rig_txf * viewUp;
        viewUp = viewUp - origin;
      }
    }

    // If additionally, the user has selected a transformation matrix, we move camera accordingly.
    // Note, 2 use-cases:
    // (a) User specifies camera to world - just use the matrix as given.
    // (b) User specified eye-hand matrix - multiply by eye-hand then tracking matrix
    //                                    - to construct the camera to world.

    // this is the camera modeling matrix (not the view matrix, its inverse)
    mat4 camera_to_world;
    mat4 supplied_matrix = VLUtils::getVLMatrix( m_CameraNode->GetData() );
    VIVID_CHECK( ! supplied_matrix.isNull() );
    if ( ! supplied_matrix.isNull() )
    {
      if ( m_EyeHandMatrix.isNull() )
      {
        // Use case (a) - supplied transform is camera to world.
        camera_to_world = supplied_matrix;
      }
      else
      {
        // Use case (b) - supplied transform is a tracking transform.
        camera_to_world = supplied_matrix * m_EyeHandMatrix;
      }

      origin = camera_to_world * origin;
      focalPoint = camera_to_world * focalPoint;
      viewUp = camera_to_world * viewUp;
      viewUp = viewUp - origin;
    }

    m_Camera->setViewMatrix( mat4::getLookAt(origin, focalPoint, viewUp) );
  }
}
