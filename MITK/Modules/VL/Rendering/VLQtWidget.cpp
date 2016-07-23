/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// NOTE:
// In production you may want to review these checks and handle them as gracefully as possible.
#if defined(_MSC_VER)
  #define VIVID_TRAP() { if (IsDebuggerPresent()) { __debugbreak(); } else ::vl::abort_vl(); }
#elif defined(__GNUG__) || defined(__MINGW32__)
  #define VIVID_TRAP() { fflush(stdout); fflush(stderr); asm("int $0x3"); }
#else
  #define VIVID_TRAP() { ::vl::abort_vl(); }
#endif
#if 1
  // This is better for debugging
  #define VIVID_CHECK(expr) { if ( ! ( expr ) ) { ::vl::log_failed_check( #expr, __FILE__, __LINE__ ); VIVID_TRAP(); } }
#else
  // This allows the user to ignore the exception while giving full info about the error
  #define STRINGIZE_DETAIL(x) #x
  #define STRINGIZE(x) STRINGIZE_DETAIL(x)
  #define VIVID_CHECK(expr) { if ( ! (expr) ) { throw std::runtime_error( __FILE__ " line " STRINGIZE(__LINE__) ": " #expr ); } }
#endif

#include "VLQtWidget.h"

#include <QTextStream>
#include <QFile>
#include <QDir>

#include <vlQt5/QtDirectory.hpp>
#include <vlQt5/QtFile.hpp>
#include <vlCore/Log.hpp>
#include <vlCore/Time.hpp>
#include <vlCore/Colors.hpp>
#include <vlCore/GlobalSettings.hpp>
#include <vlCore/FileSystem.hpp>
#include <vlCore/ResourceDatabase.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/RenderQueueSorter.hpp>
#include <vlGraphics/GLSL.hpp>
#include <vlGraphics/plugins/ioVLX.hpp>
#include <vlGraphics/FramebufferObject.hpp>
#include <vlGraphics/AdjacencyExtractor.hpp>
#include <vlVivid/VividVolume.hpp>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataNormals.h>
#include <vtkImageData.h>
#include <mitkDataStorage.h>
#include <mitkProperties.h>
#include <mitkEnumerationProperty.h>
#include <mitkImage.h>
#include <mitkCoordinateAxesData.h>
#include <mitkImageReadAccessor.h>
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

using namespace vl;

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

  #define VL_CUDA_STREAM_NAME "VL-CUDA-STREAM"

  // #define VL_CUDA_TEST

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
      glLoadMatrixf( vl::mat4::getRotationXYZ( 0, 0, zrot ).ptr() );
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
#endif

//-----------------------------------------------------------------------------
// VLUserData
//-----------------------------------------------------------------------------

struct VLUserData: public vl::Object
{
  VLUserData(): m_TransformModifiedTime(0), m_ImageModifiedTime(0) { }
  itk::ModifiedTimeType m_TransformModifiedTime;
  itk::ModifiedTimeType m_ImageModifiedTime;
};

//-----------------------------------------------------------------------------
// mitk::EnumerationProperty wrapper classes
//-----------------------------------------------------------------------------

class VL_Render_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Render_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Render_Mode_Property() {
    AddEnum("DepthPeeling",  0);
    AddEnum("FastRender",    1);
    AddEnum("StencilRender", 2);
  }
};

class VL_Volume_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Volume_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Volume_Mode_Property() {
    AddEnum("Direct",     0);
    AddEnum("Isosurface", 1);
    AddEnum("MIP",        2);
  }
};

class VL_Point_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Point_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Point_Mode_Property() {
    AddEnum("3D", 0);
    AddEnum("2D", 1);
  }
};

class VL_Smart_Target_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Smart_Target_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Smart_Target_Property() {
    AddEnum("Color",      0);
    AddEnum("Alpha",      1);
    AddEnum("Saturation", 2);
  }
};

class VL_Fog_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Fog_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Fog_Mode_Property() {
    AddEnum("Off",    0);
    AddEnum("Linear", 1);
    AddEnum("Exp",    2);
    AddEnum("Exp2",   3);
  }
};

class VL_Clip_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Clip_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Clip_Mode_Property() {
    AddEnum("Off",    0);
    AddEnum("Sphere", 1);
    AddEnum("Box",    2);
    AddEnum("Plane",  3);
  }
};

class VL_Surface_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Surface_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Surface_Mode_Property() {
    AddEnum("Polys",           0);
    AddEnum("Outline3D",       1);
    AddEnum("Polys+Outline3D", 2);
    AddEnum("Slice",           3);
    AddEnum("Outline2D",       4);
    AddEnum("Polys+Outline2D", 5);
  }
};

//-----------------------------------------------------------------------------
// Util functions
//-----------------------------------------------------------------------------

namespace
{
  vl::vec3 getVector3DProp( const mitk::DataNode* node, const char* prop_name, vl::vec3 defval ) {
    VIVID_CHECK( dynamic_cast<const mitk::Vector3DProperty*>( node->GetProperty( prop_name ) ) );
    const mitk::Vector3DProperty* prop = dynamic_cast<const mitk::Vector3DProperty*>( node->GetProperty( prop_name ) );
    if ( prop ) {
      double* val = prop->GetValue().GetDataPointer();
      return vl::vec3( (float)val[0], (float)val[1], (float)val[2] );
    } else {
      return defval;
    }
  }

  vl::vec3 getPoint3DProp( const mitk::DataNode* node, const char* prop_name, vl::vec3 defval ) {
    VIVID_CHECK( dynamic_cast<const mitk::Point3dProperty*>( node->GetProperty( prop_name ) ) );
    const mitk::Point3dProperty* prop = dynamic_cast<const mitk::Point3dProperty*>( node->GetProperty( prop_name ) );
    if ( prop ) {
      double* val = prop->GetValue().GetDataPointer();
      return vl::vec3( (float)val[0], (float)val[1], (float)val[2] );
    } else {
      return defval;
    }
  }

  vl::vec4 getPoint4DProp( const mitk::DataNode* node, const char* prop_name, vl::vec4 defval ) {
    VIVID_CHECK( dynamic_cast<const mitk::Point4dProperty*>( node->GetProperty( prop_name ) ) );
    const mitk::Point4dProperty* prop = dynamic_cast<const mitk::Point4dProperty*>( node->GetProperty( prop_name ) );
    if ( prop ) {
      double* val = prop->GetValue().GetDataPointer();
      return vl::vec4( (float)val[0], (float)val[1], (float)val[2], (float)val[3] );
    } else {
      return defval;
    }
  }

  int getEnumProp( const mitk::DataNode* node, const char* prop_name, int defval = 0 ) {
    VIVID_CHECK( dynamic_cast<const mitk::EnumerationProperty*>( node->GetProperty( prop_name ) ) );
    const mitk::EnumerationProperty* prop = dynamic_cast<const mitk::EnumerationProperty*>( node->GetProperty( prop_name ) );
    if ( prop ) {
      return prop->GetValueAsId();
    } else {
      return defval;
    }
  }

  bool getBoolProp( const mitk::DataNode* node, const char* prop_name, bool defval ) {
    VIVID_CHECK( dynamic_cast<const mitk::BoolProperty*>( node->GetProperty( prop_name ) ) );
    bool val = defval;
    node->GetBoolProperty( prop_name, val );
    return val;
  }

  bool setBoolProp( mitk::DataNode* node, const char* prop_name, bool val ) {
    VIVID_CHECK( dynamic_cast<const mitk::BoolProperty*>( node->GetProperty( prop_name ) ) );
    mitk::BoolProperty* prop = dynamic_cast<mitk::BoolProperty*>( node->GetProperty( prop_name ) );
    if ( ! prop ) {
      return false;
    }
    prop->SetValue( val );
    return true;
  }

  float getFloatProp( const mitk::DataNode* node, const char* prop_name, float defval = 0 ) {
    VIVID_CHECK( dynamic_cast<const mitk::FloatProperty*>( node->GetProperty( prop_name ) ) );
    float val = defval;
    node->GetFloatProperty( prop_name, val );
    return val;
  }

  int getIntProp( const mitk::DataNode* node, const char* prop_name, int defval = 0 ) {
    VIVID_CHECK( dynamic_cast<const mitk::IntProperty*>( node->GetProperty( prop_name ) ) );
    int val = defval;
    node->GetIntProperty( prop_name, val );
    return val;
  }

  vl::vec4 getColorProp( const mitk::DataNode* node, const char* prop_name, vl::vec4 defval = vl::white ) {
    VIVID_CHECK( dynamic_cast<const mitk::ColorProperty*>( node->GetProperty( prop_name ) ) );
    float rgb[3] = { defval.r(), defval.g(), defval.b() };
    node->GetColor(rgb, NULL, prop_name );
    return vl::vec4( rgb[0], rgb[1], rgb[2], defval.a() );
  }

  void initVolumeProps( mitk::DataNode* node ) {
    // init only once if multiple views are open
    if ( node->GetProperty("VL.Volume.Mode") ) {
      return;
    }

    mitk::EnumerationProperty::Pointer mode = VL_Volume_Mode_Property::New();
    node->AddProperty( "VL.Volume.Mode", mode);
    mode->SetValue( 0 );

    mitk::FloatProperty::Pointer iso = mitk::FloatProperty::New();
    node->SetProperty("VL.Volume.Iso", iso );
    iso->SetValue( 0.5f );

    mitk::FloatProperty::Pointer density = mitk::FloatProperty::New();
    node->SetProperty("VL.Volume.Density", density);
    density->SetValue( 4.0f );

    mitk::IntProperty::Pointer samples = mitk::IntProperty::New();
    node->SetProperty("VL.Volume.SamplesPerRay", samples);
    samples->SetValue( 512 );
  }

  void updateVolumeProps( vl::VividVolume* vol, const mitk::DataNode* node )
  {
    int mode = getEnumProp( node, "VL.Volume.Mode" );
    float iso = getFloatProp( node, "VL.Volume.Iso" );
    float density = getFloatProp( node, "VL.Volume.Density" );
    int samples = getIntProp( node, "VL.Volume.SamplesPerRay" );

    vol->setVolumeMode( (vl::VividVolume::EVolumeMode)mode );
    vol->setIsoValue( iso );
    vol->setVolumeDensity( density );
    vol->setSamplesPerRay( samples );
  }

  void initMaterialProps( mitk::DataNode* node )
  {
    // init only once if multiple views are open
    if ( node->GetProperty("VL.Material.Color") ) {
      return;
    }

    // Get defaults from vl::Material
    vl::Material m;

    mitk::ColorProperty::Pointer color = mitk::ColorProperty::New();
    node->SetProperty("VL.Material.Color", color);
    color->SetValue( m.frontDiffuse().ptr() );

    mitk::FloatProperty::Pointer opacity = mitk::FloatProperty::New();
    node->SetProperty("VL.Material.Opacity", opacity);
    opacity->SetValue( 1 );

    mitk::ColorProperty::Pointer spec_color = mitk::ColorProperty::New();
    node->SetProperty("VL.Material.Specular.Color", spec_color);
    spec_color->SetValue( m.frontSpecular().ptr() );

    mitk::FloatProperty::Pointer spec_shininess = mitk::FloatProperty::New();
    node->SetProperty("VL.Material.Specular.Shininess", spec_shininess);
    spec_shininess->SetValue( m.frontShininess() );

    // Stencil option is available only for Surface-like objects

    mitk::BoolProperty::Pointer is_stencil = mitk::BoolProperty::New();
    node->SetProperty("VL.IsStencil", is_stencil);
    is_stencil->SetValue( m.frontShininess() );
  }

  void updateMaterialProps( Effect* fx, const mitk::DataNode* node )
  {
#if 0
    vec4 color = getColorProp( node, "VL.Material.Color" );
    color.a() = getFloatProp( node, "VL.Material.Opacity" );
#else
    vec4 color = getColorProp( node, "color" );
    color.a() = getFloatProp( node, "opacity" );
#endif
    vec4 spec_color = getColorProp( node, "VL.Material.Specular.Color" );
    float shininess = getFloatProp( node, "VL.Material.Specular.Shininess" );

    Shader* sh = fx->shader();

    sh->getMaterial()->setDiffuse( color );
    sh->getMaterial()->setSpecular( spec_color );
    sh->getMaterial()->setShininess( shininess );
  }


  void initFogProps( mitk::DataNode* node )
  {
    // init only once if multiple views are open
    if ( node->GetProperty("VL.Fog.Mode") ) {
      return;
    }

    // gocUniform("vl_Vivid.smartFog.mode")
    mitk::EnumerationProperty::Pointer fog_mode = VL_Fog_Mode_Property::New();
    node->SetProperty("VL.Fog.Mode", fog_mode);
    fog_mode->SetValue( 0 );

    // gocUniform("vl_Vivid.smartFog.target")
    mitk::EnumerationProperty::Pointer fog_target = VL_Smart_Target_Property::New();
    node->SetProperty("VL.Fog.Target", fog_target);
    fog_target->SetValue( 0 );

    // gocFog()->setColor( . );
    mitk::ColorProperty::Pointer fog_color = mitk::ColorProperty::New();
    node->SetProperty("VL.Fog.Color", fog_color);
    fog_color->SetValue( vl::darkgray.ptr() );

    // Only used with Linear mode
    // gocFog()->setStart( . );
    mitk::FloatProperty::Pointer fog_start = mitk::FloatProperty::New();
    node->SetProperty("VL.Fog.Start", fog_start);
    fog_start->SetValue( 0 );

    // Only used with Linear mode
    // gocFog()->setEnd( . );
    mitk::FloatProperty::Pointer fog_stop = mitk::FloatProperty::New();
    node->SetProperty("VL.Fog.End", fog_stop);
    fog_stop->SetValue( 1000 );

    // Only used with Exp & Exp2 mode
    // gocFog()->setDensity( . );
    mitk::FloatProperty::Pointer fog_density = mitk::FloatProperty::New();
    node->SetProperty("VL.Fog.Density", fog_density);
    fog_density->SetValue( 1 );
  }

  void updateFogProps( Effect* fx, const mitk::DataNode* node )
  {
    int fog_mode = getEnumProp( node, "VL.Fog.Mode", 0 );
    int fog_target = getEnumProp( node, "VL.Fog.Target", 0 );
    vec4 fog_color = getColorProp( node, "VL.Fog.Color", vl::black );
    float fog_start = getFloatProp( node, "VL.Fog.Start", 0 );
    float fog_end = getFloatProp( node, "VL.Fog.End", 0 );
    float fog_density = getFloatProp( node, "VL.Fog.Density", 0 );

    Shader* sh = fx->shader();

    sh->gocFog()->setColor( fog_color);
    sh->getUniform("vl_Vivid.smartFog.mode")->setUniformI( fog_mode );
    sh->getUniform("vl_Vivid.smartFog.target")->setUniformI( fog_target );
    sh->gocFog()->setStart( fog_start );
    sh->gocFog()->setEnd( fog_end );
    sh->gocFog()->setDensity( fog_density );
  }

  void initClipProps( mitk::DataNode* node )
  {
    // init only once if multiple views are open
    if ( node->GetProperty("VL.Clip.0.Mode") ) {
      return;
    }

    #define CLIP_UNIT(field) (std::string("VL.Clip.") + i + '.' + field).c_str()

    for( char i = '0'; i < '4'; ++i ) {

      // gocUniform("vl_Vivid.smartClip[0].mode")
      mitk::EnumerationProperty::Pointer mode = VL_Clip_Mode_Property::New();
      node->SetProperty(CLIP_UNIT("Mode"), mode);
      mode->SetValue( 0 );

      // gocUniform("vl_Vivid.smartClip[0].target")
      mitk::EnumerationProperty::Pointer target = VL_Smart_Target_Property::New();
      node->SetProperty(CLIP_UNIT("Target"), target);
      target->SetValue( 0 );

      // gocUniform("vl_Vivid.smartClip[0].color")
      mitk::ColorProperty::Pointer color = mitk::ColorProperty::New();
      node->SetProperty(CLIP_UNIT("Color"), color);
      color->SetValue( vl::fuchsia.ptr() );

      // gocUniform("vl_Vivid.smartClip[0].fadeRange")
      mitk::FloatProperty::Pointer fade_range = mitk::FloatProperty::New();
      node->SetProperty(CLIP_UNIT("FadeRange"), fade_range);
      fade_range->SetValue( 0 );

      // gocUniform("vl_Vivid.smartClip[0].plane")
      mitk::Point4dProperty::Pointer plane = mitk::Point4dProperty::New();
      node->SetProperty(CLIP_UNIT("Plane"), plane);
      plane->SetValue( vec4(1,0,0,0).ptr() );

      // gocUniform("vl_Vivid.smartClip[0].sphere")
      mitk::Point4dProperty::Pointer sphere = mitk::Point4dProperty::New();
      node->SetProperty(CLIP_UNIT("Sphere"), sphere);
      sphere->SetValue( vec4(0, 0, 0, 250).ptr() );

      // gocUniform("vl_Vivid.smartClip[0].boxMin")
      mitk::Point3dProperty::Pointer box_min = mitk::Point3dProperty::New();
      node->SetProperty(CLIP_UNIT("BoxMin"), box_min);
      box_min->SetValue( vec3(-100,-100,-100).ptr() );

      // gocUniform("vl_Vivid.smartClip[0].boxMax")
      mitk::Point3dProperty::Pointer box_max = mitk::Point3dProperty::New();
      node->SetProperty(CLIP_UNIT("BoxMax"), box_max);
      box_max->SetValue( vec3(+100,+100,+100).ptr() );

      // gocUniform("vl_Vivid.smartClip[0].reverse")
      mitk::BoolProperty::Pointer reverse = mitk::BoolProperty::New();
      node->SetProperty(CLIP_UNIT("Reverse"), reverse);
      reverse->SetValue( false );
    }

    #undef CLIP_UNIT
  }

  void updateClipProps( Effect* fx, const mitk::DataNode* node )
  {
    #define CLIP_UNIT(field) (std::string("VL.Clip.") + i + '.' + field).c_str()
    #define CLIP_UNIT2(field) (std::string("vl_Vivid.smartClip[") + i + "]." + field).c_str()

    for( char i = '0'; i < '4'; ++i ) {

      int mode = getEnumProp( node, CLIP_UNIT("Mode"), 0 );
      int targ = getEnumProp( node, CLIP_UNIT("Target"), 0 );
      vl::vec4 color = getColorProp( node, CLIP_UNIT("Color"), vl::black );
      float range = getFloatProp( node, CLIP_UNIT("FadeRange"), 0 );
      vl::vec4 plane = getPoint4DProp( node, CLIP_UNIT("Plane"), vl::vec4(0,0,0,0) );
      vl::vec4 sphere = getPoint4DProp( node, CLIP_UNIT("Sphere"), vl::vec4(0,0,0,0) );
      vl::vec3 bmin = getPoint3DProp( node, CLIP_UNIT("BoxMin"), vl::vec3(0,0,0) );
      vl::vec3 bmax = getPoint3DProp( node, CLIP_UNIT("BoxMax"), vl::vec3(0,0,0) );
      bool reverse = getBoolProp( node, CLIP_UNIT("Reverse"), false );

      Shader* sh = fx->shader();

      sh->gocUniform(CLIP_UNIT2("mode"))->setUniformI( mode );
      sh->gocUniform(CLIP_UNIT2("target"))->setUniformI( targ );
      sh->gocUniform(CLIP_UNIT2("color"))->setUniform( color  );
      sh->gocUniform(CLIP_UNIT2("fadeRange"))->setUniform( range );
      sh->gocUniform(CLIP_UNIT2("plane"))->setUniform( plane );
      sh->gocUniform(CLIP_UNIT2("sphere"))->setUniform( sphere );
      sh->gocUniform(CLIP_UNIT2("boxMin"))->setUniform( bmin );
      sh->gocUniform(CLIP_UNIT2("boxMax"))->setUniform( bmax );
      sh->gocUniform(CLIP_UNIT2("reverse"))->setUniformI( reverse);
    }

    #undef CLIP_UNIT
    #undef CLIP_UNIT2
  }

  void initRenderModeProps( mitk::DataNode* node )
  {
    // init only once if multiple views are open
    if ( node->GetProperty("VL.SurfaceMode") ) {
      return;
    }

    // gocUniform("vl_Vivid.renderMode")
    mitk::EnumerationProperty::Pointer mode = VL_Surface_Mode_Property::New();
    node->SetProperty("VL.SurfaceMode", mode);
    mode->SetValue( 0 );

    // gocUniform("vl_Vivid.outline.color")
    mitk::ColorProperty::Pointer outline_color = mitk::ColorProperty::New();
    node->SetProperty("VL.Outline.Color", outline_color);
    outline_color->SetValue( vl::yellow.ptr() );

    // gocUniform("vl_Vivid.outline.width")
    mitk::IntProperty::Pointer outline_width = mitk::IntProperty::New();
    node->SetProperty("VL.Outline.Width", outline_width);
    outline_width->SetValue( 2 );

    // gocUniform("vl_Vivid.outline.slicePlane")
    mitk::Point4dProperty::Pointer outline_slice_plane = mitk::Point4dProperty::New();
    node->SetProperty("VL.Outline.SlicePlane", outline_slice_plane);
    outline_slice_plane->SetValue( vec4(1,0,0,0).ptr() );
  }

  void updateRenderModeProps( Effect* fx, const mitk::DataNode* node ) {
    int mode = getEnumProp( node, "VL.SurfaceMode", 0 );
#if 0
    vec4 color = getColorProp( node, "VL.Outline.Color", vl::yellow );
#else
    vec4 color = getColorProp( node, "color" );
    color.a() = getFloatProp( node, "opacity" );
#endif
    int width = getIntProp( node, "VL.Outline.Width", 2 );
    vec4 slice_plane = getPoint4DProp( node, "VL.Outline.SlicePlane", vec4(0,0,0,0) );

    Shader* sh = fx->shader();

    sh->getUniform("vl_Vivid.renderMode")->setUniformI( mode );
    sh->getUniform("vl_Vivid.outline.color")->setUniform( color );
    sh->getUniform("vl_Vivid.outline.width")->setUniformF( (float)width );
    sh->getUniform("vl_Vivid.outline.slicePlane")->setUniform( slice_plane );
  }

  vl::EImageType MapITKPixelTypeToVL(int itkComponentType)
  {
    static const vl::EImageType typeMap[] =
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

  vl::EImageFormat MapComponentsToVLColourFormat(int components)
  {
    // this assumes the image data is a normal colour image, not encoding pointers or indices, or similar stuff.

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

  ref<vl::Image> wrapMitk2DImage( const mitk::Image* mitk_image ) {
    mitk::PixelType  mitk_pixel_type = mitk_image->GetPixelType();
    vl::EImageType   vl_type         = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
    vl::EImageFormat vl_format       = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
    unsigned int*    dims            = mitk_image->GetDimensions();
    VIVID_CHECK( dims[2] == 1 );

    try
    {
      unsigned int buffer_bytes = dims[0] * dims[1] * dims[2] * mitk_pixel_type.GetSize();
      mitk::ImageReadAccessor image_reader( mitk_image, mitk_image->GetVolumeData(0) );
      void* buffer_ptr = const_cast<void*>( image_reader.GetData() );
      // Use VTK buffer directly, no VL image allocation needed
      ref<vl::Image> vl_img = new vl::Image( buffer_ptr, buffer_bytes );
      vl_img->allocate2D(dims[0], dims[1], 1, vl_format, vl_type);
      VIVID_CHECK( vl_img->requiredMemory() == buffer_bytes );
      return vl_img;
    }
    catch (...)
    {
      MITK_ERROR << "Did not get pixel read access to 2D image.";
      return NULL;
    }
  }

  //-----------------------------------------------------------------------------

  VLUserData* GetUserData(vl::Actor* actor)
  {
    VIVID_CHECK( actor );
    ref<VLUserData> userdata = actor->userData()->as<VLUserData>();
    if ( ! userdata )
    {
      userdata = new VLUserData;
      actor->setUserData( userdata.get() );
    }

    return userdata.get();
  }

  //-----------------------------------------------------------------------------

  vl::mat4 GetVLMatrixFromData(const mitk::BaseData::ConstPointer& data)
  {
    vl::mat4  mat;
    // Intentionally not setIdentity()
    mat.setNull();

    if ( data )
    {
      mitk::BaseGeometry::Pointer geom = data->GetGeometry();
      if ( geom ) {
        if ( geom->GetVtkTransform() ) {
          vtkSmartPointer<vtkMatrix4x4> vtkmat = vtkSmartPointer<vtkMatrix4x4>::New();
          geom->GetVtkTransform()->GetMatrix(vtkmat);
          if ( vtkmat.GetPointer() ) {
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 4; j++) {
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

  void UpdateTransformFromData(vl::Transform* tr, const mitk::BaseData::ConstPointer& data)
  {
    vl::mat4 m = GetVLMatrixFromData(data);

    if ( ! m.isNull() )
    {
      tr->setLocalMatrix(m);
      tr->computeWorldMatrix();
#if 0
      printf("Transform: %p\n", tr );
      for(int i = 0; i < 4; ++i ) {
        printf("%f %f %f %f\n", m.e(0,i), m.e(1,i), m.e(2,i), m.e(3,i) );
      }
#endif
    }
  }

  //-----------------------------------------------------------------------------

  void UpdateActorTransformFromNode( vl::Actor* actor, const mitk::DataNode* node )
  {
    if ( ! node ) {
      return;
    }
    const mitk::BaseData* data = node->GetData();
    if ( ! data ) {
      return;
    }
    const mitk::BaseGeometry* geom = data->GetGeometry();
    if ( ! geom ) {
      return;
    }
    VLUserData* userdata = GetUserData( actor );
    if ( geom->GetMTime() > userdata->m_TransformModifiedTime )
    {
      UpdateTransformFromData( actor->transform(), data );
      userdata->m_TransformModifiedTime = geom->GetMTime();
    }
  }

  //-----------------------------------------------------------------------------

  void UpdateTransformFromNode(vl::Transform* txf, const mitk::DataNode* node)
  {
    if ( node ) {
      UpdateTransformFromData(txf, node->GetData());
    }
  }

  //-----------------------------------------------------------------------------

  ref<vl::Geometry> CreateGeometryFor2DImage(int width, int height)
  {
    ref<vl::Geometry>    geom = new vl::Geometry;
    ref<vl::ArrayFloat3> vert  = new vl::ArrayFloat3;
    vert->resize(4);
    geom->setVertexArray( vert.get() );

    ref<vl::ArrayFloat3> tex_coord = new vl::ArrayFloat3;
    tex_coord->resize(4);
    geom->setTexCoordArray(0, tex_coord.get());

    //  1---2 image-top
    //  |   |
    //  0---3 image-bottom

    vert->at(0).x() = 0;     vert->at(0).y() = 0;      vert->at(0).z() = 0; tex_coord->at(0).s() = 0; tex_coord->at(0).t() = 1;
    vert->at(1).x() = 0;     vert->at(1).y() = height; vert->at(1).z() = 0; tex_coord->at(1).s() = 0; tex_coord->at(1).t() = 0;
    vert->at(2).x() = width; vert->at(2).y() = height; vert->at(2).z() = 0; tex_coord->at(2).s() = 1; tex_coord->at(2).t() = 0;
    vert->at(3).x() = width; vert->at(3).y() = 0;      vert->at(3).z() = 0; tex_coord->at(3).s() = 1; tex_coord->at(3).t() = 1;

    ref<vl::DrawArrays> polys = new vl::DrawArrays(vl::PT_QUADS, 0, 4);
    geom->drawCalls().push_back( polys.get() );

    return geom;
  }

  //-----------------------------------------------------------------------------

  ref<vl::Geometry> ConvertVTKPolyData(vtkPolyData* vtkPoly)
  {
    if ( ! vtkPoly ) {
      return NULL;
    }

    ref<vl::Geometry> vl_geom = new vl::Geometry;

    vtkSmartPointer<vtkPoints> points = vtkPoly->GetPoints();
    if ( ! points )
    {
      MITK_ERROR << "No points in vtkPolyData. Skipping.\n";
      return NULL;
    }

    if ( ! vtkPoly->GetPointData() )
    {
      MITK_ERROR << "No points data in the vtkPolyData data. Skipping.\n";
      return NULL;
    }

    // Build the cell data if not present already
    int cell_array_count = vtkPoly->GetCellData()->GetNumberOfArrays();
    if ( cell_array_count == 0 ) {
      vtkPoly->BuildCells();
    }

    vtkSmartPointer<vtkCellArray> primitives = NULL;

    // For the moment we only support tris/quads/polygons (no strips, points and lines).
    // Supporting other types is possible with some care. I just didn't have any data to test.
    // Things to look out for:
    // - AdjacencyExtractor expects polygons/tris/strips and transforms them into triangles: skip it in this case.
    // - Compute normals also expects triangles: skip it in this case.
    // - 3D Outline rendering geometry shader also expects triangles: skip it in this case.
    // - Maybe something else...

    // Access primitive/cell data
    if ( vtkPoly->GetPolys() && vtkPoly->GetPolys()->GetNumberOfCells() ) {
      primitives = vtkPoly->GetPolys();
      MITK_INFO << "vtkPolyData polygons found.\n";
    }
    else
    if ( vtkPoly->GetStrips() && vtkPoly->GetStrips()->GetNumberOfCells() ) {
      primitives = vtkPoly->GetStrips();
      MITK_ERROR << "vtkPolyData strips not supported. Skipping.\n";
      return NULL;
    }
    else
    if ( vtkPoly->GetVerts() && vtkPoly->GetVerts()->GetNumberOfCells() ) {
      primitives = vtkPoly->GetVerts();
      MITK_ERROR << "vtkPolyData verts not supported. Skipping.\n";
      return NULL;
    }
    else
    if ( vtkPoly->GetLines() && vtkPoly->GetLines()->GetNumberOfCells() ) {
      primitives = vtkPoly->GetLines();
      MITK_ERROR << "vtkPolyData lines not supported. Skipping.\n";
      return NULL;
    }

    if ( ! primitives )
    {
      MITK_ERROR << "No primitive found in vtkPolyData data. Skipping.\n";
      return NULL;
    }

    // NOTE:
    // We now support a list of eterogeneous polygons of any size thanks to VL's primitive restart and triangle iterator.
    int max_cell_size = primitives->GetMaxCellSize();

    unsigned int point_buffer_size = 0;
    unsigned int point_count = static_cast<unsigned int> (points->GetNumberOfPoints());
    point_buffer_size = point_count * sizeof(float) * 3;

    // setup vertices
    ref<vl::ArrayFloat3> vl_verts = new vl::ArrayFloat3;
    vl_verts->resize(point_count);
    memcpy(vl_verts->ptr(), points->GetVoidPointer(0), point_buffer_size);
    vl_geom->setVertexArray(vl_verts.get());

    // setup normals
    vtkSmartPointer<vtkDataArray> normals = vtkPoly->GetPointData()->GetNormals();
    if ( normals )
    {
      // Get the number of normals we have to deal with
      int normal_count = (int)normals->GetNumberOfTuples();
      if ( normal_count == point_count )
      {
        ref<vl::ArrayFloat3> vl_normals = new vl::ArrayFloat3;
        vl_normals->resize(point_count);
        memcpy(vl_normals->ptr(), normals->GetVoidPointer(0), point_buffer_size);
        vl_geom->setNormalArray(vl_normals.get());
      }
      else
      {
        MITK_ERROR << "Invalid normals for vtkPolyData. VL will recompute them.\n";
        MITK_ERROR << "normal_count: " << normal_count << " vs point_count: " << point_count << "\n";
        normals = NULL;
      }
    }

    // setup triangles index buffer

    int primitive_count = (int)primitives->GetNumberOfCells();

    ref<vl::DrawElementsUInt> vl_draw_elements = new vl::DrawElementsUInt( vl::PT_POLYGON );
    vl_draw_elements->setPrimitiveRestartEnabled(true);
    vl_geom->drawCalls().push_back( vl_draw_elements.get() );
    std::vector< vl::DrawElementsUInt::index_type > indices;
    indices.reserve( primitive_count * max_cell_size );

    // copy triangles
    primitives->InitTraversal();
    for( int cell_index = 0; cell_index < primitive_count; ++cell_index )
    {
      vtkIdType npts = 0;
      vtkIdType *pts = 0;
      primitives->GetNextCell( npts, pts );
      // mark the start of a new primitive
      if ( cell_index != 0 ) {
        indices.push_back( vl::DrawElementsUInt::primitive_restart_index );
      }
      for (vtkIdType i = 0; i < npts; ++i) {
        VIVID_CHECK( pts[i] < vl_verts->size() );
        indices.push_back( pts[i] );
      }
    }
    if ( indices.empty() ) {
      MITK_ERROR << "No polygons found. Skipping.\n";
      return NULL;
    }
    vl_draw_elements->indexBuffer()->resize( indices.size() );
    memcpy( vl_draw_elements->indexBuffer()->ptr(), &indices[0], indices.size() * sizeof(indices[0]) );

    MITK_INFO << "Computing surface adjacency... ";

    vl_geom = vl::AdjacencyExtractor::extract( vl_geom.get() );

    vl_draw_elements = vl_geom->drawCalls().at(0)->as<vl::DrawElementsUInt>();

    MITK_INFO << "Surface data initialized. Points: " << points->GetNumberOfPoints() << ", Cells: " << primitives->GetNumberOfCells() << "\n";

    return vl_geom;
  }

  //-----------------------------------------------------------------------------

  void dumpNodeInfo( const std::string& prefix, const mitk::DataNode* node ) {
    printf( "\n%s: ", prefix.c_str() );
    const char* class_name = node->GetData() ? node->GetData()->GetNameOfClass() : "<unknown-class>";
    mitk::StringProperty* name_prop = dynamic_cast<mitk::StringProperty*>(node->GetProperty("name"));
    const char* object_name2 = "<unknown-name>";
    if (name_prop != 0) {
      object_name2 = name_prop->GetValue();
    }
    printf( "%s <%s>\n", object_name2, class_name );

    const mitk::PropertyList::PropertyMap* propList = node->GetPropertyList()->GetMap();
    mitk::PropertyList::PropertyMap::const_iterator it = node->GetPropertyList()->GetMap()->begin();
    for( ; it != node->GetPropertyList()->GetMap()->end(); ++it ) {
      const std::string name = it->first;
      const mitk::BaseProperty::Pointer prop = it->second;
      printf( "\t%s: %s <%s>\n", name.c_str(), prop->GetValueAsString().c_str(), prop->GetNameOfClass() );
    }
  }

  void dumpNodeInfo( const std::string& prefix, const mitk::BaseData::ConstPointer& data ) {
    printf( "\n%s: ", prefix.c_str() );
    const char* class_name = data->GetNameOfClass();
    std::string object_name2 = data->GetProperty("name") ? data->GetProperty("name")->GetValueAsString() : "<unknown-name>";
    printf( "%s <%s>\n", object_name2.c_str(), class_name );

    const mitk::PropertyList::PropertyMap* propList = data->GetPropertyList()->GetMap();
    mitk::PropertyList::PropertyMap::const_iterator it = data->GetPropertyList()->GetMap()->begin();
    for( ; it != data->GetPropertyList()->GetMap()->end(); ++it ) {
      const std::string name = it->first;
      const mitk::BaseProperty::Pointer prop = it->second;
      printf( "\t%s: %s <%s>\n", name.c_str(), prop->GetValueAsString().c_str(), prop->GetNameOfClass() );
    }
  }
}

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
// VLGlobalSettingsDataNode
//-----------------------------------------------------------------------------

class VLDummyData: public mitk::BaseData
{
public:
  mitkClassMacro(VLDummyData, BaseData);
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)
protected:
  virtual bool VerifyRequestedRegion(){return false;};
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion(){return false;};
  virtual void SetRequestedRegionToLargestPossibleRegion(){};
  virtual void SetRequestedRegion( const itk::DataObject * /*data*/){};
};

class VLGlobalSettingsDataNode: public mitk::DataNode
{
public:
  mitkClassMacro(VLGlobalSettingsDataNode, DataNode);
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  VLGlobalSettingsDataNode() {
    SetName( VLGlobalSettingsName() );
    // Needs dummy data otherwise it doesn't show up
    mitk::BaseData::Pointer data = VLDummyData::New();
    SetData( data.GetPointer() );

    initGlobalProperties();
  }

  static const char* VLGlobalSettingsName() { return "VL Debug"; }

protected:
  void initGlobalProperties()
  {
    // initRenderModeProps(this);
    // initFogProps(this);
    // initClipProps(this);

    // Truly globals

    mitk::BoolProperty::Pointer enable = mitk::BoolProperty::New();
    AddProperty( "VL.Global.Stencil.Enable", enable );
    enable->SetValue( false );

    mitk::ColorProperty::Pointer stencil_bg_color = mitk::ColorProperty::New();
    AddProperty( "VL.Global.Stencil.BackgroundColor", stencil_bg_color );
    stencil_bg_color->SetValue( vl::black.ptr() );

    mitk::FloatProperty::Pointer stencil_smooth = mitk::FloatProperty::New();
    AddProperty( "VL.Global.Stencil.Smoothness", stencil_smooth );
    stencil_smooth->SetValue( 10 );

    mitk::EnumerationProperty::Pointer render_mode = VL_Render_Mode_Property::New();
    AddProperty( "VL.Global.RenderMode", render_mode );
    render_mode->SetValue( 0 );

    mitk::ColorProperty::Pointer bg_color = mitk::ColorProperty::New();
    AddProperty( "VL.Global.BackgroundColor", bg_color );
    bg_color->SetValue( vl::lightgray.ptr() );

    mitk::FloatProperty::Pointer opacity = mitk::FloatProperty::New();
    AddProperty( "VL.Global.Opacity", opacity );
    opacity->SetValue( 1 );

    mitk::IntProperty::Pointer passes = mitk::IntProperty::New();
    AddProperty( "VL.Global.DepthPeelingPasses", passes );
    passes->SetValue( 4 );
  }
};

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

VLMapper::VLMapper( const mitk::DataNode* node, VLSceneView* sv ) {
  // Init
  VIVID_CHECK( node );
  VIVID_CHECK( sv );
  m_DataNode = node;
  m_VLSceneView = sv;
  m_OpenGLContext = sv->openglContext();
  m_VividRendering = sv->vividRendering();
  m_DataStorage = sv->dataStorage();
  m_DataNodeVividUpdateEnabled = true;
  VIVID_CHECK( m_OpenGLContext );
  VIVID_CHECK( m_VividRendering );
  VIVID_CHECK( m_DataStorage );
}

//-----------------------------------------------------------------------------

vl::ref<vl::Actor> VLMapper::initActor(vl::Geometry* geom, vl::Effect* effect, vl::Transform* transform) {
  VIVID_CHECK( m_DataNode );
  VIVID_CHECK( m_VividRendering );
  ref<vl::Effect> fx = effect ? effect : vl::VividRendering::makeVividEffect();
  ref<vl::Transform> tr = transform ? transform : new vl::Transform;
  UpdateTransformFromData( tr.get(), m_DataNode->GetData() );
  ref<vl::Actor> actor = new vl::Actor( geom, fx.get(), tr.get() );
  actor->setEnableMask( vl::Vivid::VividEnableMask );
  return actor;
}

//-----------------------------------------------------------------------------

void VLMapper::updateCommon() {
  if ( ! m_Actor ) {
    return;
  }

  // Update visibility
  bool visible = getBoolProp( m_DataNode, "visible", true );
  m_Actor->setEnabled( visible );

  // Update transform
  UpdateTransformFromData( m_Actor->transform(), m_DataNode->GetData() );
}

//-----------------------------------------------------------------------------

class VLMapperVLGlobalSettings: public VLMapper {
public:
  VLMapperVLGlobalSettings( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_VLGlobalSettings = dynamic_cast<const VLGlobalSettingsDataNode*>( node );
  }

  virtual bool init() { return true; }

  virtual void update() {
    bool enable = getBoolProp( m_DataNode, "VL.Global.Stencil.Enable", false );
    vec4 stencil_bg_color = getColorProp( m_DataNode, "VL.Global.Stencil.BackgroundColor", vl::black );
    float stencil_smooth = getFloatProp( m_DataNode, "VL.Global.Stencil.Smoothness", 10 );
    int render_mode = getEnumProp( m_DataNode, "VL.Global.RenderMode", 0 );
    vec4 bg_color = getColorProp( m_DataNode, "VL.Global.BackgroundColor", vl::black );
    float opacity = getFloatProp( m_DataNode, "VL.Global.Opacity", 1 );
    int passes = getIntProp( m_DataNode, "VL.Global.DepthPeelingPasses", 4 );

    m_VividRendering->setStencilEnabled( enable );
    m_VividRendering->setStencilBackground( stencil_bg_color );
    m_VividRendering->setStencilSmoothness( stencil_smooth );
    m_VividRendering->setRenderingMode( (vl::Vivid::ERenderingMode)render_mode );
    m_VividRendering->setBackgroundColor( bg_color );
    m_VividRendering->setOpacity( opacity );
    m_VividRendering->vividRenderer()->setNumPasses( passes );
  }

  virtual void updateVLGlobalSettings() { /* we don't have anything to set */ }

protected:
  VLGlobalSettingsDataNode::ConstPointer m_VLGlobalSettings;
};

//-----------------------------------------------------------------------------

class VLMapperSurface: public VLMapper {
public:
  VLMapperSurface( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkSurf = dynamic_cast<mitk::Surface*>( node->GetData() );
    VIVID_CHECK( m_MitkSurf );
  }

  virtual bool init() {
    VIVID_CHECK( m_MitkSurf );

    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    initRenderModeProps( node );
    initMaterialProps( node );
    initFogProps( node );
    initClipProps( node );

    ref<vl::Geometry> geom = ConvertVTKPolyData( m_MitkSurf->GetVtkPolyData() );
    if ( ! geom ) {
      return false;
    }

    // in VL if verts are shared across primitives they're smoothed out, VTK however seem to keep them flat.
    if ( ! geom->normalArray() ) {
      geom->computeNormals();
    }

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );

    return true;
  }

  virtual void update() {
    updateCommon();
    if ( isDataNodeVividUpdateEnabled() ) {
      updateMaterialProps( m_Actor->effect(), m_DataNode );
      updateRenderModeProps( m_Actor->effect(), m_DataNode );
      updateFogProps( m_Actor->effect(), m_DataNode );
      updateClipProps( m_Actor->effect(), m_DataNode );

      // Stencil
      bool is_stencil = getBoolProp( m_DataNode, "VL.IsStencil", false );
      setIsStencil( is_stencil );
    }
  }

protected:
  const mitk::Surface* m_MitkSurf;
};

//-----------------------------------------------------------------------------

class VLMapper2DImage: public VLMapper {
public:
  VLMapper2DImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    VIVID_CHECK( m_MitkImage );
  }

  virtual bool init() {
    VIVID_CHECK( m_MitkImage );

    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    // initRenderModeProps( node ); /* does not apply */
    initFogProps( node );
    initClipProps( node );

    ref<vl::Image> img = wrapMitk2DImage( m_MitkImage );
    ref<vl::Geometry> geom = CreateGeometryFor2DImage( img->width(), img->height() );

    m_VertexArray = geom->vertexArray()->as<vl::ArrayFloat3>(); VIVID_CHECK( m_VertexArray );
    m_TexCoordArray = geom->vertexArray()->as<vl::ArrayFloat3>(); VIVID_CHECK( m_TexCoordArray );

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    // These must be present as part of the default Vivid material
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::Vivid::UserTexture ) )
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture() )
    VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture2D") );
    VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture2D")->getUniformI() == vl::Vivid::UserTexture );
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
    texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );
    fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
    fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
    // When texture mapping is enabled the texture is modulated by the vertex color, including the alpha
    geom->setColorArray( vl::white );

    return true;
  }

  virtual void update() {
    VIVID_CHECK( m_MitkImage );

    updateCommon();
    if ( isDataNodeVividUpdateEnabled() ) {
      // updateRenderModeProps(); /* does not apply here */
      updateFogProps( m_Actor->effect(), m_DataNode );
      updateClipProps( m_Actor->effect(), m_DataNode );
    }

    if ( m_MitkImage->GetVtkImageData()->GetMTime() <= GetUserData( m_Actor.get() )->m_ImageModifiedTime ) {
      return;
    }

    vl::Texture* tex = m_Actor->effect()->shader()->gocTextureSampler( vl::Vivid::UserTexture )->texture();
    VIVID_CHECK( tex );
    ref<vl::Image> img = wrapMitk2DImage( m_MitkImage );
    tex->setMipLevel(0, img.get(), false);
    GetUserData( m_Actor.get() )->m_ImageModifiedTime = m_MitkImage->GetVtkImageData()->GetMTime();
  }

  Texture* texture() { return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture(); }
  const Texture* texture() const { return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture(); }

  //! This vertex array contains 4 points representing the plane
  ArrayFloat3* vertexArray() { return m_VertexArray.get(); }
  const ArrayFloat3* vertexArray() const { return m_VertexArray.get(); }

  //! This texture coordinates array contains 4 3D texture coordinates one for each plane corner
  ArrayFloat3* texCoordarray() { return m_TexCoordArray.get(); }
  const ArrayFloat3* texCoordarray() const { return m_TexCoordArray.get(); }

protected:
  mitk::Image* m_MitkImage;
  ref<ArrayFloat3> m_VertexArray;
  ref<ArrayFloat3> m_TexCoordArray;
};

//-----------------------------------------------------------------------------

class VLMapper3DImage: public VLMapper {
public:
  VLMapper3DImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    m_VividVolume = new vl::VividVolume( m_VividRendering );
    VIVID_CHECK( m_MitkImage );
  }

  virtual bool init() {
    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    initVolumeProps( node );

    mitk::PixelType mitk_pixel_type = m_MitkImage->GetPixelType();

#if 1
    std::cout << "MITK pixel type:"       << std::endl;
    std::cout << "\tPixelType: "          << mitk_pixel_type.GetTypeAsString() << std::endl;
    std::cout << "\tBitsPerElement: "     << mitk_pixel_type.GetBpe() << std::endl;
    std::cout << "\tNumberOfComponents: " << mitk_pixel_type.GetNumberOfComponents() << std::endl;
    std::cout << "\tBitsPerComponent: "   << mitk_pixel_type.GetBitsPerComponent() << std::endl;
#endif

    unsigned int* dims = dims = m_MitkImage->GetDimensions();
    VIVID_CHECK( dims[2] > 1 );
    ref<vl::Image> vl_img;

    try
    {
      mitk::ImageReadAccessor image_reader( m_MitkImage, m_MitkImage->GetVolumeData(0) );
      void* img_ptr = const_cast<void*>( image_reader.GetData() );
      unsigned int buffer_bytes = (dims[0] * dims[1] * dims[2]) * mitk_pixel_type.GetSize();

      vl::EImageType   vl_type   = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
      vl::EImageFormat vl_format = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());

      // Don't allocate the image, use VTK buffer directly
      vl_img = new vl::Image(img_ptr, buffer_bytes);
      vl_img->allocate3D(dims[0], dims[1], dims[2], 1, vl_format, vl_type);
      VIVID_CHECK(vl_img->requiredMemory() == buffer_bytes);

      // FIXME: at the moment we only support 1 channel with 0..1 float data
      // NOTE:
      // - This creates and destroys one temp image per conversion
      // - Within the volume rendering shader values are all mapped to 0..1 however we could
      //   pass a `vec2 vl_Vivid.volume.dataRange` uniform to inform the shader of what the
      //   original data rage was so we can map back to it. We could use this range also to map
      //   the Iso value. We could use this to support more easily things like Hounsfield units etc.
      vl_img = vl_img->convertFormat( vl::IF_LUMINANCE )->convertType( vl::IT_FLOAT );
    }
    catch(mitk::Exception& e)
    {
      // deal with the situation not to have access
      VIVID_CHECK( false );
    }

    vl::vec3 origin(0,0,0), spacing(1,1,1);
    if ( m_MitkImage->GetGeometry() ) {
      origin.x()  = m_MitkImage->GetGeometry()->GetOrigin()[0];
      origin.y()  = m_MitkImage->GetGeometry()->GetOrigin()[1];
      origin.z()  = m_MitkImage->GetGeometry()->GetOrigin()[2];
      spacing.x() = m_MitkImage->GetGeometry()->GetSpacing()[0];
      spacing.y() = m_MitkImage->GetGeometry()->GetSpacing()[1];
      spacing.z() = m_MitkImage->GetGeometry()->GetSpacing()[2];
    }

    float vx = dims[0] * spacing.x() / 2.0f;
    float vy = dims[1] * spacing.y() / 2.0f;
    float vz = dims[2] * spacing.z() / 2.0f;

    vl::AABB volume_box( vl::vec3(-vx + origin.x(), -vy + origin.y(), -vz + origin.z() ),
                         vl::vec3( vx + origin.x(),  vy + origin.y(),  vz + origin.z() ) );

    m_VividVolume->setupVolume( vl_img.get(), volume_box, NULL);
    m_Actor = m_VividVolume->volumeActor();
    m_VividRendering->sceneManager()->tree()->eraseActor( m_Actor.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );

#if 0
    vtkLinearTransform * nodeVtkTr = m_MitkImage->GetGeometry()->GetVtkTransform();
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
#endif
    return true;
  }

  virtual void update() {
    updateCommon();
    // Neutralize scaling - screws up our rendering.
    // VTK seems to need it to render non cubic volumes.
    // NOTE: we assume there is no rotation.
    m_Actor->transform()->localMatrix().e(0,0) =
    m_Actor->transform()->localMatrix().e(1,1) =
    m_Actor->transform()->localMatrix().e(2,2) = 1;
    m_Actor->transform()->computeWorldMatrix();
    updateVolumeProps( m_VividVolume.get(), m_DataNode );
  }

protected:
  const mitk::Image* m_MitkImage;
  ref<vl::VividVolume> m_VividVolume;
};

//-----------------------------------------------------------------------------

class VLMapperCoordinateAxes: public VLMapper {
public:
  VLMapperCoordinateAxes( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkAxes = dynamic_cast<mitk::CoordinateAxesData*>( node->GetData() );
    VIVID_CHECK( m_MitkAxes );
  }

  virtual bool init() {
    VIVID_CHECK( m_MitkAxes );

    ref<vl::ArrayFloat3> verts  = m_Vertices = new vl::ArrayFloat3;
    ref<vl::ArrayFloat4> colors = new vl::ArrayFloat4;
    verts->resize(6);
    colors->resize(6);

    // Axis length
    int S = 100;
    mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
    if ( size_prop ) {
      S = size_prop->GetValue();
    }

    // X Axis
    verts ->at(0) = vec3(0, 0, 0);
    verts ->at(1) = vec3(S, 0, 0);
    colors->at(0) = vl::red;
    colors->at(1) = vl::red;
    // Y Axis
    verts ->at(2) = vec3(0, 0, 0);
    verts ->at(3) = vec3(0, S, 0);
    colors->at(2) = vl::green;
    colors->at(3) = vl::green;
    // Z Axis
    verts ->at(4) = vec3(0, 0, 0);
    verts ->at(5) = vec3(0, 0, S);
    colors->at(4) = vl::blue;
    colors->at(5) = vl::blue;

    ref<vl::Geometry> geom = new vl::Geometry;
    geom->drawCalls().push_back( new vl::DrawArrays( vl::PT_LINES, 0, 6 ) );
    geom->setVertexArray(verts.get());
    geom->setColorArray(colors.get());

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    fx->shader()->getLineWidth()->set( 2 );
    // Use color array instead of lighting
    fx->shader()->gocUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );

    return true;
  }

  virtual void update() {
    updateCommon();
    // updateRenderModeProps();
    // updateFogProps();
    // updateClipProps();

    mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
    if ( size_prop ) {
      ref<vl::ArrayFloat3> verts = m_Vertices;
      int S = size_prop->GetValue();
      // X Axis
      verts ->at(0) = vec3(0, 0, 0);
      verts ->at(1) = vec3(S, 0, 0);
      // Y Axis
      verts ->at(2) = vec3(0, 0, 0);
      verts ->at(3) = vec3(0, S, 0);
      // Z Axis
      verts ->at(4) = vec3(0, 0, 0);
      verts ->at(5) = vec3(0, 0, S);
      // Update VBO
      m_Vertices->updateBufferObject();
    }
  }

protected:
  const mitk::CoordinateAxesData* m_MitkAxes;
  ref<vl::ArrayFloat3> m_Vertices;
};

//-----------------------------------------------------------------------------

class VLMapperPoints: public VLMapper {
public:
  VLMapperPoints( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_3DSphereMode = true;
    m_Point2DFX = vl::VividRendering::makeVividEffect();
    m_PositionArray = new vl::ArrayFloat3;
    m_ColorArray = new vl::ArrayFloat4;
    m_DrawPoints = new vl::DrawArrays( vl::PT_POINTS, 0, 0 );
  }

  virtual void updatePoints( const vl::vec4& color ) = 0 ;

  void initPointSetProps()
  {
    // init only once if multiple views are open
    if ( m_DataNode->GetProperty("VL.Point.Mode") ) {
      return;
    }

    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    // initRenderModeProps( node ); /* does not apply to points */
    initFogProps( node );
    initClipProps( node );

    mitk::EnumerationProperty::Pointer point_set_mode = VL_Point_Mode_Property::New();
    const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Mode", point_set_mode);
    point_set_mode->SetValue( 0 );

    mitk::FloatProperty::Pointer point_size_2d = mitk::FloatProperty::New();
    const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Size2D", point_size_2d);
    point_size_2d->SetValue( 5 );

    mitk::FloatProperty::Pointer point_size_3d = mitk::FloatProperty::New();
    const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Size3D", point_size_3d);
    point_size_3d->SetValue( 5 );

    mitk::FloatProperty::Pointer point_opacity = mitk::FloatProperty::New();
    const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Opacity", point_opacity);
    point_opacity->SetValue( 1 );

    mitk::ColorProperty::Pointer point_color = mitk::ColorProperty::New();
    const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Color", point_color);
    point_color->SetValue( vl::yellow.ptr() );
  }

  virtual bool init() { initPointSetProps(); return true; }

  void init3D() {
    VIVID_CHECK( m_3DSphereMode );

    // Remove 2D data and init 3D data.
    remove();
    m_SphereActors = new vl::ActorTree;
    m_VividRendering->sceneManager()->tree()->addChild( m_SphereActors.get() );

    m_3DSphereGeom = vl::makeIcosphere( vec3(0,0,0), 1, 2, true );
    for( int i = 0; i < m_PositionArray->size(); ++i )
    {
      const vl::vec3& pos = m_PositionArray->at( i );
      ref<Actor> actor = initActor( m_3DSphereGeom.get() );
      actor->transform()->setLocalAndWorldMatrix( vl::mat4::getTranslation( pos ) );
      m_SphereActors->addActor( actor.get() );
      // Colorize the sphere with the point's color
      actor->effect()->shader()->getMaterial()->setDiffuse( m_ColorArray->at( i ) );
    }
  }

  void init2D() {
    VIVID_CHECK( ! m_3DSphereMode );

    // Remove 3D data and init 2D data.
    remove();

    // Initialize color array
    for( int i = 0; i < m_ColorArray->size(); ++i ) {
      m_ColorArray->at( i ) = vl::white;
    }

    m_2DGeometry = new vl::Geometry;
    m_DrawPoints = new vl::DrawArrays( vl::PT_POINTS, 0, m_PositionArray->size() );
    m_2DGeometry->drawCalls().push_back( m_DrawPoints.get() );
    m_2DGeometry->setVertexArray( m_PositionArray.get() );
    m_2DGeometry->setColorArray( m_ColorArray.get() );

    m_Actor = initActor( m_2DGeometry.get(), m_Point2DFX.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<vl::Effect> fx = m_Actor->effect();
    ref<vl::Image> img = new Image("/vivid/images/sphere.png");
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
    texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );

    // 2d mode settings
    m_Point2DFX->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );
    m_Point2DFX->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( 1 );
    m_Point2DFX->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( 1 );
  }

  virtual void update() {
    // updateCommon();

    // Get mode
    m_3DSphereMode = 0 == getEnumProp( m_DataNode, "VL.Point.Mode", 0 );

    // Get visibility
    bool visible = getBoolProp( m_DataNode, "visible", true );

    // Get point size
    float pointsize = getFloatProp( m_DataNode, m_3DSphereMode ? "VL.Point.Size3D" : "VL.Point.Size2D", 1.0f );

    // Get color
    vl::vec4 color = getColorProp( m_DataNode, "VL.Point.Color", vl::white );

    // Get opacity
    color.a() = getFloatProp( m_DataNode, "VL.Point.Opacity", 1.0f );

    updatePoints( color );

    if ( m_3DSphereMode ) {
      if ( ! m_SphereActors ) {
        init3D();
      }

      for( int i = 0; i < m_SphereActors->actors()->size(); ++i ) {
        // Set visible
        Actor* act = m_SphereActors->actors()->at( i );
        act->setEnabled( visible );
        // Set color/opacity
        act->effect()->shader()->getMaterial()->setDiffuse( m_ColorArray->at( i ) );
        // Update other Vivid settings
        // updateRenderModeProps(); /* does not apply here */
        updateFogProps( act->effect(), m_DataNode );
        updateClipProps( act->effect(), m_DataNode );
        // Set size
        Transform* tr = act->transform();
        mat4& local = tr->localMatrix();
        local.e(0,0) = pointsize * 2;
        local.e(1,1) = pointsize * 2;
        local.e(2,2) = pointsize * 2;
        tr->computeWorldMatrix();
      }
    } else {
      if ( ! m_2DGeometry ) {
        init2D();
      }

      VIVID_CHECK( m_Actor );
      VIVID_CHECK( m_Point2DFX->shader()->getPointSize() );

      // updateRenderModeProps(); /* does not apply here */
      updateFogProps( m_Point2DFX.get(), m_DataNode );
      updateClipProps( m_Point2DFX.get(), m_DataNode );

      // set point size
      m_Point2DFX->shader()->getPointSize()->set( pointsize );
    }
  }

  void remove() {
    VLMapper::remove();
    m_2DGeometry = NULL;
    if ( m_SphereActors ) {
      m_SphereActors->actors()->clear();
      m_VividRendering->sceneManager()->tree()->eraseChild( m_SphereActors.get() );
      m_SphereActors = NULL;
      m_3DSphereGeom = NULL;
    }
  }

protected:
  bool m_3DSphereMode;
  ref<vl::ActorTree> m_SphereActors;
  ref<Geometry> m_3DSphereGeom;
  ref<Effect> m_Point2DFX;
  ref<vl::Geometry> m_2DGeometry;
  ref<vl::ArrayFloat3> m_PositionArray;
  ref<vl::ArrayFloat4> m_ColorArray;
  ref<vl::DrawArrays> m_DrawPoints;
};

//-----------------------------------------------------------------------------

class VLMapperPointSet: public VLMapperPoints {
public:
  VLMapperPointSet( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapperPoints( node, sv ) {
    m_MitkPointSet = dynamic_cast<mitk::PointSet*>( node->GetData() );
    VIVID_CHECK( m_MitkPointSet );
  }

  virtual void updatePoints( const vl::vec4& color ) {
    VIVID_CHECK( m_MitkPointSet );

    // If point set size changed force a rebuild of the 3D spheres, actors etc.
    // TODO: use event listeners instead of this brute force approach
    if ( m_PositionArray->size() != m_MitkPointSet->GetSize() ) {
      if ( m_3DSphereMode ) {
        remove();
      } else {
        m_DrawPoints->setCount( m_MitkPointSet->GetSize() );
      }
    }

    m_PositionArray->resize( m_MitkPointSet->GetSize() );
    m_ColorArray->resize( m_MitkPointSet->GetSize() );

    int j = 0;
    for ( mitk::PointSet::PointsConstIterator i = m_MitkPointSet->Begin(); i != m_MitkPointSet->End(); ++i, ++j )
    {
      const mitk::PointSet::PointType& p = i->Value();
      m_PositionArray->at( j ) = vl::vec3( p[0], p[1], p[2] );
      m_ColorArray->at( j ) = color;
    }
    m_PositionArray->updateBufferObject();
    m_ColorArray->updateBufferObject();
  }

protected:
  const mitk::PointSet* m_MitkPointSet;
};

//-----------------------------------------------------------------------------

#ifdef _USE_PCL

class VLMapperPCL: public VLMapperPoints {
public:
  VLMapperPCL( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapperPoints( node, sv ) {
    m_NiftkPCL = dynamic_cast<niftk::PCLData*>( node->GetData() );
    VIVID_CHECK( m_NiftkPCL );
  }

  virtual void updatePoints( const vl::vec4& /*color*/ ) {
    VIVID_CHECK( m_NiftkPCL );
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud = m_NiftkPCL->GetCloud();

    // If point set size changed force a rebuild of the 3D spheres, actors etc.
    // TODO: use event listeners instead of this brute force approach
    if ( m_PositionArray->size() != cloud->size() ) {
      if ( m_3DSphereMode ) {
        remove();
      } else {
        m_DrawPoints->setCount( cloud->size() );
      }
    }

    m_PositionArray->resize( cloud->size() );
    m_ColorArray->resize( cloud->size() );

    int j = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator i = cloud->begin(); i != cloud->end(); ++i, ++j) {
      const pcl::PointXYZRGB& p = *i;
      m_PositionArray->at(j) = vl::vec3(p.x, p.y, p.z);
      m_ColorArray->at(j) = vl::vec4(p.r / 255.0f, p.g / 255.0f, p.b / 255.0f, 1);
    }

    m_PositionArray->updateBufferObject();
    m_ColorArray->updateBufferObject();
  }

protected:
  const niftk::PCLData* m_NiftkPCL;
};

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA

class VLMapperCUDAImage: public VLMapper {
public:
  VLMapperCUDAImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_CudaResource = NULL;
  }

  virtual niftk::LightweightCUDAImage getLWCI() {
    niftk::LightweightCUDAImage lwci;
    niftk::CUDAImage* cuda_image = dynamic_cast<niftk::CUDAImage*>( m_DataNode->GetData() );
    if ( cuda_image ) {
      lwci = cuda_image->GetLightweightCUDAImage();
    } else {
      niftk::CUDAImageProperty* cuda_img_prop = dynamic_cast<niftk::CUDAImageProperty*>( m_DataNode->GetData()->GetProperty("CUDAImageProperty").GetPointer() );
      if  (cuda_img_prop ) {
        lwci = cuda_img_prop->Get();
      }
    }
    VIVID_CHECK(lwci.GetId() != 0);
    return lwci;
  }

  virtual bool init() {
    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    // initRenderModeProps( node ); /* does not apply */
    initFogProps( node );
    initClipProps( node );

    niftk::LightweightCUDAImage lwci = getLWCI();

    ref<vl::Geometry> vlquad = CreateGeometryFor2DImage( lwci.GetWidth(), lwci.GetHeight() );

    m_Actor = initActor( vlquad.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    Effect* fx = m_Actor->effect();

    fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
    fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
    vlquad->setColorArray( vl::white );

    m_Texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
    VIVID_CHECK( m_Texture );
    VIVID_CHECK( m_Texture->handle() );
    cudaError_t err = cudaSuccess;
    err = cudaGraphicsGLRegisterImage( &m_CudaResource, m_Texture->handle(), m_Texture->dimension(), cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess ) {
      throw std::runtime_error("cudaGraphicsGLRegisterImage() failed.");
      return false;
    }
    return true;
  }

  virtual void update() {
    updateCommon();
    if ( isDataNodeVividUpdateEnabled() ) {
      // updateRenderModeProps(); /* does not apply here */
      updateFogProps( m_Actor->effect(), m_DataNode );
      updateClipProps( m_Actor->effect(), m_DataNode );
    }

    // Get the niftk::LightweightCUDAImage

    niftk::LightweightCUDAImage lwci = getLWCI();

    cudaError_t err = cudaSuccess;

    // Update texture size and cuda graphics resource

    if ( m_Texture->width() != lwci.GetWidth() || m_Texture->height() != lwci.GetHeight() ) {
      VIVID_CHECK(m_CudaResource);
      cudaGraphicsUnregisterResource(m_CudaResource);
      m_CudaResource = NULL;
      m_Texture->createTexture2D( lwci.GetWidth(), lwci.GetHeight(), TF_RGBA, false );
      err = cudaGraphicsGLRegisterImage( &m_CudaResource, m_Texture->handle(), m_Texture->dimension(), cudaGraphicsRegisterFlagsNone );
      if ( err != cudaSuccess ) {
        throw std::runtime_error("cudaGraphicsGLRegisterImage() failed.");
      }
    }

    niftk::CUDAManager* cm = niftk::CUDAManager::GetInstance();
    cudaStream_t mystream = cm->GetStream(VL_CUDA_STREAM_NAME);
    niftk::ReadAccessor ra = cm->RequestReadAccess(lwci);

    // make sure producer of the cuda-image finished.
    err = cudaStreamWaitEvent(mystream, ra.m_ReadyEvent, 0);
    VIVID_CHECK(err == cudaSuccess);

    err = cudaGraphicsMapResources(1, &m_CudaResource, mystream);
    VIVID_CHECK(err == cudaSuccess);

    cudaArray_t arr = 0;
    err = cudaGraphicsSubResourceGetMappedArray(&arr, m_CudaResource, 0, 0);
    VIVID_CHECK(err == cudaSuccess);

    err = cudaMemcpy2DToArrayAsync(arr, 0, 0, ra.m_DevicePointer, ra.m_BytePitch, lwci.GetWidth() * 4, lwci.GetHeight(), cudaMemcpyDeviceToDevice, mystream);

    err = cudaGraphicsUnmapResources(1, &m_CudaResource, mystream);
    VIVID_CHECK(err == cudaSuccess);

    cm->Autorelease(ra, mystream);
  }

  virtual void remove() {
    if ( m_CudaResource ) {
      cudaError_t err = cudaGraphicsUnregisterResource( m_CudaResource );
      if (err != cudaSuccess) {
        MITK_WARN << "cudaGraphicsUnregisterResource() failed.";
      }
      m_CudaResource = NULL;
    }

    VLMapper::remove();
  }

  Texture* texture() { return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture(); }
  const Texture* texture() const { return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture(); }

protected:
    cudaGraphicsResource_t m_CudaResource;
    ref<Texture> m_Texture;
};

#endif

//-----------------------------------------------------------------------------

vl::ref<VLMapper> VLMapper::create( const mitk::DataNode* node, VLSceneView* sv )
{
  // Map DataNode type to VLMapper type
  vl::ref<VLMapper> vl_node;

  const VLGlobalSettingsDataNode* vl_global = dynamic_cast<const VLGlobalSettingsDataNode*>(node);
  mitk::Surface*            mitk_surf = dynamic_cast<mitk::Surface*>(node->GetData());
  mitk::Image*              mitk_image = dynamic_cast<mitk::Image*>( node->GetData() );
  mitk::CoordinateAxesData* mitk_axes = dynamic_cast<mitk::CoordinateAxesData*>( node->GetData() );
  mitk::PointSet*           mitk_pset = dynamic_cast<mitk::PointSet*>( node->GetData() );
#ifdef _USE_PCL
  niftk::PCLData*           mitk_pcld = dynamic_cast<niftk::PCLData*>( node->GetData() );
#endif
#ifdef _USE_CUDA
  mitk::BaseData*           cuda_img = dynamic_cast<niftk::CUDAImage*>( node->GetData() );
  niftk::CUDAImageProperty* cuda_img_prop = dynamic_cast<niftk::CUDAImageProperty*>( node->GetData()->GetProperty("CUDAImageProperty").GetPointer() );
#endif

  if ( vl_global ) {
    vl_node = new VLMapperVLGlobalSettings( node, sv );
  }
  else
  if ( mitk_surf ) {
    vl_node = new VLMapperSurface( node, sv );
  }
  else
#ifdef _USE_CUDA
  if ( cuda_img || cuda_img_prop ) {
    vl_node = new VLMapperCUDAImage( node, sv );
  }
  else
#endif
  if ( mitk_image ) {
    unsigned int depth = mitk_image->GetDimensions()[2];
    if ( depth > 1 ) {
      vl_node = new VLMapper3DImage( node, sv );
    } else {
      vl_node = new VLMapper2DImage( node, sv );
    }
  }
  else
  if ( mitk_axes ) {
    vl_node = new VLMapperCoordinateAxes( node, sv );
  }
  else
  if ( mitk_pset ) {
    vl_node = new VLMapperPointSet( node, sv );
  }
#ifdef _USE_PCL
  else
  if ( mitk_pcld ) {
    vl_node = new VLMapperPCL( node, sv );
  }
#endif
  return vl_node;
}

//-----------------------------------------------------------------------------
// VLSceneView
//-----------------------------------------------------------------------------

VLSceneView::VLSceneView() :
  m_ScheduleTrackballAdjustView( true ),
  m_ScheduleInitScene ( true ),
  m_RenderingInProgressGuard ( true),
  m_QGLWidget( NULL ),
  m_OclService( 0 )
{
#ifdef _USE_CUDA
  m_CudaTest = new CudaTest;
#endif
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
  ScopedOpenGLContext glctx(m_QGLWidget);

  removeDataStorageListeners();

  clearScene();
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

void VLSceneView::setDataStorage(const mitk::DataStorage::Pointer& ds)
{
  ScopedOpenGLContext glctx(m_QGLWidget);

  removeDataStorageListeners();

  m_DataStorage = ds;
  addDataStorageListeners();

  clearScene();

  // Initialize VL Global Settings if not present
  if ( ! ds->GetNamedNode( VLGlobalSettingsDataNode::VLGlobalSettingsName() ) ) {
    mitk::DataNode::Pointer node = VLGlobalSettingsDataNode::New();
    ds->Add( node.GetPointer() );
  }

  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::setOclResourceService(OclResourceService* oclserv)
{
 // no idea if this is really a necessary restriction.
 // if it is then maybe the ocl-service should be a constructor parameter.
 if (m_OclService != 0)
   throw std::runtime_error("Can set OpenCL service only once");

 m_OclService = oclserv;
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

  ScopedOpenGLContext glctx(m_QGLWidget);

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
  ScopedOpenGLContext glctx(m_QGLWidget);

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
  ScopedOpenGLContext glctx(m_QGLWidget);

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
  ScopedOpenGLContext glctx(m_QGLWidget);

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
  m_VividRendering->setBackgroundColor( fvec4(r, g, b, 1) );
  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::initEvent()
{
  VIVID_CHECK( contextIsCurrent() );

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

  // Interface VL with Qt's resource system to load GLSL shaders.
  vl::defFileSystem()->directories().clear();
  vl::defFileSystem()->directories().push_back( new vl::QtDirectory( ":/VL/" ) );

  // Create our VividRendering!
  m_VividRendering = new vl::VividRendering;
  m_VividRendering->setRenderingMode( vl::Vivid::DepthPeeling ); /* (default) */
  m_VividRendering->setCullingEnabled( false );
  // This creates some flickering on the skin for some reason
  m_VividRendering->setNearFarClippingPlanesOptimized( false );

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
   VIVID_CHECK( contextIsCurrent() );

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
  VIVID_CHECK( contextIsCurrent() );

  renderScene();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateScene() {
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );
  VIVID_CHECK( contextIsCurrent() );

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
    m_NodesToUpdate.size() > 0 ? openglContext()->update() : 0;
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

  VIVID_CHECK( contextIsCurrent() );

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

  ScopedOpenGLContext glctx(m_QGLWidget);

  // Shut down VLMappers
  for ( DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.begin(); it != m_DataNodeVLMapperMap.end(); ++it ) {
    it->second->remove();
  }

  m_VividRendering->stencilActors().clear();
  m_SceneManager->tree()->actors()->clear();
  m_SceneManager->tree()->eraseAllChildren();

  m_CameraNode = 0;
  m_BackgroundNode = 0;
  m_DataNodeVLMapperMap.clear();
  m_NodesToUpdate.clear();
  m_NodesToAdd.clear();
  m_NodesToRemove.clear();

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

bool VLSceneView::setBackgroundNode(const mitk::DataNode* node)
{
  m_BackgroundNode = node;

  // update camera viewport based on background node intrinsics present or not
  updateCameraParameters();

  if ( ! node ) {
    m_VividRendering->setBackgroundImageEnabled( false );
    return true;
  }

  // Wire up background texture
  VLMapper2DImage* img2d_mapper = dynamic_cast<VLMapper2DImage*>( getVLMapper( node ) );
  VLMapperCUDAImage* imgCu_mapper = dynamic_cast<VLMapperCUDAImage*>( getVLMapper( node ) );
  vl::Texture* tex = NULL;
  if ( img2d_mapper ) {
    tex = img2d_mapper->texture();
  } else
  if ( imgCu_mapper ) {
    tex = imgCu_mapper->texture();
  } else {
    return false;
  }
  m_VividRendering->backgroundTexSampler()->setTexture( tex );

  // Hide 3D plane with 2D image on it
  setBoolProp( const_cast<mitk::DataNode*>(node), "visible", false );

  // Enable background rendering
  m_VividRendering->setBackgroundImageEnabled( true );

  openglContext()->update();

  return true;
}

//-----------------------------------------------------------------------------

bool VLSceneView::setCameraTrackingNode(const mitk::DataNode* node)
{
  VIVID_CHECK( m_Trackball );

  // Whenever we set the camera node to NULL we recenter the scene using the trackball

  m_CameraNode = node;

  if (m_CameraNode.IsNull())
  {
    m_Trackball->setEnabled( true );
    scheduleTrackballAdjustView( true );
  } else {
    dumpNodeInfo( "CameraNode():", node );
    dumpNodeInfo( "node->GetData()", node->GetData() );
    m_Trackball->setEnabled( false );
    scheduleTrackballAdjustView( false );
    // update camera position
    updateCameraParameters();
  }

  openglContext()->update();

  return true;
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
    vl::mat4 mat = GetVLMatrixFromData( m_CameraNode->GetData() );
    VIVID_CHECK( ! mat.isNull() );
    if ( ! mat.isNull() ) {
      m_Camera->setModelingMatrix( mat );
    }
  }
}

//-----------------------------------------------------------------------------
/*
                                    Obsolete
*/

void VLSceneView::updateThresholdVal( int isoVal )
{
  float iso = isoVal / 10000.0f;
  iso = vl::clamp( iso, 0.0f, 1.0f );
  // m_VividRendering->vividVolume()->setIsoValue( iso );
  VIVID_CHECK( 0 );
}

