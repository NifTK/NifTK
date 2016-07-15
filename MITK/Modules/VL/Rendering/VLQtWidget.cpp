/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if 1
  // MS VS
  #if defined(_MSC_VER)
    #define VIVID_TRAP() { if (IsDebuggerPresent()) { __debugbreak(); } else ::vl::abort_vl(); }
  // GNU GCC
  #elif defined(__GNUG__) || defined(__MINGW32__)
    #define VIVID_TRAP() { fflush(stdout); fflush(stderr); asm("int $0x3"); }
  #else
    #define VIVID_TRAP() { ::vl::abort_vl(); }
  #endif
  #define VIVID_CHECK(expr) { if(!(expr)) { ::vl::log_failed_check(#expr,__FILE__,__LINE__); VIVID_TRAP() } }
  #define VIVID_WARN(expr)  { if(!(expr)) { ::vl::log_failed_check(#expr,__FILE__,__LINE__); } }
#else
  #define VIVID_CHECK(expr) { (expr) }
  #define VIVID_WARN(expr) { (expr) }
  #define VIVID_TRAP() {}
#endif

#include <QTextStream>
#include <QFile>
#include <QDir>

#include "VLQtWidget.h"
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
#include <mitkEnumerationProperty.h>
#include <mitkProperties.h>
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <stdexcept>
#include <sstream>
#include <niftkScopedOGLContext.h>
// #include "TrackballManipulator.h"
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

//-----------------------------------------------------------------------------
// CUDA stuff
//-----------------------------------------------------------------------------

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
    std::string m_NodeName;
    mitk::DataStorage::Pointer m_DataStorage;

    VLFramebufferAdaptor* m_FBOAdaptor;

    CUDAInterop() : m_FBOAdaptor(0)
    {
    }

    ~CUDAInterop()
    {
      delete m_FBOAdaptor;
    }
  };

  //-----------------------------------------------------------------------------

  struct TextureDataPOD
  {
    vl::ref<vl::Texture>   m_Texture;       // on the vl side
    unsigned int           m_LastUpdatedID; // on cuda-manager side
    cudaGraphicsResource_t m_CUDARes;       // on cuda(-driver) side

    TextureDataPOD(): m_LastUpdatedID(0) , m_CUDARes(0) {
    }
  };

// #else
//   struct CUDAInterop { };
#endif

using namespace vl;

//-----------------------------------------------------------------------------
// VLUserData
//-----------------------------------------------------------------------------

struct VLUserData: public vl::Object
{
  VLUserData()
    : m_TransformModifiedTime(0)
    , m_ImageModifiedTime(0)
  {
  }

  itk::ModifiedTimeType m_TransformModifiedTime;
  itk::ModifiedTimeType m_ImageModifiedTime;
};

//-----------------------------------------------------------------------------
// mitk::EnumerationProperty wrapper classes
//-----------------------------------------------------------------------------

class VL_Vivid_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Vivid_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Vivid_Mode_Property() {
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

class VL_Render_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Render_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Render_Mode_Property() {
    AddEnum("Polys",           0);
    AddEnum("Outline3D",       1);
    AddEnum("Polys+Outline3D", 2);
    AddEnum("Slice",           3);
    AddEnum("Outline2D",       4);
    AddEnum("Polys+Outline2D", 5);
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
    vec4 color = getColorProp( node, "VL.Material.Color" );
    color.a() = getFloatProp( node, "VL.Material.Opacity" );
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
    if ( node->GetProperty("VL.RenderMode") ) {
      return;
    }

    // gocUniform("vl_Vivid.renderMode")
    mitk::EnumerationProperty::Pointer mode = VL_Render_Mode_Property::New();
    node->SetProperty("VL.RenderMode", mode);
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
    int mode = getEnumProp( node, "VL.RenderMode", 0 );
    vec4 color = getColorProp( node, "VL.Outline.Color", vl::yellow );
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

    printf("Transform: %p\n", tr );
    if ( ! m.isNull() )
    {
      tr->setLocalMatrix(m);
      tr->computeWorldMatrix();
#if 0
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

  void UpdateTransformFromNode(vl::Transform* txf, const mitk::DataNode::ConstPointer& node)
  {
    if (node.IsNotNull())
    {
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

    ref<vl::ArrayFloat2> tex_coord = new vl::ArrayFloat2;
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

    ref<vl::Geometry> vlPoly = new vl::Geometry;

    // Buffer in host memory to store cell info
    unsigned int* m_IndexBuffer = 0;

    // Buffer in host memory to store vertex points
    float* m_PointBuffer = 0;

    // Buffer in host memory to store normals associated with vertices
    float* m_NormalBuffer = 0;

    // Buffer in host memory to store scalar info associated with vertices
    char* m_ScalarBuffer = 0;

    unsigned int numOfvtkPolyPoints = vtkPoly->GetNumberOfPoints();

    // A polydata will always have point data
    int pointArrayNum = vtkPoly->GetPointData()->GetNumberOfArrays();

    if (pointArrayNum == 0 && numOfvtkPolyPoints == 0)
    {
      MITK_ERROR << "No points detected in the vtkPoly data!\n";
      return NULL;
    }

    // We'll have to build the cell data if not present already
    int cellArrayNum  = vtkPoly->GetCellData()->GetNumberOfArrays();
    if ( cellArrayNum == 0 ) {
      vtkPoly->BuildCells();
    }

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
      MITK_ERROR << "More than three vertices / cell detected, can't handle this data type!\n";
      return NULL;
    }

    vtkSmartPointer<vtkPoints> points = vtkPoly->GetPoints();

    if (points == 0)
    {
      MITK_ERROR << "Corrupt vtkPoly, returning! \n";
      return NULL;
    }

    // Deal with normals
    vtkSmartPointer<vtkDataArray> normals = vtkPoly->GetPointData()->GetNormals();

    if (normals == 0)
    {
      MITK_INFO << "Generating normals for the vtkPoly data (mitk::OclSurface)";

      vtkSmartPointer<vtkPolyDataNormals> normalGen = vtkSmartPointer<vtkPolyDataNormals>::New();
      normalGen->SetInputData(vtkPoly);
      normalGen->AutoOrientNormalsOn();
      normalGen->Update();

      normals = normalGen->GetOutput()->GetPointData()->GetNormals();

      if (normals == 0)
      {
        MITK_ERROR << "Couldn't generate normals, returning! \n";
        return NULL;
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
    pointBufferSize = numOfPoints * sizeof(float) * 3;

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
      VIVID_CHECK(m_NormalCount == numOfPoints);

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
    MITK_INFO << "Surface data initialized. Num of Points: " <<points->GetNumberOfPoints() << " Num of Cells: " <<verts->GetNumberOfCells() << "\n";

    ref<vl::ArrayFloat3>  vl_verts   = new vl::ArrayFloat3;
    ref<vl::ArrayFloat3>  vlNormals = new vl::ArrayFloat3;
    ref<vl::DrawElementsUInt> vlTriangles = new vl::DrawElementsUInt(vl::PT_TRIANGLES);

    vl_verts->resize(numOfPoints * 3);
    vlNormals->resize(numOfPoints * 3);

    vlPoly->drawCalls().push_back(vlTriangles.get());
    vlTriangles->indexBuffer()->resize(numOfTriangles*3);

    vlPoly->setVertexArray(vl_verts.get());
    vlPoly->setNormalArray(vlNormals.get());

    float* vertBufFlotPtr = reinterpret_cast<float *>(vl_verts->ptr());
    float* normBufFlotPtr = reinterpret_cast<float *>(vlNormals->ptr());

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
    vl_verts->updateBufferObject();
    vlNormals->updateBufferObject();
    vlTriangles->indexBuffer()->updateBufferObject();
    glFinish();

    // Buffer in host memory to store cell info
    if (m_IndexBuffer != 0)
      delete m_IndexBuffer;

    // Buffer in host memory to store vertex points
    if (m_PointBuffer != 0)
      delete m_PointBuffer;

    // Buffer in host memory to store normals associated with vertices
    if (m_NormalBuffer != 0)
      delete m_NormalBuffer;

    // Buffer in host memory to store scalar info associated with vertices
    if (m_ScalarBuffer != 0)
      delete m_ScalarBuffer;

    // MITK_INFO << "Num of VL vertices: " << vlPoly->vertexArray()->size() / 3;

    // Finally convert to adjacency format so we can render silhouettes etc.
    return vl::AdjacencyExtractor::extract( vlPoly.get() );
  }

  //-----------------------------------------------------------------------------

  void dumpNodeInfo( const std::string& prefix, const mitk::DataNode::ConstPointer& node ) {
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

  static const char* VLGlobalSettingsName() { return "VL Global Settings"; }

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

    mitk::EnumerationProperty::Pointer render_mode = VL_Vivid_Mode_Property::New();
    AddProperty( "VL.Global.RenderMode", render_mode );
    render_mode->SetValue( 0 );

    mitk::ColorProperty::Pointer bg_color = mitk::ColorProperty::New();
    AddProperty( "VL.Global.BackgroundColor", bg_color );
    bg_color->SetValue( vl::lightgray.ptr() );

    mitk::FloatProperty::Pointer opacity = mitk::FloatProperty::New();
    AddProperty( "VL.Global.Opacity", opacity );
    opacity->SetValue( 1 );
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
  actor->setEnableMask( vl::VividRenderer::DefaultEnableMask );
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

  virtual void init() { }

  virtual void update() {
    bool enable = getBoolProp( m_DataNode, "VL.Global.Stencil.Enable", false );
    vec4 stencil_bg_color = getColorProp( m_DataNode, "VL.Global.Stencil.BackgroundColor", vl::black );
    float stencil_smooth = getFloatProp( m_DataNode, "VL.Global.Stencil.Smoothness", 10 );
    int render_mode = getEnumProp( m_DataNode, "VL.Global.RenderMode", 0 );
    vec4 bg_color = getColorProp( m_DataNode, "VL.Global.BackgroundColor", vl::black );
    float opacity = getFloatProp( m_DataNode, "VL.Global.Opacity", 1 );

    m_VividRendering->setStencilEnabled( enable );
    m_VividRendering->setStencilBackground( stencil_bg_color );
    m_VividRendering->setStencilSmoothness( stencil_smooth );
    m_VividRendering->setRenderingMode( (VividRendering::ERenderingMode)render_mode );
    m_VividRendering->setBackgroundColor( bg_color );
    m_VividRendering->setAlpha( opacity );
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

  virtual void init() {
    VIVID_CHECK( m_MitkSurf );

    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    initRenderModeProps( node );
    initMaterialProps( node );
    initFogProps( node );
    initClipProps( node );

    ref<vl::Geometry> geom = ConvertVTKPolyData( m_MitkSurf->GetVtkPolyData() );
    if ( ! geom->normalArray() ) {
      geom->computeNormals();
    }

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  }

  virtual void update() {
    updateCommon();
    updateMaterialProps( m_Actor->effect(), m_DataNode );
    updateRenderModeProps( m_Actor->effect(), m_DataNode );
    updateFogProps( m_Actor->effect(), m_DataNode );
    updateClipProps( m_Actor->effect(), m_DataNode );

    // Stencil
    bool is_stencil = getBoolProp( m_DataNode, "VL.IsStencil", false );
    std::vector< ref<Actor> >::iterator it = std::find( m_VividRendering->stencilActors().begin(), m_VividRendering->stencilActors().end(), m_Actor.get() );
    if ( ! is_stencil && it != m_VividRendering->stencilActors().end() ) {
      m_VividRendering->stencilActors().erase( it );
    } else
    if ( is_stencil && it == m_VividRendering->stencilActors().end() ) {
      m_VividRendering->stencilActors().push_back( m_Actor );
    }
  }

protected:
  mitk::Surface::Pointer m_MitkSurf;
};

//-----------------------------------------------------------------------------

class VLMapper2DImage: public VLMapper {
public:
  VLMapper2DImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    VIVID_CHECK( m_MitkImage.IsNotNull() );
  }

  virtual void init() {
    VIVID_CHECK( m_MitkImage.IsNotNull() );

    mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
    // initRenderModeProps( node ); /* does not apply */
    initFogProps( node );
    initClipProps( node );

    mitk::PixelType  mitk_pixel_type = m_MitkImage->GetPixelType();
    vl::EImageType   vl_type         = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
    vl::EImageFormat vl_format       = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
    unsigned int*    dims            = m_MitkImage->GetDimensions();

    ref<vl::Image> vl_img;

    try {
      unsigned int buffer_bytes = dims[0] * dims[1] * dims[2] * mitk_pixel_type.GetSize();
      mitk::ImageReadAccessor image_reader( m_MitkImage, m_MitkImage->GetVolumeData(0) );
      void* buffer_ptr = const_cast<void*>( image_reader.GetData() );
      // std::memcpy( vl_img->pixels(), ptr, byte_count );
      // Use VTK buffer directly instead of allocating one
      vl_img = new vl::Image( buffer_ptr, buffer_bytes );
      vl_img->allocate2D(dims[0], dims[1], 1, vl_format, vl_type);
      VIVID_CHECK( vl_img->requiredMemory() == buffer_bytes );
    }
    catch (...) {
      // FIXME: error handling?
      MITK_ERROR << "Did not get pixel read access to 2D image.";
    }

    ref<vl::Geometry> geom = CreateGeometryFor2DImage(dims[0], dims[1]);

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    // These must be present as part of the default Vivid material
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::VividRendering::UserTexture ) )
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture() )
    VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture")->getUniformI() == vl::VividRendering::UserTexture );
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture();
    texture->createTexture2D( vl_img.get(), vl::TF_UNKNOWN, false, false );
    fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
    fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
    // When texture mapping is enabled the texture is modulated by the vertex color, including the alpha
    geom->setColorArray( vl::white );
  }

  virtual void update() {
    VIVID_CHECK( m_MitkImage.IsNotNull() );

    updateCommon();
    // updateRenderModeProps(); /* does not apply here */
    updateFogProps( m_Actor->effect(), m_DataNode );
    updateClipProps( m_Actor->effect(), m_DataNode );

    if ( m_MitkImage->GetVtkImageData()->GetMTime() <= GetUserData( m_Actor.get() )->m_ImageModifiedTime ) {
      return;
    }

    ref<vl::Texture> tex = m_Actor->effect()->shader()->gocTextureSampler( vl::VividRendering::UserTexture )->texture();
    if ( tex )
    {
      mitk::PixelType  mitk_pixel_type = m_MitkImage->GetPixelType();
      vl::EImageType   vl_type         = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
      vl::EImageFormat vl_format       = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
      unsigned int*    dims            = m_MitkImage->GetDimensions();

      try
      {
        unsigned int buffer_bytes = dims[0] * dims[1] * dims[2] * mitk_pixel_type.GetSize();
        mitk::ImageReadAccessor image_reader( m_MitkImage, m_MitkImage->GetVolumeData(0) );
        void* buffer_ptr = const_cast<void*>( image_reader.GetData() );
        // Use VTK buffer directly, no VL imag allocation needed
        ref<vl::Image> vl_img = new vl::Image( buffer_ptr, buffer_bytes );
        vl_img->allocate2D(dims[0], dims[1], 1, vl_format, vl_type);
        VIVID_CHECK( vl_img->requiredMemory() == buffer_bytes );
        tex->setMipLevel(0, vl_img.get(), false);
      }
      catch (...)
      {
        // FIXME: error handling?
        MITK_ERROR << "Did not get pixel read access to 2D image.";
      }

      GetUserData( m_Actor.get() )->m_ImageModifiedTime = m_MitkImage->GetVtkImageData()->GetMTime();
    }
  }

protected:
  mitk::Image::Pointer m_MitkImage;
};

//-----------------------------------------------------------------------------

class VLMapper3DImage: public VLMapper {
public:
  VLMapper3DImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    m_VividVolume = new vl::VividVolume( m_VividRendering );
    VIVID_CHECK( m_MitkImage.IsNotNull() );
  }

  virtual void init() {
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
  mitk::Image::Pointer m_MitkImage;
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

  virtual void init() {
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
  mitk::CoordinateAxesData::Pointer m_MitkAxes;
  ref<vl::ArrayFloat3> m_Vertices;
};

//-----------------------------------------------------------------------------

class VLMapperPointSet: public VLMapper {
public:
  VLMapperPointSet( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    m_MitkPointSet = dynamic_cast<mitk::PointSet*>( node->GetData() );
    m_3DSphereMode = true;
    m_PointFX = vl::VividRendering::makeVividEffect();
    VIVID_CHECK( m_MitkPointSet );
  }

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

  virtual void init() { initPointSetProps(); }

  void init3D() {
    VIVID_CHECK( m_MitkPointSet );
    VIVID_CHECK( m_3DSphereMode );

    // Remove 2D data and init 3D data.
    remove();
    m_SphereActors = new vl::ActorTree;
    m_VividRendering->sceneManager()->tree()->addChild( m_SphereActors.get() );

    m_3DSphereGeom = vl::makeIcosphere( vec3(0,0,0), 1, 2, true );
    int j = 0;
    for (mitk::PointSet::PointsConstIterator i = m_MitkPointSet->Begin(); i != m_MitkPointSet->End(); ++i, ++j)
    {
      mitk::PointSet::PointType p = i->Value();
      vl::vec3 pos( p[0], p[1], p[2] );
      ref<Actor> actor = initActor( m_3DSphereGeom.get(), m_PointFX.get() );
      actor->transform()->setLocalAndWorldMatrix( vl::mat4::getTranslation( pos ) );
      m_SphereActors->addActor( actor.get() );
    }
  }

  void init2D() {
    VIVID_CHECK( m_MitkPointSet );
    VIVID_CHECK( ! m_3DSphereMode );

    // Remove 3D data and init 2D data.
    remove();

    ref<vl::ArrayFloat3> verts = new vl::ArrayFloat3;
    verts->resize(m_MitkPointSet->GetSize());
    int j = 0;
    for (mitk::PointSet::PointsConstIterator i = m_MitkPointSet->Begin(); i != m_MitkPointSet->End(); ++i, ++j)
    {
      mitk::PointSet::PointType p = i->Value();
      verts->at(j).x() = p[0];
      verts->at(j).y() = p[1];
      verts->at(j).z() = p[2];
    }

    m_2DGeometry = new vl::Geometry;
    ref<vl::DrawArrays> draw_arrays = new vl::DrawArrays( vl::PT_POINTS, 0, verts->size() );
    m_2DGeometry->drawCalls().push_back(draw_arrays.get());
    m_2DGeometry->setVertexArray( verts.get() );
    m_2DGeometry->setColorArray( vl::white );

    m_Actor = initActor( m_2DGeometry.get(), m_PointFX.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<vl::Effect> fx = m_Actor->effect();
    ref<vl::Image> img = new Image("/vivid/images/sphere.png");
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture();
    texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );
  }

  virtual void update() {
    // updateCommon();
    // updateRenderModeProps(); /* does not apply here */
    updateFogProps( m_PointFX.get(), m_DataNode );
    updateClipProps( m_PointFX.get(), m_DataNode );

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

    if ( m_3DSphereMode ) {
      if ( ! m_SphereActors ) {
        init3D();
      }

      // 3D mode settings
      m_PointFX->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( 1 );
      m_PointFX->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( 0 );
      m_PointFX->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( 0 );

      // Set color/opacity
      m_PointFX->shader()->getMaterial()->setDiffuse( color );
      for( int i = 0; i < m_SphereActors->actors()->size(); ++i ) {
        // Set visible
        Actor* act = m_SphereActors->actors()->at( i );
        act->setEnabled( visible );
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
      VIVID_CHECK( m_PointFX->shader()->getPointSize() );

      // 2d mode settings
      m_PointFX->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );
      m_PointFX->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( 1 );
      m_PointFX->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( 1 );

      // set point size
      m_PointFX->shader()->getPointSize()->set( pointsize );
      // set color
      m_2DGeometry->setColorArray( color );
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
  mitk::PointSet::Pointer m_MitkPointSet;
  bool m_3DSphereMode;
  ref<vl::ActorTree> m_SphereActors;
  ref<Geometry> m_3DSphereGeom;
  ref<Effect> m_PointFX;
  ref<vl::Geometry> m_2DGeometry;
};

//-----------------------------------------------------------------------------

#ifdef _USE_PCL
/*
       WARNING:
never compiled nor tested

     _.--""--._
    /  _    _  \
 _  ( (_\  /_) )  _
{ \._\   /\   /_./ }
/_"=-.}______{.-="_\
 _  _.=("""")=._  _
(_'"_.-"`~~`"-._"'_)
 {_"            "_}

*/
class VLMapperPCL: public VLMapper {
public:
  VLMapperPCL( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node , sv ) {
    m_NiftkPCL = dynamic_cast<niftk::PCLData*>( node->GetData() );
    VIVID_CHECK( m_NiftkPCL );
  }

  virtual void init() {
    VIVID_CHECK( m_NiftkPCL );
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud = m_NiftkPCL->GetCloud();

    ref<vl::ArrayFloat3> vl_verts = new vl::ArrayFloat3;
    ref<vl::ArrayFloat4> vl_colors = new vl::ArrayFloat4;
    vl_verts->resize(cloud->size());
    vl_colors->resize(cloud->size());
    // We could interleave the color and vert array but do we trust the VTK layout?
    int j = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator i = cloud->begin(); i != cloud->end(); ++i, ++j) {
      const pcl::PointXYZRGB& p = *i;

      vl_verts->at(j).x() = p.x;
      vl_verts->at(j).y() = p.y;
      vl_verts->at(j).z() = p.z;

      vl_colors->at(j).r() = (float)p.r / 255.0f;
      vl_colors->at(j).g() = (float)p.g / 255.0f;
      vl_colors->at(j).b() = (float)p.b / 255.0f;
      vl_colors->at(j).a() = 1;
    }

    ref<vl::Geometry> geom = new vl::Geometry;
    ref<vl::DrawArrays> draw_arrays = new vl::DrawArrays( vl::PT_POINTS, 0, vl_verts->size() );
    geom->drawCalls().push_back( draw_arrays.get() );
    geom->setVertexArray( vl_verts.get() );
    geom->setColorArray( vl_colors.get() );

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  }

  virtual void update() {
    updateCommon();
    // Update point size
    float pointsize = 1;
    m_DataNode->GetFloatProperty( "pointsize", pointsize );
    Shader* shader = m_Actor->effect()->shader();
    // This is part of the standard vivid shader so it must be present.
    VIVID_CHECK( shader->getPointSize() );
    shader->getPointSize()->set( pointsize );
    if ( pointsize > 1 ) {
      shader->enable( vl::EN_POINT_SMOOTH );
    } else {
      shader->disable( vl::EN_POINT_SMOOTH );
    }
  }

protected:
  niftk::PCLData::Pointer m_NiftkPCL;
};

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA
/*
       WARNING:
never compiled nor tested

     _.--""--._
    /  _    _  \
 _  ( (_\  /_) )  _
{ \._\   /\   /_./ }
/_"=-.}______{.-="_\
 _  _.=("""")=._  _
(_'"_.-"`~~`"-._"'_)
 {_"            "_}


This is just stub code, a raw attempt at reorganizing the legacy experimental CUDA code into the new VLMapper logic

*/
class VLMapperCUDAImage: public VLMapper {
public:
  VLMapperCUDAImage( const mitk::DataNode* node, VLSceneView* sv )
    : VLMapper( node, sv ) {
    niftk::CUDAImage* cuda_image = dynamic_cast<niftk::CUDAImage*>( node->GetData() );
    if ( cuda_image ) {
      m_NiftkLightweightCUDAImage = cuda_image->GetLightweightCUDAImage();
    } else {
      niftk::CUDAImageProperty* cuda_image_prop = dynamic_cast<niftk::CUDAImageProperty*>(m_DataNode->GetProperty("CUDAImageProperty").GetPointer());
      if  (cuda_image_prop ) {
        m_NiftkLightweightCUDAImage = cuda_image_prop->Get();
      }
    }
    VIVID_CHECK(m_NiftkLightweightCUDAImage.GetId() != 0);
  }

  virtual void init() {

    niftk::LightweightCUDAImage lwci;
    const niftk::CUDAImage* cudaImg = dynamic_cast<const niftk::CUDAImage*>(m_NiftkCUDAImage);
    if (cudaImg != 0)
    {
      lwci = cudaImg->GetLightweightCUDAImage();
    }
    else
    {
      niftk::CUDAImageProperty::Pointer prop = dynamic_cast<niftk::CUDAImageProperty*>(m_DataNode->GetProperty("CUDAImageProperty").GetPointer());
      if (prop.IsNotNull())
      {
        lwci = prop->Get();
      }
    }
    VIVID_CHECK(lwci.GetId() != 0);

    ref<vl::Geometry> vlquad = CreateGeometryFor2DImage(m_NiftkLightweightCUDAImage.GetWidth(), m_NiftkLightweightCUDAImage.GetHeight());

    m_Actor = initActor( vlquad.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    fx->shader()->disable(vl::EN_LIGHTING);
    fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
    fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());
  }

  virtual void update() {
    updateCommon();
    VIVID_CHECK(m_NiftkLightweightCUDAImage.GetId() != 0);

    // BEWARE:
    // All the logic below is completely outdated especially with regard to accessing the user texture. See VLMapper2DImage for more info.
    // PS. All the horrific code formatting is from the original code...
    // - Michele

    // whatever we had cached from a previous frame.
    TextureDataPOD          texpod    = m_TextureDataPOD;

    // only need to update the vl texture, if content in our cuda buffer has changed.
    // and the cuda buffer can change only when we have a different id.
    if (texpod.m_LastUpdatedID != m_NiftkLightweightCUDAImage.GetId())
    {
      cudaError_t   err = cudaSuccess;
      bool          neednewvltexture = texpod.m_Texture.get() == 0;

      // check if vl-texture size needs to change
      if (texpod.m_Texture.get() != 0)
      {
        neednewvltexture |= m_NiftkLightweightCUDAImage.GetWidth()  != texpod.m_Texture->width();
        neednewvltexture |= m_NiftkLightweightCUDAImage.GetHeight() != texpod.m_Texture->height();
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

        texpod.m_Texture = new vl::Texture(m_NiftkLightweightCUDAImage.GetWidth(), m_NiftkLightweightCUDAImage.GetHeight(), vl::TF_RGBA8, false);
        actor->effect()->shader()->gocTextureSampler(0)->setTexture(texpod.m_Texture.get());
        actor->effect()->shader()->gocTextureSampler(0)->setTexParameter(m_DefaultTextureParams.get());

        err = cudaGraphicsGLRegisterImage(&texpod.m_CUDARes, texpod.m_Texture->handle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess)
        {
          texpod.m_CUDARes = 0;
          MITK_WARN << "Registering VL texture into CUDA failed. Will not update (properly).";
        }
      }

      if (texpod.m_CUDARes)
      {
        VIVID_CHECK(actor->effect()->shader()->getTextureSampler(0)->texture() == texpod.m_Texture);

        niftk::CUDAManager*  cudamng   = niftk::CUDAManager::GetInstance();
        cudaStream_t         mystream  = cudamng->GetStream("VLSceneView vl-texture update");
        niftk::ReadAccessor  inputRA   = cudamng->RequestReadAccess(m_NiftkLightweightCUDAImage);

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
            err = cudaMemcpy2DToArrayAsync(arr, 0, 0, inputRA.m_DevicePointer, inputRA.m_BytePitch, m_NiftkLightweightCUDAImage.GetWidth() * 4, m_NiftkLightweightCUDAImage.GetHeight(), cudaMemcpyDeviceToDevice, mystream);
            if (err == cudaSuccess)
            {
              texpod.m_LastUpdatedID = m_NiftkLightweightCUDAImage.GetId();
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
      m_TextureDataPOD = texpod;

      // helps with debugging
      actor->effect()->shader()->disable(vl::EN_CULL_FACE);
    }
  }

  virtual void remove() {
    if ( m_TextureDataPOD.m_CUDARes ) {
      cudaError_t err = cudaGraphicsUnregisterResource( m_TextureDataPOD.m_CUDARes );
      if (err != cudaSuccess)
      {
        MITK_WARN << "Failed to unregister VL texture from CUDA";
      }
      m_TextureDataPOD.m_CUDARes = 0;
    }

    VLMapper::remove();
  }

protected:
  niftk::LightweightCUDAImage m_NiftkLightweightCUDAImage;
  TextureDataPOD m_TextureDataPOD; // m_NodeToTextureMap
};

#endif

//-----------------------------------------------------------------------------

vl::ref<VLMapper> VLMapper::create( const mitk::DataNode* node, VLSceneView* sv ) {

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
#endif

  if ( vl_global ) {
    vl_node = new VLMapperVLGlobalSettings( node, sv );
  }
  else
  if ( mitk_surf ) {
    vl_node = new VLMapperSurface( node, sv );
  }
  else
  if ( mitk_image ) {
    unsigned int depth = mitk_image->GetDimensions()[2];
    // In VTK a NxMx1 image is 2D (in VL a 2D image is NxMx0)
    if ( depth <= 1 ) {
      vl_node = new VLMapper2DImage( node, sv );
    } else {
      vl_node = new VLMapper3DImage( node, sv );
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
#ifdef _USE_CUDA
  else
  if ( mitk_pcld ) {
    vl_node = new VLMapperCUDAImage( node, sv );
  }
#endif
  return vl_node;
}

//-----------------------------------------------------------------------------
// VLSceneView
//-----------------------------------------------------------------------------

VLSceneView::VLSceneView() :
  // Qt5Widget(parent, shareWidget, f)
  m_BackgroundWidth( 0 )
  , m_BackgroundHeight( 0 )
  , m_ScheduleTrackballAdjustView( true )
  , m_ScheduleInitScene ( true )
  , m_OclService( 0 )
#ifdef _USE_CUDA
  , m_CUDAInteropPimpl(0)
#endif
{
}

//-----------------------------------------------------------------------------

 void VLSceneView::destroyEvent()
{
  openglContext()->makeCurrent();

  removeDataStorageListeners();

#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif
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
  openglContext()->makeCurrent();

  removeDataStorageListeners();

#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif

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
  if ( ! node /* || ! node->GetData() */ ) {
    return;
  }

  m_NodesToRemove.insert( mitk::DataNode::ConstPointer ( node ) ); // remove it
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

  openglContext()->makeCurrent();

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

VLMapper* VLSceneView::addDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  // Add only once and only if valid
  if ( ! node || ! node->GetData() || getVLMapper( node ) != NULL ) {
    return NULL;
  }

  #if 1
    dumpNodeInfo( "addDataNode()", node );
    dumpNodeInfo( "addDataNode()->GetData()", node->GetData() );
  #endif

  ref<VLMapper> vl_node = VLMapper::create( node.GetPointer(), this );
  if ( vl_node ) {
    m_DataNodeVLMapperMap[ node ] = vl_node;
    vl_node->init();
    vl_node->update();
  }

  return vl_node.get();
}

//-----------------------------------------------------------------------------

void VLSceneView::removeDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

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

void VLSceneView::updateDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  if ( node.IsNull() || node->GetData() == 0 ) {
    return;
  }

  #if 0
    dumpNodeInfo( "updateDataNode()", node );
  #endif

  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  if ( it != m_DataNodeVLMapperMap.end() ) {
    // this might recreate new Actors
    it->second->update();
    return;
  }

  // Update camera
  if (node == m_CameraNode) {
    updateCameraParameters();
  }
}

//-----------------------------------------------------------------------------

VLMapper* VLSceneView::getVLMapper( const mitk::DataNode::ConstPointer& node )
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

  // vl::OpenGLContext::initGLContext();

  // Interface VL with Qt's resource system to load GLSL shaders.
  vl::defFileSystem()->directories().clear();
  vl::defFileSystem()->directories().push_back( new vl::QtDirectory( ":/VL/" ) );


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

#ifdef _MSC_VER
  // NvAPI_OGL_ExpertModeSet(NVAPI_OGLEXPERT_DETAIL_ALL, NVAPI_OGLEXPERT_DETAIL_BASIC_INFO, NVAPI_OGLEXPERT_OUTPUT_TO_ALL, 0);
#endif

  // Create our VividRendering!
  m_VividRendering = new vl::VividRendering;
  m_VividRendering->setRenderingMode( vl::VividRendering::DepthPeeling ); /* (default) */
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

  // This is only used by the CUDA stuff
  createAndUpdateFBOSizes( openglContext()->width(), openglContext()->height() );

#if 0
  // Point cloud data test
  mitk::DataNode::Pointer n = mitk::DataNode::New();
  mitk::PCLData::Pointer  p = niftk::PCLData::New();
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

void VLSceneView::resizeEvent( int w, int h )
{
   VIVID_CHECK( contextIsCurrent() );

  // dont do anything if window is zero size.
  // it's an opengl error to have a viewport like that!
  if ( w <= 0 || h <= 0 ) {
    return;
  }

  m_VividRendering->camera()->viewport()->set( 0, 0, w, h );
  m_VividRendering->camera()->setProjectionPerspective();

  createAndUpdateFBOSizes( w, h );

  // MIC FIXME: update calibrated camera setup
  updateViewportAndCameraAfterResize();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateEvent()
{
  VIVID_CHECK( contextIsCurrent() );

  renderScene();
}

//-----------------------------------------------------------------------------

void VLSceneView::createAndUpdateFBOSizes( int width, int height )
{
  openglContext()->makeCurrent();

#ifdef _USE_CUDA
  // sanitise dimensions. depending on how windows are resized we can get zero here.
  // but that breaks on the ogl side.
  width  = std::max(1, width);
  height = std::max(1, height);

  ref<vl::FramebufferObject> opaqueFBO = vl::OpenGLContext::createFramebufferObject(width, height);
  opaqueFBO->setObjectName("opaqueFBO");
  opaqueFBO->addDepthAttachment(new vl::FBODepthBufferAttachment(vl::DBF_DEPTH_COMPONENT24));
  opaqueFBO->addColorAttachment(vl::AP_COLOR_ATTACHMENT0, new vl::FBOColorBufferAttachment(vl::CBF_RGBA));   // this is a renderbuffer
  opaqueFBO->setDrawBuffer(vl::RDB_COLOR_ATTACHMENT0);

  if (m_CUDAInteropPimpl)
  {
    delete m_CUDAInteropPimpl->m_FBOAdaptor;
    m_CUDAInteropPimpl->m_FBOAdaptor = new VLFramebufferAdaptor(opaqueFBO.get());
  }
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::updateViewportAndCameraAfterResize()
{
  // some sane defaults
  // m_Camera->viewport()->set( 0, 0, QWidget::width(), QWidget::height() );
  // m_BackgroundCamera->viewport()->set(0, 0, QWidget::width(), QWidget::height());

  if ( m_BackgroundNode.IsNotNull() )
  {
    //NodeActorMapType::iterator ni = m_NodeActorMap.find(m_BackgroundNode);
    //if (ni == m_NodeActorMap.end())
    //{
    //  // actor not ready yet, try again later.
    //  // this is getting messy... but stuffing our widget here into an editor causes various methods
    //  // to be called at the wrong time.
    //  QMetaObject::invokeMethod(this, "updateViewportAndCameraAfterResize", Qt::QueuedConnection);
    //}
    //else
    //{
      // ref<vl::Actor> backgroundactor = ni->second;

      // this is based on my old araknes video-ar app.
      // FIXME: aspect ratio?
      float   width_scale  = (float) openglContext()->width()  / (float) m_BackgroundWidth;
      float   height_scale = (float) openglContext()->height() / (float) m_BackgroundHeight;
      int     vpw = openglContext()->width();
      int     vph = openglContext()->height();
      if (width_scale < height_scale)
        vph = (int) ((float) m_BackgroundHeight * width_scale);
      else
        vpw = (int) ((float) m_BackgroundWidth * height_scale);

      int   vpx = openglContext()->width()  / 2 - vpw / 2;
      int   vpy = openglContext()->height() / 2 - vph / 2;

      // m_BackgroundCamera->viewport()->set(vpx, vpy, vpw, vph);
      // the main-scene-camera should conform to this viewport too!
      // otherwise geometry would never line up with the background (for overlays, etc).
      m_Camera->viewport()->set(vpx, vpy, vpw, vph);
    //}
  }
  // this default perspective depends on the viewport!
  m_Camera->setProjectionPerspective();

  updateCameraParameters();
}

void VLSceneView::updateScene() {
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );
  VIVID_CHECK( contextIsCurrent() );

  if ( m_ScheduleInitScene ) {
    initSceneFromDataStorage();
    m_ScheduleInitScene = false;
  } else {
    // Execute scheduled removals
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToRemove.begin(); it != m_NodesToRemove.end(); ++it)
    {
      removeDataNode(*it);
    }
    m_NodesToRemove.clear();

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

  if ( m_ScheduleTrackballAdjustView ) {
    m_Trackball->adjustView( m_VividRendering.get(), vl::vec3(0,0,1), vl::vec3(0,1,0), 1.0f );
    m_ScheduleTrackballAdjustView = false;
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::renderScene()
{
  VIVID_CHECK( contextIsCurrent() );

  updateScene();

  // Set frame time for all the rendering
  vl::real now_time = vl::Time::currentTime();
  m_VividRendering->setFrameClock( now_time );

  // Execute rendering
  m_VividRendering->render( openglContext()->framebuffer() );

  // Show rendering
  if ( openglContext()->hasDoubleBuffer() ) {
#ifdef _USE_CUDA
    cudaSwapBuffers();
#endif
    openglContext()->swapBuffers();
  }

  VL_CHECK_OGL();
}

//-----------------------------------------------------------------------------

void VLSceneView::clearScene()
{
  openglContext()->makeCurrent();

  if ( m_SceneManager )
  {
    if ( m_SceneManager->tree() ) {
      m_SceneManager->tree()->actors()->clear();
      m_SceneManager->tree()->eraseAllChildren();
      m_VividRendering->stencilActors().clear();
      // These depend on the global settings
      // m_VividRendering->setBackgroundImageEnabled(false);
      // m_VividRendering->setStencilEnabled(false);
      // m_VividRendering->setStencilBackground(vl::black);
    }
  }

  m_CameraNode = 0;
  m_BackgroundNode = 0;
  m_DataNodeVLMapperMap.clear();
  m_NodesToUpdate.clear();
  m_NodesToAdd.clear();
  m_NodesToRemove.clear();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateThresholdVal( int isoVal )
{
  float iso = isoVal / 10000.0f;
  iso = vl::clamp( iso, 0.0f, 1.0f );
  // m_VividRendering->vividVolume()->setIsoValue( iso );
  VIVID_CHECK( 0 );
}

//-----------------------------------------------------------------------------

bool VLSceneView::setCameraTrackingNode(const mitk::DataNode::ConstPointer& node)
{
  VIVID_CHECK( m_Trackball );

  // Whenever we set the camera node to NULL we recenter the scene using the trackball

  m_CameraNode = node;

  if (m_CameraNode.IsNull())
  {
    m_Trackball->setEnabled( true );
    scheduleTrackballAdjustView( true );
  } else {
    dumpNodeInfo( "CameraNode()", node );
    m_Trackball->setEnabled( false );
    scheduleTrackballAdjustView( false );
    updateCameraParameters();
  }

  openglContext()->update();

  return true;
}

//-----------------------------------------------------------------------------

void VLSceneView::updateCameraParameters()
{
  // calibration parameters come from the background node.
  // so no background, no camera parameters.
  if (m_BackgroundNode.IsNotNull())
  {
#ifdef BUILD_IGI
    mitk::BaseProperty::Pointer cambp = m_BackgroundNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
    if (cambp.IsNotNull())
    {
      mitk::CameraIntrinsicsProperty::Pointer cam = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cambp.GetPointer());
      if (cam.IsNotNull())
      {
        mitk::CameraIntrinsics::Pointer nodeIntrinsic = cam->GetValue();

        if (nodeIntrinsic.IsNotNull())
        {
          // based on niftkCore/Rendering/vtkOpenGLMatrixDrivenCamera
          float znear = 1;
          float zfar  = 10000;
          float pixelaspectratio = 1;   // FIXME: depends on background image

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
    vl::mat4 mat = GetVLMatrixFromData(m_CameraNode->GetData());
    if ( ! mat.isNull() ) {
      // beware: there is also a view-matrix! the inverse of modelling-matrix.
      m_Camera->setModelingMatrix(mat);
    }
  }
}

//-----------------------------------------------------------------------------

// MIC FIXME: remove this
void VLSceneView::prepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  /*
  openglContext()->makeCurrent();

  // nasty
  mitk::Image::Pointer imgp(const_cast<mitk::Image*>(img));
  ref<vl::Actor> actor = Add2DImageActor(imgp);


  // essentially copied from vl::makeGrid()
  ref<vl::Geometry>         vlquad = new vl::Geometry;

  ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls().push_back(polys.get());

  // replace original quad with ours.
  actor->setLod(0, vlquad.get());
  actor->effect()->shader()->disable(vl::EN_LIGHTING);

  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeActorMap[node] = actor;
  */
}

//-----------------------------------------------------------------------------

bool VLSceneView::setBackgroundNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  // clear up after previous background node.
  if (m_BackgroundNode.IsNotNull())
  {
    const mitk::DataNode::ConstPointer    oldbackgroundnode = m_BackgroundNode;
    m_BackgroundNode = 0;
    removeDataNode(oldbackgroundnode);
    // add back as normal node.
    addDataNode(oldbackgroundnode);
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
    removeDataNode(node);

    mitk::Image::Pointer imgdata = dynamic_cast<mitk::Image*>(basedata.GetPointer());
    if (imgdata.IsNotNull())
    {
#ifdef _USE_CUDA
      niftk::CUDAImageProperty::Pointer    cudaimgprop = dynamic_cast<niftk::CUDAImageProperty*>(imgdata->GetProperty("CUDAImageProperty").GetPointer());
      if (cudaimgprop.IsNotNull())
      {
        niftk::LightweightCUDAImage    lwci = cudaimgprop->Get();

        // does the size of cuda-image have to match the mitk-image where it's attached to?
        // i think it does: it is supposed to be the same data living in cuda.
        VIVID_CHECK(lwci.GetWidth()  == imgdata->GetDimension(0));
        VIVID_CHECK(lwci.GetHeight() == imgdata->GetDimension(1));

        prepareBackgroundActor(&lwci, imgdata->GetGeometry(), node);
        result = true;
      }
      else
#endif
      {
        prepareBackgroundActor(imgdata.GetPointer(), imgdata->GetGeometry(), node);
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
        prepareBackgroundActor(&lwci, cudaimgdata->GetGeometry(), node);
        result = true;

        m_BackgroundWidth  = lwci.GetWidth();
        m_BackgroundHeight = lwci.GetHeight();
      }
      // no else here
#endif
    }

    // updateDataNode() depends on m_BackgroundNode.
    m_BackgroundNode = node;
    updateDataNode(node);
  }

  updateViewportAndCameraAfterResize();

  // now that the camera may have changed, fit-view-to-scene again.
  //if (m_CameraNode.IsNull())
  //{
  //  m_Trackball->setEnabled( true );
  //  m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);
  //}

  return result;
}

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA
void VLSceneView::cudaSwapBuffers()
{
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
      VIVID_CHECK(false);
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
}

//-----------------------------------------------------------------------------

void VLSceneView::FreeCUDAInteropTextures()
{
  openglContext()->makeCurrent();

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

}

//-----------------------------------------------------------------------------

void VLSceneView::EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage, const std::string& nodename)
{
  openglContext()->makeCurrent();

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
}

//-----------------------------------------------------------------------------

void VLSceneView::prepareBackgroundActor(const niftk::LightweightCUDAImage* lwci, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  openglContext()->makeCurrent();

  VIVID_CHECK(lwci != 0);

  vl::mat4  mat;
  mat = mat.setIdentity();
  ref<vl::Transform> tr     = new vl::Transform();
  tr->setLocalMatrix(mat);


  // essentially copied from vl::makeGrid()
  ref<vl::Geometry>         vlquad = new vl::Geometry;

  ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls().push_back(polys.get());


  ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  // updateDataNode() takes care of assigning colour etc.

  ref<vl::Actor> actor = m_VividRendering->sceneManager()->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  actor->setEnableMask( vl::VividRenderer::DefaultEnableMask );


  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeActorMap[node] = actor;
  m_NodeToTextureMap[node] = TextureDataPOD();
}

//-----------------------------------------------------------------------------

#endif