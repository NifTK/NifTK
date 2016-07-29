/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLUtils.h"

#include <vlGraphics/Effect.hpp>
#include <vlGraphics/Actor.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/AdjacencyExtractor.hpp>
#include <vlVivid/VividVolume.hpp>

#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataNormals.h>

#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkImage.h>

using namespace niftk;
using namespace vl;

//-----------------------------------------------------------------------------
// niftk::VLUtils
//-----------------------------------------------------------------------------

vl::vec3 VLUtils::getVector3DProp( const mitk::DataNode* node, const char* prop_name, vl::vec3 defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::Vector3DProperty*>( node->GetProperty( prop_name ) ) );
  const mitk::Vector3DProperty* prop = dynamic_cast<const mitk::Vector3DProperty*>( node->GetProperty( prop_name ) );
  if ( prop ) {
    double* val = prop->GetValue().GetDataPointer();
    return vl::vec3( (float)val[0], (float)val[1], (float)val[2] );
  } else {
    return defval;
  }
}

//-----------------------------------------------------------------------------

vl::vec3 VLUtils::getPoint3DProp( const mitk::DataNode* node, const char* prop_name, vl::vec3 defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::Point3dProperty*>( node->GetProperty( prop_name ) ) );
  const mitk::Point3dProperty* prop = dynamic_cast<const mitk::Point3dProperty*>( node->GetProperty( prop_name ) );
  if ( prop ) {
    double* val = prop->GetValue().GetDataPointer();
    return vl::vec3( (float)val[0], (float)val[1], (float)val[2] );
  } else {
    return defval;
  }
}

//-----------------------------------------------------------------------------

vec4 VLUtils::getPoint4DProp( const mitk::DataNode* node, const char* prop_name, vec4 defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::Point4dProperty*>( node->GetProperty( prop_name ) ) );
  const mitk::Point4dProperty* prop = dynamic_cast<const mitk::Point4dProperty*>( node->GetProperty( prop_name ) );
  if ( prop ) {
    double* val = prop->GetValue().GetDataPointer();
    return vec4( (float)val[0], (float)val[1], (float)val[2], (float)val[3] );
  } else {
    return defval;
  }
}

//-----------------------------------------------------------------------------

int VLUtils::getEnumProp( const mitk::DataNode* node, const char* prop_name, int defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::EnumerationProperty*>( node->GetProperty( prop_name ) ) );
  const mitk::EnumerationProperty* prop = dynamic_cast<const mitk::EnumerationProperty*>( node->GetProperty( prop_name ) );
  if ( prop ) {
    return prop->GetValueAsId();
  } else {
    return defval;
  }
}

//-----------------------------------------------------------------------------

bool VLUtils::getBoolProp( const mitk::DataNode* node, const char* prop_name, bool defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::BoolProperty*>( node->GetProperty( prop_name ) ) );
  bool val = defval;
  node->GetBoolProperty( prop_name, val );
  return val;
}

//-----------------------------------------------------------------------------

bool VLUtils::setBoolProp( mitk::DataNode* node, const char* prop_name, bool val ) {
  VIVID_CHECK( dynamic_cast<const mitk::BoolProperty*>( node->GetProperty( prop_name ) ) );
  mitk::BoolProperty* prop = dynamic_cast<mitk::BoolProperty*>( node->GetProperty( prop_name ) );
  if ( ! prop ) {
    return false;
  }
  prop->SetValue( val );
  return true;
}

//-----------------------------------------------------------------------------

float VLUtils::getFloatProp( const mitk::DataNode* node, const char* prop_name, float defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::FloatProperty*>( node->GetProperty( prop_name ) ) );
  float val = defval;
  node->GetFloatProperty( prop_name, val );
  return val;
}

//-----------------------------------------------------------------------------

int VLUtils::getIntProp( const mitk::DataNode* node, const char* prop_name, int defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::IntProperty*>( node->GetProperty( prop_name ) ) );
  int val = defval;
  node->GetIntProperty( prop_name, val );
  return val;
}

//-----------------------------------------------------------------------------

vec4 VLUtils::getColorProp( const mitk::DataNode* node, const char* prop_name, vec4 defval ) {
  VIVID_CHECK( dynamic_cast<const mitk::ColorProperty*>( node->GetProperty( prop_name ) ) );
  float rgb[3] = { defval.r(), defval.g(), defval.b() };
  node->GetColor(rgb, NULL, prop_name );
  return vec4( rgb[0], rgb[1], rgb[2], defval.a() );
}

//-----------------------------------------------------------------------------

void VLUtils::initGlobalProps( mitk::DataNode* node )
{
  // initRenderModeProps(this);
  // initFogProps(this);
  // initClipProps(this);

  // Truly globals

  mitk::BoolProperty::Pointer enable = mitk::BoolProperty::New();
  node->AddProperty( "VL.Global.Stencil.Enable", enable );
  enable->SetValue( false );

  mitk::ColorProperty::Pointer stencil_bg_color = mitk::ColorProperty::New();
  node->AddProperty( "VL.Global.Stencil.BackgroundColor", stencil_bg_color );
  stencil_bg_color->SetValue( vl::black.ptr() );

  mitk::FloatProperty::Pointer stencil_smooth = mitk::FloatProperty::New();
  node->AddProperty( "VL.Global.Stencil.Smoothness", stencil_smooth );
  stencil_smooth->SetValue( 10 );

  VL_Render_Mode_Property::Pointer render_mode = VL_Render_Mode_Property::New();
  node->AddProperty( "VL.Global.RenderMode", render_mode );
  render_mode->SetValue( 0 );

  //mitk::ColorProperty::Pointer bg_color = mitk::ColorProperty::New();
  //node->AddProperty( "VL.Global.BackgroundColor", bg_color );
  //bg_color->SetValue( vl::black.ptr() );

  //mitk::FloatProperty::Pointer opacity = mitk::FloatProperty::New();
  //node->AddProperty( "VL.Global.Opacity", opacity );
  //opacity->SetValue( 1 );

  mitk::IntProperty::Pointer passes = mitk::IntProperty::New();
  node->AddProperty( "VL.Global.DepthPeelingPasses", passes );
  passes->SetValue( 4 );
}

//-----------------------------------------------------------------------------

void VLUtils::initVolumeProps( mitk::DataNode* node ) {
  // init only once if multiple views are open
  if ( node->GetProperty("VL.Volume.Mode") ) {
    return;
  }

  VL_Volume_Mode_Property::Pointer mode = VL_Volume_Mode_Property::New();
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

//-----------------------------------------------------------------------------

void VLUtils::updateVolumeProps( vl::VividVolume* vol, const mitk::DataNode* node )
{
  int mode = VLUtils::getEnumProp( node, "VL.Volume.Mode" );
  float iso = VLUtils::getFloatProp( node, "VL.Volume.Iso" );
  float density = VLUtils::getFloatProp( node, "VL.Volume.Density" );
  int samples = VLUtils::getIntProp( node, "VL.Volume.SamplesPerRay" );

  vol->setVolumeMode( (vl::VividVolume::EVolumeMode)mode );
  vol->setIsoValue( iso );
  vol->setVolumeDensity( density );
  vol->setSamplesPerRay( samples );
}

//-----------------------------------------------------------------------------

void VLUtils::initMaterialProps( mitk::DataNode* node )
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

//-----------------------------------------------------------------------------

void VLUtils::updateMaterialProps( Effect* fx, const mitk::DataNode* node )
{
#if 0
  vec4 color = VLUtils::getColorProp( node, "VL.Material.Color" );
  color.a() = VLUtils::getFloatProp( node, "VL.Material.Opacity" );
#else
  vec4 color = VLUtils::getColorProp( node, "color" );
  color.a() = VLUtils::getFloatProp( node, "opacity" );
#endif
  vec4 spec_color = VLUtils::getColorProp( node, "VL.Material.Specular.Color" );
  float shininess = VLUtils::getFloatProp( node, "VL.Material.Specular.Shininess" );

  Shader* sh = fx->shader();


  sh->getUniform( "vl_Vivid.material.diffuse" )->setUniform( color );
  sh->getUniform( "vl_Vivid.material.specular" )->setUniform( spec_color );
  // sh->getUniform( "vl_Vivid.material.ambient" )->setUniform( ... );
  // sh->getUniform( "vl_Vivid.material.emission" )->setUniform( ... );
  sh->getUniform( "vl_Vivid.material.shininess" )->setUniformF( shininess );
}

//-----------------------------------------------------------------------------

void VLUtils::initFogProps( mitk::DataNode* node )
{
  // init only once if multiple views are open
  if ( node->GetProperty("VL.Fog.Mode") ) {
    return;
  }

  // gocUniform("vl_Vivid.fog.mode")
  VL_Fog_Mode_Property::Pointer fog_mode = VL_Fog_Mode_Property::New();
  node->SetProperty("VL.Fog.Mode", fog_mode);
  fog_mode->SetValue( 0 );

  // gocUniform("vl_Vivid.fog.target")
  VL_Smart_Target_Property::Pointer fog_target = VL_Smart_Target_Property::New();
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

//-----------------------------------------------------------------------------

void VLUtils::updateFogProps( Effect* fx, const mitk::DataNode* node )
{
  int fog_mode = VLUtils::getEnumProp( node, "VL.Fog.Mode", 0 );
  int fog_target = VLUtils::getEnumProp( node, "VL.Fog.Target", 0 );
  vec4 fog_color = VLUtils::getColorProp( node, "VL.Fog.Color", vl::black );
  float fog_start = VLUtils::getFloatProp( node, "VL.Fog.Start", 0 );
  float fog_end = VLUtils::getFloatProp( node, "VL.Fog.End", 0 );
  float fog_density = VLUtils::getFloatProp( node, "VL.Fog.Density", 0 );

  Shader* sh = fx->shader();

  sh->getUniform("vl_Vivid.fog.mode")->setUniformI( fog_mode );
  sh->getUniform("vl_Vivid.fog.target")->setUniformI( fog_target );
  sh->getUniform("vl_Vivid.fog.color")->setUniform( fog_color );
  sh->getUniform("vl_Vivid.fog.start")->setUniformF( fog_start );
  sh->getUniform("vl_Vivid.fog.end")->setUniformF( fog_end );
  sh->getUniform("vl_Vivid.fog.density")->setUniformF( fog_density );
}

//-----------------------------------------------------------------------------

void VLUtils::initClipProps( mitk::DataNode* node )
{
  // init only once if multiple views are open
  if ( node->GetProperty("VL.Clip.0.Mode") ) {
    return;
  }

  #define CLIP_UNIT(field) (std::string("VL.Clip.") + i + '.' + field).c_str()

  for( char i = '0'; i < '4'; ++i ) {

    // gocUniform("vl_Vivid.smartClip[0].mode")
    VL_Clip_Mode_Property::Pointer mode = VL_Clip_Mode_Property::New();
    node->SetProperty(CLIP_UNIT("Mode"), mode);
    mode->SetValue( 0 );

    // gocUniform("vl_Vivid.smartClip[0].target")
    VL_Smart_Target_Property::Pointer target = VL_Smart_Target_Property::New();
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

//-----------------------------------------------------------------------------

void VLUtils::updateClipProps( Effect* fx, const mitk::DataNode* node )
{
  #define CLIP_UNIT(field) (std::string("VL.Clip.") + i + '.' + field).c_str()
  #define CLIP_UNIT2(field) (std::string("vl_Vivid.smartClip[") + i + "]." + field).c_str()

  for( char i = '0'; i < '4'; ++i ) {

    int mode = VLUtils::getEnumProp( node, CLIP_UNIT("Mode"), 0 );
    int targ = VLUtils::getEnumProp( node, CLIP_UNIT("Target"), 0 );
    vec4 color = VLUtils::getColorProp( node, CLIP_UNIT("Color"), vl::black );
    float range = VLUtils::getFloatProp( node, CLIP_UNIT("FadeRange"), 0 );
    vec4 plane = getPoint4DProp( node, CLIP_UNIT("Plane"), vec4(0,0,0,0) );
    vec4 sphere = getPoint4DProp( node, CLIP_UNIT("Sphere"), vec4(0,0,0,0) );
    vl::vec3 bmin = getPoint3DProp( node, CLIP_UNIT("BoxMin"), vl::vec3(0,0,0) );
    vl::vec3 bmax = getPoint3DProp( node, CLIP_UNIT("BoxMax"), vl::vec3(0,0,0) );
    bool reverse = VLUtils::getBoolProp( node, CLIP_UNIT("Reverse"), false );

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

//-----------------------------------------------------------------------------

void VLUtils::initRenderModeProps( mitk::DataNode* node )
{
  // init only once if multiple views are open
  if ( node->GetProperty("VL.SurfaceMode") ) {
    return;
  }

  // gocUniform("vl_Vivid.renderMode")
  VL_Surface_Mode_Property::Pointer mode = VL_Surface_Mode_Property::New();
  node->SetProperty("VL.SurfaceMode", mode);
  mode->SetValue( 0 );

  // gocUniform("vl_Vivid.outline.color")
  mitk::ColorProperty::Pointer outline_color = mitk::ColorProperty::New();
  node->SetProperty("VL.Outline.Color", outline_color);
  outline_color->SetValue( vl::yellow.ptr() );

  mitk::FloatProperty::Pointer outline_opacity = mitk::FloatProperty::New();
  node->SetProperty("VL.Outline.Opacity", outline_opacity);
  outline_opacity->SetValue( 1 );

  // gocUniform("vl_Vivid.outline.width")
  mitk::IntProperty::Pointer outline_width = mitk::IntProperty::New();
  node->SetProperty("VL.Outline.Width", outline_width);
  outline_width->SetValue( 2 );

  // gocUniform("vl_Vivid.outline.slicePlane")
  mitk::Point4dProperty::Pointer outline_slice_plane = mitk::Point4dProperty::New();
  node->SetProperty("VL.Outline.SlicePlane", outline_slice_plane);
  outline_slice_plane->SetValue( vec4(1,0,0,0).ptr() );
}

void VLUtils::updateRenderModeProps( Effect* fx, const mitk::DataNode* node ) {
  int mode = VLUtils::getEnumProp( node, "VL.SurfaceMode", 0 );
#if 1
  vec4 color = VLUtils::getColorProp( node, "VL.Outline.Color", vl::yellow );
  color.a() = VLUtils::getFloatProp( node, "VL.Outline.Opacity" );
#else
  vec4 color = VLUtils::getColorProp( node, "color" );
  color.a() = VLUtils::getFloatProp( node, "opacity" );
#endif
  int width = VLUtils::getIntProp( node, "VL.Outline.Width", 2 );
  vec4 slice_plane = getPoint4DProp( node, "VL.Outline.SlicePlane", vec4(0,0,0,0) );

  Shader* sh = fx->shader();

  sh->getUniform("vl_Vivid.renderMode")->setUniformI( mode );
  sh->getUniform("vl_Vivid.outline.color")->setUniform( color );
  sh->getUniform("vl_Vivid.outline.width")->setUniformF( (float)width );
  sh->getUniform("vl_Vivid.outline.slicePlane")->setUniform( slice_plane );
}

//-----------------------------------------------------------------------------

vl::EImageType VLUtils::mapITKPixelTypeToVL(int itkComponentType)
{
  const vl::EImageType typeMap[] =
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

vl::EImageFormat VLUtils::mapComponentsToVLColourFormat(int components)
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

ref<vl::Image> VLUtils::wrapMitk2DImage( const mitk::Image* mitk_image ) {
  mitk::PixelType  mitk_pixel_type = mitk_image->GetPixelType();
  vl::EImageType   vl_type         = mapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
  vl::EImageFormat vl_format       = mapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
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

VLUserData* VLUtils::getUserData(vl::Actor* actor)
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

mat4 VLUtils::getVLMatrix(const itk::Matrix<float, 4, 4>& itkmat)
{
  mat4 mat;
  mat.setNull();
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      mat.e(i, j) = itkmat[i][j];
    }
  }
  return mat;
}

//-----------------------------------------------------------------------------

// Returns null matrix if no vtk matrix is found
mat4 VLUtils::getVLMatrix(vtkSmartPointer<vtkMatrix4x4> vtkmat)
{
  mat4 mat;
  mat.setNull();
  if ( vtkmat.GetPointer() ) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        mat.e(i, j) = vtkmat->GetElement(i, j);
      }
    }
  }

  return mat;
}

//-----------------------------------------------------------------------------

// Returns null matrix if no vtk matrix is found
mat4 VLUtils::getVLMatrix(const mitk::BaseData* data)
{
  mat4 mat;
  mat.setNull();

  if ( data )
  {
    mitk::BaseGeometry::Pointer geom = data->GetGeometry();
    if ( geom ) {
      if ( geom->GetVtkTransform() ) {
        vtkSmartPointer<vtkMatrix4x4> vtkmat = vtkSmartPointer<vtkMatrix4x4>::New();
        geom->GetVtkTransform()->GetMatrix(vtkmat);
        mat = getVLMatrix(vtkmat);
      }
    }
  }

  return mat;
}

//-----------------------------------------------------------------------------

void VLUtils::updateTransform(vl::Transform* tr, const mitk::BaseData* data)
{
  mat4 m = getVLMatrix(data);

  if ( ! m.isNull() )
  {
    tr->setLocalMatrix(m);
    tr->computeWorldMatrix();
#if 0
    bool print_matrix = false;
    if ( print_matrix ) {
      printf("Transform: %p\n", tr );
      for(int i = 0; i < 4; ++i ) {
        printf("%f %f %f %f\n", m.e(0,i), m.e(1,i), m.e(2,i), m.e(3,i) );
      }
    }
#endif
  }
}

//-----------------------------------------------------------------------------

void VLUtils::updateTransform(vl::Transform* txf, const mitk::DataNode* node)
{
  if ( node ) {
    updateTransform(txf, node->GetData());
  }
}

//-----------------------------------------------------------------------------

ref<vl::Geometry> VLUtils::make2DImageGeometry(int width, int height)
{
  ref<vl::Geometry>    geom = new vl::Geometry;
  ref<vl::ArrayFloat3> vert = new vl::ArrayFloat3;
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

ref<vl::Geometry> VLUtils::getVLGeometry(vtkPolyData* vtkPoly)
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

  // setup triangles

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
      indices.push_back( vl::DrawElementsUInt::index_type(~0) );
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

  // setup normals

  // looks like the normals we get at this point are sometimes not up to date,
  // we may want to use vtkPolyDataNormals to generate them, for the moment we
  // dont bother and let VL compute them.
#if 0
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
#else
  // in VL if verts are shared across primitives they're smoothed out, VTK however seem to keep them flat.
  if ( ! vl_geom->normalArray() ) {
    vl_geom->computeNormals();
  }
#endif

  MITK_INFO << "Computing surface adjacency... ";

  vl_geom = vl::AdjacencyExtractor::extract( vl_geom.get() );

  vl_draw_elements = vl_geom->drawCalls().at(0)->as<vl::DrawElementsUInt>();

  MITK_INFO << "Surface data initialized. Points: " << points->GetNumberOfPoints() << ", Cells: " << primitives->GetNumberOfCells() << "\n";

  return vl_geom;
}

//-----------------------------------------------------------------------------

void VLUtils::dumpNodeInfo( const std::string& prefix, const mitk::DataNode* node ) {
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

//-----------------------------------------------------------------------------

void VLUtils::dumpNodeInfo( const std::string& prefix, const mitk::BaseData* data ) {
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

//-----------------------------------------------------------------------------

