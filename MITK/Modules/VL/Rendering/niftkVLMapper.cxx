/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLMapper.h"
#include "niftkVLSceneView.h"
#include "niftkVLUtils.h"
#include "niftkVLGlobalSettingsDataNode.h"
#include <vlGraphics/Actor.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlVivid/VividRendering.hpp>
#include <vlVivid/VividVolume.hpp>
#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkImage.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkPointSet.h>
#include <mitkDataStorage.h>
#include <mitkSurface.h>

#include <niftkCoordinateAxesData.h>

#ifdef _USE_PCL
  #include <niftkPCLData.h>
#endif

using namespace vl;

namespace niftk
{

//-----------------------------------------------------------------------------
// niftk::VLMapper
//-----------------------------------------------------------------------------

VLMapper::VLMapper( const mitk::DataNode* node, VLSceneView* sv )
{
  // Init
  VIVID_CHECK( node );
  VIVID_CHECK( sv );
  m_DataNode = node;
  m_VLSceneView = sv;
  m_OpenGLContext = sv->openglContext();
  m_VividRendering = sv->vividRendering();
  m_DataStorage = sv->dataStorage();
  m_DataNodeTrackingEnabled = true;
  VIVID_CHECK( m_OpenGLContext );
  VIVID_CHECK( m_VividRendering );
  VIVID_CHECK( m_DataStorage );
}

//-----------------------------------------------------------------------------

vl::ref<vl::Actor> VLMapper::initActor(vl::Geometry* geom, vl::Effect* effect, vl::Transform* transform)
{
  VIVID_CHECK( m_DataNode );
  VIVID_CHECK( m_VividRendering );
  vl::ref<vl::Effect> fx = effect ? effect : vl::VividRendering::makeVividEffect();
  vl::ref<vl::Transform> tr = transform ? transform : new vl::Transform;
  VLUtils::updateTransform( tr.get(), m_DataNode->GetData() );
  vl::ref<vl::Actor> actor = new vl::Actor( geom, fx.get(), tr.get() );
  actor->setEnableMask( vl::Vivid::VividEnableMask );
  return actor;
}

//-----------------------------------------------------------------------------

void VLMapper::updateCommon()
{
  if ( ! m_Actor )
  {
    return;
  }

  if ( isDataNodeTrackingEnabled() )
  {
    // Update visibility
    bool visible = VLUtils::getBoolProp( m_DataNode, "visible", true );
    m_Actor->setEnabled( visible );
  }

  // Update transform
  VLUtils::updateTransform( m_Actor->transform(), m_DataNode->GetData() );
}

//-----------------------------------------------------------------------------

vl::ref<VLMapper> VLMapper::create( const mitk::DataNode* node, VLSceneView* sv )
{
  // Map DataNode type to VLMapper type
  vl::ref<VLMapper> vl_node;

  const VLGlobalSettingsDataNode* vl_global = dynamic_cast<const VLGlobalSettingsDataNode*>(node);
  mitk::Surface*            mitk_surf = dynamic_cast<mitk::Surface*>(node->GetData());
  mitk::Image*              mitk_image = dynamic_cast<mitk::Image*>( node->GetData() );
  CoordinateAxesData* mitk_axes = dynamic_cast<CoordinateAxesData*>( node->GetData() );
  mitk::PointSet*           mitk_pset = dynamic_cast<mitk::PointSet*>( node->GetData() );
#ifdef _USE_PCL
  niftk::PCLData*           mitk_pcld = dynamic_cast<niftk::PCLData*>( node->GetData() );
#endif
#ifdef _USE_CUDA
  mitk::BaseData*           cuda_img = dynamic_cast<niftk::CUDAImage*>( node->GetData() );
  niftk::CUDAImageProperty* cuda_img_prop = dynamic_cast<niftk::CUDAImageProperty*>( node->GetData()->GetProperty("CUDAImageProperty").GetPointer() );
#endif

  if ( vl_global )
  {
    vl_node = new VLMapperVLGlobalSettings( node, sv );
  }
  else if ( mitk_surf )
  {
    vl_node = new VLMapperSurface( node, sv );
  }
  else
#ifdef _USE_CUDA
  if ( cuda_img || cuda_img_prop )
  {
    vl_node = new VLMapperCUDAImage( node, sv );
  }
  else
#endif
  if ( mitk_image )
  {
    unsigned int depth = mitk_image->GetDimensions()[2];
    if ( depth > 1 )
    {
      vl_node = new VLMapper3DImage( node, sv );
    }
    else
    {
      vl_node = new VLMapper2DImage( node, sv );
    }
  }
  else
  if ( mitk_axes )
  {
    vl_node = new VLMapperCoordinateAxes( node, sv );
  }
  else
  if ( mitk_pset )
  {
    vl_node = new VLMapperPointSet( node, sv );
  }
#ifdef _USE_PCL
  else
  if ( mitk_pcld )
  {
    vl_node = new VLMapperPCL( node, sv );
  }
#endif
  return vl_node;
}

//-----------------------------------------------------------------------------

VLMapperVLGlobalSettings::VLMapperVLGlobalSettings( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
}

bool VLMapperVLGlobalSettings::init()
{
  return true;
}

void VLMapperVLGlobalSettings::update()
{
  bool enable = VLUtils::getBoolProp( m_DataNode, "VL.Global.Stencil.Enable", false );
  vl::vec4 stencil_bg_color = VLUtils::getColorProp( m_DataNode, "VL.Global.Stencil.BackgroundColor", vl::black );
  float stencil_smooth = VLUtils::getFloatProp( m_DataNode, "VL.Global.Stencil.Smoothness", 10 );
  int render_mode = VLUtils::getEnumProp( m_DataNode, "VL.Global.RenderMode", 0 );
  // vl::vec4 bg_color = VLUtils::getColorProp( m_DataNode, "VL.Global.BackgroundColor", vl::black );
  // float opacity = VLUtils::getFloatProp( m_DataNode, "VL.Global.Opacity", 1 );
  // int passes = VLUtils::getIntProp( m_DataNode, "VL.Global.DepthPeelingPasses", 4 );

  m_VLSceneView->setStencilEnabled( enable );
  m_VLSceneView->setStencilBackgroundColor( stencil_bg_color );
  m_VLSceneView->setStencilSmoothness( stencil_smooth );
  m_VLSceneView->setRenderingMode( (vl::Vivid::ERenderingMode)render_mode );
  // m_VividRendering->setBackgroundColor( bg_color );
  // m_VividRendering->setOpacity( opacity );
  // m_VLSceneView->setDepthPeelingPasses( passes );
}

void VLMapperVLGlobalSettings::updateVLGlobalSettings()
{
  /* we don't have anything to set */
}

//-----------------------------------------------------------------------------

VLMapperSurface::VLMapperSurface( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
  m_MitkSurf = dynamic_cast<mitk::Surface*>( node->GetData() );
  VIVID_CHECK( m_MitkSurf );
}

bool VLMapperSurface::init()
{
  VIVID_CHECK( m_MitkSurf );

  mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
  VLUtils::initRenderModeProps( node );
  VLUtils::initMaterialProps( node );
  VLUtils::initFogProps( node );
  VLUtils::initClipProps( node );

  vl::ref<vl::Geometry> geom = VLUtils::getVLGeometry( m_MitkSurf->GetVtkPolyData() );
  if ( ! geom ) {
    return false;
  }

  m_Actor = initActor( geom.get() );
  m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );

  return true;
}

void VLMapperSurface::update()
{
  updateCommon();
  if ( isDataNodeTrackingEnabled() )
  {
    VLUtils::updateMaterialProps( m_Actor->effect(), m_DataNode );
    VLUtils::updateRenderModeProps( m_Actor->effect(), m_DataNode );
    VLUtils::updateFogProps( m_Actor->effect(), m_DataNode );
    VLUtils::updateClipProps( m_Actor->effect(), m_DataNode );

    // Stencil
    bool is_stencil = VLUtils::getBoolProp( m_DataNode, "VL.IsStencil", false );
    setIsStencil( is_stencil );
  }
}

//-----------------------------------------------------------------------------

VLMapper2DImage::VLMapper2DImage( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
  m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
  VIVID_CHECK( m_MitkImage );
}

bool VLMapper2DImage::init()
{
  VIVID_CHECK( m_MitkImage );

  mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
  // initRenderModeProps( node ); /* does not apply */
  VLUtils::initFogProps( node );
  VLUtils::initClipProps( node );

  vl::ref<vl::Image> img = VLUtils::wrapMitk2DImage( m_MitkImage );
  vl::ref<vl::Geometry> geom = VLUtils::make2DImageGeometry( img->width(), img->height() );

  m_VertexArray = geom->vertexArray()->as<vl::ArrayFloat3>(); VIVID_CHECK( m_VertexArray );
  m_TexCoordArray = geom->vertexArray()->as<vl::ArrayFloat3>(); VIVID_CHECK( m_TexCoordArray );

  m_Actor = initActor( geom.get() );
  // NOTE: for the moment we don't render it
  // FIXME: the DataNode itself should be disabled at init time?
  // m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  vl::ref<vl::Effect> fx = m_Actor->effect();

  // These must be present as part of the default Vivid material
  VIVID_CHECK( fx->shader()->getTextureSampler( vl::Vivid::UserTexture ) )
  VIVID_CHECK( fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture() )
  VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture2D") );
  VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture2D")->getUniformI() == vl::Vivid::UserTexture );
  vl::ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );
  fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
  fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
  // When texture mapping is enabled the texture is modulated by the vertex color, including the alpha
  geom->setColorArray( vl::white );

  return true;
}

void VLMapper2DImage::update()
{
  VIVID_CHECK( m_MitkImage );

  updateCommon();
  if ( isDataNodeTrackingEnabled() )
  {
    // VLUtils::updateRenderModeProps(); /* does not apply here */
    VLUtils::updateFogProps( m_Actor->effect(), m_DataNode );
    VLUtils::updateClipProps( m_Actor->effect(), m_DataNode );
  }

  if ( m_MitkImage->GetVtkImageData()->GetMTime() <= VLUtils::getUserData( m_Actor.get() )->m_ImageModifiedTime )
  {
    return;
  }

  vl::Texture* tex = m_Actor->effect()->shader()->gocTextureSampler( vl::Vivid::UserTexture )->texture();
  VIVID_CHECK( tex );
  vl::ref<vl::Image> img = VLUtils::wrapMitk2DImage( m_MitkImage );
  tex->setMipLevel(0, img.get(), false);
  VLUtils::getUserData( m_Actor.get() )->m_ImageModifiedTime = m_MitkImage->GetVtkImageData()->GetMTime();
}

//-----------------------------------------------------------------------------

VLMapper3DImage::VLMapper3DImage( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
  m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
  m_VividVolume = new vl::VividVolume( m_VividRendering );
  VIVID_CHECK( m_MitkImage );
}

bool VLMapper3DImage::init()
{
  mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
  VLUtils::initVolumeProps( node );

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
  vl::ref<vl::Image> vl_img;

  try
  {
    mitk::ImageReadAccessor image_reader( m_MitkImage, m_MitkImage->GetVolumeData(0) );
    void* img_ptr = const_cast<void*>( image_reader.GetData() );
    unsigned int buffer_bytes = (dims[0] * dims[1] * dims[2]) * mitk_pixel_type.GetSize();

    vl::EImageType   vl_type   = VLUtils::mapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
    vl::EImageFormat vl_format = VLUtils::mapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());

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
  if ( m_MitkImage->GetGeometry() )
  {
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
  // MIC: not sure if we need this
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
  mat4 mat(vals);
  tr->setLocalMatrix(mat);
#endif
  return true;
}

void VLMapper3DImage::update()
{
  updateCommon();
  // Neutralize scaling - screws up our rendering.
  // VTK seems to need it to render non cubic volumes.
  // NOTE: we assume there is no rotation.
  m_Actor->transform()->localMatrix().e(0,0) =
  m_Actor->transform()->localMatrix().e(1,1) =
  m_Actor->transform()->localMatrix().e(2,2) = 1;
  m_Actor->transform()->computeWorldMatrix();
  VLUtils::updateVolumeProps( m_VividVolume.get(), m_DataNode );
}

//-----------------------------------------------------------------------------

VLMapperCoordinateAxes::VLMapperCoordinateAxes( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
  m_MitkAxes = dynamic_cast<CoordinateAxesData*>( node->GetData() );
  VIVID_CHECK( m_MitkAxes );
}

bool VLMapperCoordinateAxes::init()
{
  VIVID_CHECK( m_MitkAxes );

  vl::ref<vl::ArrayFloat3> verts  = m_Vertices = new vl::ArrayFloat3;
  vl::ref<vl::ArrayFloat4> colors = new vl::ArrayFloat4;
  verts->resize(6);
  colors->resize(6);

  // Axis length
  int S = 100;
  mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
  if ( size_prop )
  {
    S = size_prop->GetValue();
  }

  // X Axis
  verts ->at(0) = vl::vec3(0, 0, 0);
  verts ->at(1) = vl::vec3(S, 0, 0);
  colors->at(0) = vl::red;
  colors->at(1) = vl::red;
  // Y Axis
  verts ->at(2) = vl::vec3(0, 0, 0);
  verts ->at(3) = vl::vec3(0, S, 0);
  colors->at(2) = vl::green;
  colors->at(3) = vl::green;
  // Z Axis
  verts ->at(4) = vl::vec3(0, 0, 0);
  verts ->at(5) = vl::vec3(0, 0, S);
  colors->at(4) = vl::blue;
  colors->at(5) = vl::blue;

  vl::ref<vl::Geometry> geom = new vl::Geometry;
  geom->drawCalls().push_back( new vl::DrawArrays( vl::PT_LINES, 0, 6 ) );
  geom->setVertexArray(verts.get());
  geom->setColorArray(colors.get());

  m_Actor = initActor( geom.get() );
  m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  vl::ref<vl::Effect> fx = m_Actor->effect();

  fx->shader()->getLineWidth()->set( 2 );
  // Use color array instead of lighting
  fx->shader()->gocUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );

  return true;
}

void VLMapperCoordinateAxes::update()
{
  updateCommon();
  // updateRenderModeProps();
  // updateFogProps();
  // updateClipProps();

  mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
  if ( size_prop )
  {
    vl::ref<vl::ArrayFloat3> verts = m_Vertices;
    int S = size_prop->GetValue();
    // X Axis
    verts ->at(0) = vl::vec3(0, 0, 0);
    verts ->at(1) = vl::vec3(S, 0, 0);
    // Y Axis
    verts ->at(2) = vl::vec3(0, 0, 0);
    verts ->at(3) = vl::vec3(0, S, 0);
    // Z Axis
    verts ->at(4) = vl::vec3(0, 0, 0);
    verts ->at(5) = vl::vec3(0, 0, S);
    // Update VBO
    m_Vertices->updateBufferObject();
  }
}

//-----------------------------------------------------------------------------

VLMapperPoints::VLMapperPoints( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv )
{
  m_3DSphereMode = true;
  m_Point2DFX = vl::VividRendering::makeVividEffect();
  m_PositionArray = new vl::ArrayFloat3;
  m_ColorArray = new vl::ArrayFloat4;
  m_DrawPoints = new vl::DrawArrays( vl::PT_POINTS, 0, 0 );
}

void VLMapperPoints::initPointSetProps()
{
  // init only once if multiple views are open
  if ( m_DataNode->GetProperty("VL.Point.Mode") )
  {
    return;
  }

  mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
  // initRenderModeProps( node ); /* does not apply to points */
  VLUtils::initFogProps( node );
  VLUtils::initClipProps( node );

  VL_Point_Mode_Property::Pointer point_set_mode = VL_Point_Mode_Property::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Mode", point_set_mode);
  point_set_mode->SetValue( 0 );

  mitk::FloatProperty::Pointer point_size_2d = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Size2D", point_size_2d);
  point_size_2d->SetValue( 10 );

  mitk::FloatProperty::Pointer point_size_3d = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Size3D", point_size_3d);
  point_size_3d->SetValue( 1 );

  mitk::FloatProperty::Pointer point_opacity = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Opacity", point_opacity);
  point_opacity->SetValue( 1 );

  mitk::ColorProperty::Pointer point_color = mitk::ColorProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.Point.Color", point_color);
  point_color->SetValue( vl::yellow.ptr() );
}

bool VLMapperPoints::init()
{
  initPointSetProps();
  return true;
}

void VLMapperPoints::init3D() {
  VIVID_CHECK( m_3DSphereMode );

  // Remove 2D data and init 3D data.
  remove();
  m_SphereActors = new vl::ActorTree;
  m_VividRendering->sceneManager()->tree()->addChild( m_SphereActors.get() );

  m_3DSphereGeom = vl::makeIcosphere( vl::vec3(0,0,0), 1, 2, true );
  for( int i = 0; i < m_PositionArray->size(); ++i )
  {
    const vl::vec3& pos = m_PositionArray->at( i );
    vl::ref<vl::Actor> actor = initActor( m_3DSphereGeom.get() );
    actor->transform()->setLocalAndWorldMatrix( vl::mat4::getTranslation( pos ) );
    m_SphereActors->addActor( actor.get() );
    // Colorize the sphere with the point's color
    actor->effect()->shader()->gocUniform( "vl_Vivid.material.diffuse" )->setUniform( m_ColorArray->at( i ) );
  }
}

void VLMapperPoints::init2D()
{
  VIVID_CHECK( ! m_3DSphereMode );

  // Remove 3D data and init 2D data.
  remove();

  // Initialize color array
  for( int i = 0; i < m_ColorArray->size(); ++i )
  {
    m_ColorArray->at( i ) = vl::white;
  }

  m_2DGeometry = new vl::Geometry;
  m_DrawPoints = new vl::DrawArrays( vl::PT_POINTS, 0, m_PositionArray->size() );
  m_2DGeometry->drawCalls().push_back( m_DrawPoints.get() );
  m_2DGeometry->setVertexArray( m_PositionArray.get() );
  m_2DGeometry->setColorArray( m_ColorArray.get() );

  m_Actor = initActor( m_2DGeometry.get(), m_Point2DFX.get() );
  m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  vl::ref<vl::Effect> fx = m_Actor->effect();
  vl::ref<vl::Image> img = new vl::Image("/vivid/images/sphere.png");
  vl::ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );

  // 2d mode settings
  m_Point2DFX->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );
  m_Point2DFX->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( 1 );
  m_Point2DFX->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( 1 );
}

void VLMapperPoints::update()
{
  // updateCommon();

  // Get mode
  m_3DSphereMode = 0 == VLUtils::getEnumProp( m_DataNode, "VL.Point.Mode", 0 );

  // Get visibility
  bool visible = VLUtils::getBoolProp( m_DataNode, "visible", true );

  // Get point size
  float pointsize = VLUtils::getFloatProp( m_DataNode, m_3DSphereMode ? "VL.Point.Size3D" : "VL.Point.Size2D", 1.0f );

  // Get color
  vl::vec4 color = VLUtils::getColorProp( m_DataNode, "VL.Point.Color", vl::white );

  // Get opacity
  color.a() = VLUtils::getFloatProp( m_DataNode, "VL.Point.Opacity", 1.0f );

  updatePoints( color );

  if ( m_3DSphereMode )
  {
    if ( ! m_SphereActors )
    {
      init3D();
    }

    for( int i = 0; i < m_SphereActors->actors()->size(); ++i )
    {
      // Set visible
      vl::Actor* act = m_SphereActors->actors()->at( i );
      act->setEnabled( visible );
      // Set color/opacity
      act->effect()->shader()->gocUniform( "vl_Vivid.material.diffuse" )->setUniform( m_ColorArray->at( i ) );
      // Update other Vivid settings
      // VLUtils::updateRenderModeProps(); /* does not apply here */
      VLUtils::updateFogProps( act->effect(), m_DataNode );
      VLUtils::updateClipProps( act->effect(), m_DataNode );
      // Set size
      vl::Transform* tr = act->transform();
      vl::mat4& local = tr->localMatrix();
      local.e(0,0) = pointsize * 2;
      local.e(1,1) = pointsize * 2;
      local.e(2,2) = pointsize * 2;
      tr->computeWorldMatrix();
    }
  }
  else
  {
    if ( ! m_2DGeometry )
    {
      init2D();
    }

    VIVID_CHECK( m_Actor );
    VIVID_CHECK( m_Point2DFX->shader()->getPointSize() );

    // VLUtils::updateRenderModeProps(); /* does not apply here */
    VLUtils::updateFogProps( m_Point2DFX.get(), m_DataNode );
    VLUtils::updateClipProps( m_Point2DFX.get(), m_DataNode );

    // set point size
    m_Point2DFX->shader()->getPointSize()->set( pointsize );
  }
}

void VLMapperPoints::remove()
{
  VLMapper::remove();
  m_2DGeometry = NULL;
  if ( m_SphereActors )
  {
    m_SphereActors->actors()->clear();
    m_VividRendering->sceneManager()->tree()->eraseChild( m_SphereActors.get() );
    m_SphereActors = NULL;
    m_3DSphereGeom = NULL;
  }
}

//-----------------------------------------------------------------------------

VLMapperPointSet::VLMapperPointSet( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapperPoints( node, sv ) {
  m_MitkPointSet = dynamic_cast<mitk::PointSet*>( node->GetData() );
  VIVID_CHECK( m_MitkPointSet );
}

void VLMapperPointSet::updatePoints( const vl::vec4& color ) {
  VIVID_CHECK( m_MitkPointSet );

  // If point set size changed force a rebuild of the 3D spheres, actors etc.
  // TODO: use event listeners instead of this brute force approach
  if ( m_PositionArray->size() != m_MitkPointSet->GetSize() )
  {
    if ( m_3DSphereMode )
    {
      remove();
    }
    else
    {
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

//-----------------------------------------------------------------------------

#ifdef _USE_PCL

VLMapperPCL::VLMapperPCL( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapperPoints( node, sv ) {
  m_NiftkPCL = dynamic_cast<niftk::PCLData*>( node->GetData() );
  VIVID_CHECK( m_NiftkPCL );
}

void VLMapperPCL::updatePoints( const vl::vec4& /*color*/ ) {
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

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA

VLMapperCUDAImage::VLMapperCUDAImage( const mitk::DataNode* node, VLSceneView* sv )
  : VLMapper( node, sv ) {
  m_CudaResource = NULL;
}

niftk::LightweightCUDAImage VLMapperCUDAImage::getLWCI()
{
  niftk::LightweightCUDAImage lwci;
  niftk::CUDAImage* cuda_image = dynamic_cast<niftk::CUDAImage*>( m_DataNode->GetData() );
  if ( cuda_image )
  {
    lwci = cuda_image->GetLightweightCUDAImage();
  }
  else
  {
    niftk::CUDAImageProperty* cuda_img_prop = dynamic_cast<niftk::CUDAImageProperty*>( m_DataNode->GetData()->GetProperty("CUDAImageProperty").GetPointer() );
    if  (cuda_img_prop )
    {
      lwci = cuda_img_prop->Get();
    }
  }
  VIVID_CHECK(lwci.GetId() != 0);
  return lwci;
}

bool VLMapperCUDAImage::init()
{
  mitk::DataNode* node = const_cast<mitk::DataNode*>( m_DataNode );
  // VLUtils::initRenderModeProps( node ); /* does not apply */
  VLUtils::initFogProps( node );
  VLUtils::initClipProps( node );

  niftk::LightweightCUDAImage lwci = getLWCI();

  vl::ref<vl::Geometry> vlquad = VLUtils::make2DImageGeometry( lwci.GetWidth(), lwci.GetHeight() );

  m_Actor = initActor( vlquad.get() );
  // NOTE: for the moment we don't render it
  // FIXME: the DataNode itself should be disabled at init time?
  // m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  vl::Effect* fx = m_Actor->effect();

  fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
  fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
  vlquad->setColorArray( vl::white );

  m_Texture = fx->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  VIVID_CHECK( m_Texture );
  VIVID_CHECK( m_Texture->handle() );
  cudaError_t err = cudaSuccess;
  err = cudaGraphicsGLRegisterImage( &m_CudaResource, m_Texture->handle(), m_Texture->dimension(), cudaGraphicsRegisterFlagsNone );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error("cudaGraphicsGLRegisterImage() failed.");
    return false;
  }
  return true;
}

void VLMapperCUDAImage::update()
{
  updateCommon();
  if ( isDataNodeTrackingEnabled() )
  {
    // VLUtils::updateRenderModeProps(); /* does not apply here */
    VLUtils::updateFogProps( m_Actor->effect(), m_DataNode );
    VLUtils::updateClipProps( m_Actor->effect(), m_DataNode );
  }

  // Get the niftk::LightweightCUDAImage

  niftk::LightweightCUDAImage lwci = getLWCI();

  cudaError_t err = cudaSuccess;

  // Update texture size and cuda graphics resource

  if ( m_Texture->width() != lwci.GetWidth() || m_Texture->height() != lwci.GetHeight() )
  {
    VIVID_CHECK(m_CudaResource);
    cudaGraphicsUnregisterResource(m_CudaResource);
    m_CudaResource = NULL;
    m_Texture->createTexture2D( lwci.GetWidth(), lwci.GetHeight(), vl::TF_RGBA, false );
    err = cudaGraphicsGLRegisterImage( &m_CudaResource, m_Texture->handle(), m_Texture->dimension(), cudaGraphicsRegisterFlagsNone );
    if ( err != cudaSuccess )
    {
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

void VLMapperCUDAImage::remove()
{
  if ( m_CudaResource )
  {
    cudaError_t err = cudaGraphicsUnregisterResource( m_CudaResource );
    if (err != cudaSuccess)
    {
      MITK_WARN << "cudaGraphicsUnregisterResource() failed.";
    }
    m_CudaResource = NULL;
  }

  VLMapper::remove();
}

#endif

}
