/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLMapper_h
#define niftkVLMapper_h

#include <niftkVLExports.h>
#include <niftkVLUtils.h>

#include <vlCore/Vector3.hpp>
#include <vlCore/Vector4.hpp>
#include <vlCore/vlnamespace.hpp>
#include <vlGraphics/Texture.hpp>
#include <vlVivid/VividRendering.hpp>

#include <mitkDataNode.h>
#include <mitkBaseData.h>

#ifdef _USE_CUDA
  #include <niftkCUDAManager.h>
  #include <niftkCUDAImage.h>
  #include <niftkLightweightCUDAImage.h>
  #include <niftkCUDAImageProperty.h>
  #include <niftkFlipImageLauncher.h>
  #include <cuda_gl_interop.h>
#endif

#define VL_CUDA_STREAM_NAME "VL-CUDA-STREAM"

namespace mitk
{
class DataStorage;
class PointSet;
class Surface;
}

namespace niftk
{

class CoordinateAxesData;
class PCLData;
class VLSceneView;

/**
 * \brief A VL representation of a mitk::DataNode for rendering purposes.
 * 
 * The niftk::VLSceneView class keeps a map of mitk::DataNode -> niftk::VLMapper according to the events it receives
 * from the data storage. Overall when a new data node is added to the store a new VLMapper is created
 * and its init() and update() methods called. When a data node is removed its VLMapper is also removed
 * after calling its VLMapper::remove() method which cleans up the VL rendering related bits. When a data node
 * is updated its relative VLMapper::update() method is called. VLMapper keeps an internal pointer to the
 * tracked mitk::DataNode so that on update() it can fetch all its values and update accordingly.
 */
class NIFTKVL_EXPORT VLMapper : public vl::Object
{
public:
  VLMapper(const mitk::DataNode* node, VLSceneView* sv);

  /**
   * When a VLMapper is destroyed its remove() method must have been called by the VLSceneView.
   */
  virtual ~VLMapper() { VIVID_CHECK( ! m_Actor ); }

  /** 
   * Used by niftk::VLSceneview to create the appropriate niftk::VLMapper given a mitk::DataNode.
   */
  static vl::ref<VLMapper> create(const mitk::DataNode* node, VLSceneView*);

  /** 
   * Initializes all the relevant VL data structures, uniforms etc. according to the node's settings. 
   */
  virtual bool init() = 0;

  /** 
   * Updates all the relevant VL data structures, uniforms etc. according to the node's settings. 
   */
  virtual void update() = 0;

  /** 
   * Removes all the relevant Actor(s) from the scene. 
   */
  virtual void remove() {
    m_VividRendering->sceneManager()->tree()->eraseActor(m_Actor.get());
    m_Actor = 0;
  }

  /** 
   * Utility function to update the default niftk::VLMapper::actor()'s visibility and transform. 
   * Note that not all VLMappers use the default Actor.
   */
  void updateCommon();

  /** 
   * Returns the default vl::Actor associated with this VLMapper used by those VLMappers that map a data node to a single vl::Actor.
   * Note: more complex VLMappers may not use the default Actor and instantiate their own ones.
   */
  vl::Actor* actor() { return m_Actor.get(); }
  const vl::Actor* actor() const { return m_Actor.get(); }

  //--------------------------------------------------------------------------------

  /** 
   * When enabled (default) the mapper will reflect updates to the VL.* variables coming from the DataNode.
   * Enable this when you want a data node to have its settings update all VLWidgets/VLSceneViews.
   * Disable this when you want a data node to have different settings across different VLWidgets/VLSceneViews. 
   * Updates to the "visible" property are also ignored while updates to the transform are never ignored.
   * At the moment this only applies to VLMapperSurface, VLMapper2DImage, VLMapperCUDAImage. 
   */
  void setDataNodeVividUpdateEnabled( bool enable ) { m_DataNodeVividUpdateEnabled = enable; }
  bool isDataNodeVividUpdateEnabled() const { return m_DataNodeVividUpdateEnabled; }

  //--------------------------------------------------------------------------------
  // User managed Vivid API to be used when isDataNodeVividUpdateEnabled() == false
  //--------------------------------------------------------------------------------

  // -- At the moment this only applies to VLMapperSurface, VLMapper2DImage, VLMapperCUDAImage ---

  // -- Rendering Mode --

  /**
   * Whether a surface is rendered with polygons, 3D outline, 2D outline or an outline-slice through a plane.
   * 3D outlines & slice mode:
   * - computed on the GPU by a geometry shader
   * - are clipped by the stencil
   * - interact with the depth buffer (ie they're visible only if they're in front of other objects)
   * - include creases inside the silhouette regardless of whether they're facing or not the viewer
   * 2D outlines:
   * - computed using offscreen image based edge detection
   * - are not clipped against the stencil
   * - do not interact with depth buffer (ie they're always in front of any geometry)
   * - look cleaner, renders only the external silhouette of an object
  */
  void setRenderingMode(vl::Vivid::ERenderingMode mode) {
    actor()->effect()->shader()->getUniform("vl_Vivid.renderMode")->setUniformI( mode );
  }
  vl::Vivid::ERenderingMode renderingMode() const {
    return (vl::Vivid::ERenderingMode)actor()->effect()->shader()->getUniform("vl_Vivid.renderMode")->getUniformI();
  }

  // --- Outline properties of both the 2D and 3D outlines ---

  /**
   * The ouline color and transparency.
   */
  void setOutlineColor(const vl::vec4& color ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.color")->setUniform( color );
  }
  vl::vec4 outlineColor() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.color")->getUniform4F();
  }

  /**
   * The outline width in pixels (approximately).
   */
  void setOutlineWidth( float width ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.width")->setUniformF( width );
  }
  float outlineWidth() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.width")->getUniformF();
  }

  /** 
   * The plane equation to be used when rendering mode slicing mode is enabled.
   */
  void setOutlineSlicePlane( const vl::vec4& plane ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.slicePlane")->setUniform( plane );
  }
  vl::vec4 outlineSlicePlane() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.slicePlane")->getUniform4F();
  }

  // --- Stencil --- 

  /**
   * Use this VLMapper's Actor as stencil when VLSceneView->isStencilEnabled() == true.
   */
  void setIsStencil( bool is_stencil ) {
    VIVID_CHECK( actor() );
    std::vector< vl::ref<vl::Actor> >::iterator it = std::find( m_VividRendering->stencilActors().begin(), m_VividRendering->stencilActors().end(), actor() );
    if ( ! is_stencil && it != m_VividRendering->stencilActors().end() ) {
      m_VividRendering->stencilActors().erase( it );
    } else
    if ( is_stencil && it == m_VividRendering->stencilActors().end() ) {
      m_VividRendering->stencilActors().push_back( m_Actor );
    }
  }
  bool isStencil() const {
    return std::find( m_VividRendering->stencilActors().begin(), m_VividRendering->stencilActors().end(), actor() ) != m_VividRendering->stencilActors().end();
  }

  // --- Lighting properties ---

  /**
   * Enable/disable ligthing.
   * When lighting is enabled:
   *  - the vertex color is computed using the material properties below and the default light following the camera.
   *  - the object opacity is determined by its diffuse alpha value.
   * When lighting is disabled:
   *  - the vertex color is computed using the Actor's Geometry's colorArray() (required)
   *  - the object opacity is determined on a per-vertex basis according to the Actor's Geometry's colorArray()
   */
  void setLightingEnabled( bool enable ) {
    actor()->effect()->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( enable );
  }
  int isLightingEnabled() const {
    return actor()->effect()->shader()->getUniform( "vl_Vivid.enableLighting" )->getUniformI();
  }

  // --- Material and opacity properties (requires isLightingEnabled() == true) ---

  /**
   * The diffuse color and opacity of this object.
   */
  void setMaterialDiffuseRGBA(const vl::vec4& rgba) {
    actor()->effect()->shader()->getMaterial()->setFrontDiffuse( rgba );
  }
  const vl::vec4& materialDiffuseRGBA() const {
    return actor()->effect()->shader()->getMaterial()->frontDiffuse();
  }

  /**
   * The specular color of this object.
   */
  void setMaterialSpecularColor(const vl::vec4& color ) {
    actor()->effect()->shader()->getMaterial()->setFrontSpecular( color );
  }
  const vl::vec4& materialSpecularColor() const {
    return actor()->effect()->shader()->getMaterial()->frontSpecular();
  }

  /**
   * The specular shininess of this object as defined by OpenGL.
   */
  void setMaterialSpecularShininess( float shininess ) {
    actor()->effect()->shader()->getMaterial()->setFrontShininess( shininess );
  }
  float materialSpecularShininess() const {
    return actor()->effect()->shader()->getMaterial()->frontShininess();
  }

  // --- Texturing and Point Sprites ---
  // At the moment we only support one 2D texture. 1D and 3D texture support is experimental.

  /**
   * Enable/disable texture mapping for this object.
   */
  void setTextureMappingEnabled( bool enable ) {
    actor()->effect()->shader()->getUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( enable );
  }
  bool isTextureMappingEnabled() const {
    return actor()->effect()->shader()->getUniform( "vl_Vivid.enableTextureMapping" )->getUniformI();
  }

  /**
   * The 2D texture to be used when rendering this object.
   * The specific texture coordinates used depend on the VLMapper subclass.
   */
  void setTexture( vl::Texture* tex ) {
    actor()->effect()->shader()->gocTextureSampler( vl::Vivid::UserTexture )->setTexture( tex );
  }
  vl::Texture* texture() {
    return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  }
  const vl::Texture* texture() const {
    return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  }

  /**
   * Enable/disable point sprites rendering (requires isTextureMappingEnabled() == true)
   * This makes sense only if the VLMapper is rendering points using vl::PT_POINTS.
   * \sa pointSize()
   */
  void setPointSpriteEnabled( bool enable ) {
    actor()->effect()->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( enable );
  }
  bool isPointSpriteEnabled() const {
    return actor()->effect()->shader()->getUniform( "vl_Vivid.enablePointSprite" )->getUniformI();
  }

  /**
   * The size in pixels of the point or point sprites being rendered.
   */
  vl::PointSize* pointSize() { return actor()->effect()->shader()->getPointSize(); }
  const vl::PointSize* pointSize() const { return actor()->effect()->shader()->getPointSize(); }

  // --- PolygonMode ---

  /**
   * Useful to render surfaces, boxes etc. in wireframe.
   */
  vl::PolygonMode* polygonMode() { return actor()->effect()->shader()->getPolygonMode(); }
  const vl::PolygonMode* polygonMode() const { return actor()->effect()->shader()->getPolygonMode(); }

  // --- "Smart" Fogging ---
  
  /**
   * Enable/disable fogging and sets linear, exp or exp2 mode.
   * Fog behaves just like in standard OpenGL (see red book for settings) except that instead of just targeting the color
   * we can target also alpha and saturation.
   */
  void setFogMode( vl::Vivid::EFogMode mode ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.mode")->setUniformI( mode );
  }
  vl::Vivid::EFogMode fogMode() const {
    return (vl::Vivid::EFogMode)actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.mode")->getUniformI();
  }

  /**
   * The fog target: color, alpha, saturation.
   */
  void setFogTarget( vl::Vivid::ESmartTarget target ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.target")->setUniformI( target );
  }
  vl::Vivid::ESmartTarget fogTarget() const {
    return (vl::Vivid::ESmartTarget)actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.target")->getUniformI();
  }

  /** 
   * The fog color as per standard OpenGL.
   */
  void setFogColor( const vl::vec4& color ) {
    actor()->effect()->shader()->gocFog()->setColor( color );
  }
  const vl::vec4& fogColor() const {
    return actor()->effect()->shader()->getFog()->color();
  }

  /** 
   * The fog start in camera coordinates as per standard OpenGL (only used if mode == linear)
   */
  void setFogStart( float start ) {
    actor()->effect()->shader()->gocFog()->setStart( start );
  }
  float fogStart() const {
    return actor()->effect()->shader()->getFog()->start();
  }

  /** 
   * The fog end in camera coordinates as per standard OpenGL  (only used if mode == linear)
   */
  void setFogEnd( float end ) {
    actor()->effect()->shader()->gocFog()->setEnd( end );
  }
  float fogEnd() const {
    return actor()->effect()->shader()->getFog()->end();
  }

  /** 
   * The fog density in camera coordinates as per standard OpenGL  (only used if mode == exp or exp2)
   */
  void setFogDensity( float density ) {
    actor()->effect()->shader()->gocFog()->setDensity( density );
  }
  float fogDensity() const {
    return actor()->effect()->shader()->getFog()->density();
  }

  // --- "Smart" Clipping ---

  #define VL_SMARTCLIP(var) (std::string("vl_Vivid.smartClip[") + (char)('0' + i) + "]." + var).c_str()

  /**
   * Enable/disable clipping unit and sets clipping mode.
   * We can have up to 4 "clipping units" active (`i` parameter).
   * Each clipping unit can be independently enabled with its clipping mode:
   * - Plane: clipping is performed according to the clipPlane() equation (world space).
   * - Sphere: clipping is performed according to the clipSphere() settings (world space).
   * - Box: clipping is performed according to the clipBoxMin/Max() settings (world space).
   * We can target: color, alpha and saturation -> setClipTarget()
   * We can have soft clipping: setClipFadeRange()
   * We can reverse the clipping effect: setClipReverse(), by default the negative/outside space is the one "clipped".
  */
  void setClipMode( int i, vl::Vivid::EClipMode mode ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("mode"))->setUniformI( mode );
  }
  vl::Vivid::EClipMode clipMode( int i ) const {
    return (vl::Vivid::EClipMode)actor()->effect()->shader()->getUniform(VL_SMARTCLIP("mode"))->getUniformI();
  }

  /**
   * The clipping target: color, alpha, saturation.
   */
  void setClipTarget( int i, vl::Vivid::ESmartTarget target ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("target"))->setUniformI( target );
  }
  vl::Vivid::ESmartTarget clipTarget( int i ) const {
    return (vl::Vivid::ESmartTarget)actor()->effect()->shader()->getUniform(VL_SMARTCLIP("target"))->getUniformI();
  }

  /**
   * The fuzzyness of the clipping in pixels.
   */
  void setClipFadeRange( int i, float fadeRange ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("fadeRange"))->setUniformF( fadeRange );
  }
  float clipFadeRange( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("fadeRange"))->getUniformF();
  }

  /**
   * The color to use when target == color.
   */
  void setClipColor( int i, const vl::vec4& color ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("color"))->setUniform( color );
  }
  vl::vec4 clipColor( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("color"))->getUniform4F();
  }

  /**
   * The plane equation used for clipping when clipping mode == plane (world coords).
   */
  void setClipPlane( int i, const vl::vec4& plane ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("plane"))->setUniform( plane );
  }
  vl::vec4 clipPlane( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("plane"))->getUniform4F();
  }

  /**
   * The sphere equation used for clipping when clipping mode == sphere (world coords).
   */
  void setClipSphere( int i, const vl::vec4& sphere ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("sphere"))->setUniform( sphere );
  }
  vl::vec4 clipSphere( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("sphere"))->getUniform4F();
  }

  /**
   * The min corner of the box used for clipping when clipping mode == box (world coords).
   */
  void setClipBoxMin( int i, const vl::vec3& boxMin ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("boxMin"))->setUniform( boxMin );
  }
  vl::vec3 clipBoxMin( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("boxMin"))->getUniform3F();
  }

  /**
   * The max corner of the box used for clipping when clipping mode == box (world coords).
   */
  void setClipBoxMax( int i, const vl::vec3& boxMax ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("boxMax"))->setUniform( boxMax );
  }
  vl::vec3 clipBoxMax( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("boxMax"))->getUniform3F();
  }

  /**
   * Reverse the clipping effect "inside-out".
   */
  void setClipReverse( int i, bool reverse ) {
    actor()->effect()->shader()->getUniform(VL_SMARTCLIP("reverse"))->setUniformI( reverse );
  }
  bool clipReverse( int i ) const {
    return actor()->effect()->shader()->getUniform(VL_SMARTCLIP("reverse"))->getUniformI();
  }

  #undef VL_SMARTCLIP

protected:
  // Initialize an Actor to be used with the Vivid renderer
  vl::ref<vl::Actor> initActor(vl::Geometry* geom, vl::Effect* fx = NULL, vl::Transform* tr = NULL);

protected:
  vl::OpenGLContext* m_OpenGLContext;
  vl::VividRendering* m_VividRendering;
  mitk::DataStorage* m_DataStorage;
  VLSceneView* m_VLSceneView;
  const mitk::DataNode* m_DataNode;
  vl::ref<vl::Actor> m_Actor;
  bool m_DataNodeVividUpdateEnabled;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperVLGlobalSettings: public VLMapper
{
public:
  VLMapperVLGlobalSettings( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

  virtual void updateVLGlobalSettings();
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperSurface: public VLMapper
{
public:
  VLMapperSurface( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

protected:
  const mitk::Surface* m_MitkSurf;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapper2DImage: public VLMapper
{
public:
  VLMapper2DImage( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

  //! This vertex array contains 4 points representing the plane
  vl::ArrayFloat3* vertexArray() { return m_VertexArray.get(); }
  const vl::ArrayFloat3* vertexArray() const { return m_VertexArray.get(); }

  //! This texture coordinates array contains 4 3D texture coordinates one for each plane corner
  vl::ArrayFloat3* texCoordarray() { return m_TexCoordArray.get(); }
  const vl::ArrayFloat3* texCoordarray() const { return m_TexCoordArray.get(); }

protected:
  mitk::Image* m_MitkImage;
  vl::ref<vl::ArrayFloat3> m_VertexArray;
  vl::ref<vl::ArrayFloat3> m_TexCoordArray;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapper3DImage: public VLMapper {
public:
  VLMapper3DImage( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

protected:
  const mitk::Image* m_MitkImage;
  vl::ref<vl::VividVolume> m_VividVolume;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperCoordinateAxes: public VLMapper {
public:
  VLMapperCoordinateAxes( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

protected:
  const CoordinateAxesData* m_MitkAxes;
  vl::ref<vl::ArrayFloat3> m_Vertices;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperPoints: public VLMapper {
public:
  VLMapperPoints( const mitk::DataNode* node, VLSceneView* sv );

  virtual void updatePoints( const vl::vec4& color ) = 0 ;

  void initPointSetProps();

  virtual bool init();

  virtual void update();

  void remove();

protected:
  void init3D();
  void init2D();

protected:
  bool m_3DSphereMode;
  vl::ref<vl::ActorTree> m_SphereActors;
  vl::ref<vl::Geometry> m_3DSphereGeom;
  vl::ref<vl::Effect> m_Point2DFX;
  vl::ref<vl::Geometry> m_2DGeometry;
  vl::ref<vl::ArrayFloat3> m_PositionArray;
  vl::ref<vl::ArrayFloat4> m_ColorArray;
  vl::ref<vl::DrawArrays> m_DrawPoints;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperPointSet: public VLMapperPoints
{
public:
  VLMapperPointSet( const mitk::DataNode* node, VLSceneView* sv );

  virtual void updatePoints( const vl::vec4& color );

protected:
  const mitk::PointSet* m_MitkPointSet;
};

//-----------------------------------------------------------------------------

#ifdef _USE_PCL

class NIFTKVL_EXPORT VLMapperPCL: public VLMapperPoints
{
public:
  VLMapperPCL( const mitk::DataNode* node, VLSceneView* sv );

  virtual void updatePoints( const vl::vec4& /*color*/ );

protected:
  const niftk::PCLData* m_NiftkPCL;
};

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA

class NIFTKVL_EXPORT VLMapperCUDAImage: public VLMapper
{
public:
  VLMapperCUDAImage( const mitk::DataNode* node, VLSceneView* sv );

  niftk::LightweightCUDAImage getLWCI();

  virtual bool init();

  virtual void update();

  virtual void remove();

protected:
    cudaGraphicsResource_t m_CudaResource;
    vl::ref<vl::Texture> m_Texture;
};

#endif

}

#endif
