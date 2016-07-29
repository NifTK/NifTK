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

#include <vlCore/Vector3.hpp>
#include <vlCore/Vector4.hpp>
#include <vlCore/vlnamespace.hpp>
#include <vlVivid/VividRendering.hpp>

#include <mitkDataNode.h>
#include <mitkBaseData.h>

namespace mitk
{
  class DataStorage;
  class PointSet;
  class CoordinateAxesData;
  class DataStorage;
  class Surface;
}

namespace niftk
{

class VLSceneView;

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

// VLMapper
// - makeCurrent(): when creating, updating and deleting? Or should we do it externally and remove m_OpenGLContext

/**
 * Takes care of managing all VL related aspects with regard to a given mitk::DataNode, ie, maps a mitk::DataNode to VL/Vivid.
 */
class NIFTKVL_EXPORT VLMapper : public vl::Object {
public:
  VLMapper(const mitk::DataNode* node, VLSceneView* sv);

  virtual ~VLMapper() {
    remove();
  }

  /** Initializes all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual bool init() = 0;

  /** Updates all the relevant VL data structures, uniforms etc. according to the node's settings. */
  virtual void update() = 0;

  /** Removes all the relevant Actor(s) from the scene. */
  virtual void remove() {
    m_VividRendering->sceneManager()->tree()->eraseActor(m_Actor.get());
    m_Actor = 0;
  }

  /** Factory method: creates the right VLMapper subclass according to the node's type. */
  static vl::ref<VLMapper> create(const mitk::DataNode* node, VLSceneView*);

  /** Updates visibility, opacity, color, etc. and Vivid related common settings. */
  void updateCommon();

  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  vl::Actor* actor() { return m_Actor.get(); }
  /** Returns the vl::Actor associated with this VLMapper. Note: the specific subclass might handle more than one vl::Actor. */
  const vl::Actor* actor() const { return m_Actor.get(); }

  //--------------------------------------------------------------------------------

  /** When enabled (default) the mapper will reflect updates to the VL.* variables coming from the DataNode.
      This is useful when you want one object to have the same VL settings across different views/qwidgets.
      Disable this when you want one object to have different settings across different views/qwidgets and
      ignore the VL.* properties of the DataNode. Updates to the "visible" property are also ignored.
      This only applies to VLMapperSurface, VLMapper2DImage, VLMapperCUDAImage for now. */
  void setDataNodeVividUpdateEnabled( bool enable ) { m_DataNodeVividUpdateEnabled = enable; }
  bool isDataNodeVividUpdateEnabled() const { return m_DataNodeVividUpdateEnabled; }

  //--------------------------------------------------------------------------------
  // User managed Vivid API to be used when isDataNodeVividUpdateEnabled() == false
  //--------------------------------------------------------------------------------

  // Only applies to VLMapperSurface, VLMapper2DImage, VLMapperCUDAImage for now.

  // Rendering Mode
  // Whether a surface is rendered with polygons, 3D outline, 2D outline or an outline-slice through a plane.
  // 3D outlines & slice mode:
  //  - computed on the GPU by a geometry shader
  //  - are clipped by the stencil
  //  - interact with the depth buffer (ie they're visible only if they're in front of other objects)
  //  - include creases inside the silhouette regardless of whether they're facing or not the viewer
  // 2D outlines:
  //  - computed using offscreen image based edge detection
  //  - are not clipped against the stencil
  //  - do not interact with depth buffer (ie they're always in front of any geometry)
  //  - look cleaner, renders only the external silhouette of an object

  void setRenderingMode(vl::Vivid::ERenderingMode mode) {
    actor()->effect()->shader()->getUniform("vl_Vivid.renderMode")->setUniformI( mode );
  }
  vl::Vivid::ERenderingMode renderingMode() const {
    return (vl::Vivid::ERenderingMode)actor()->effect()->shader()->getUniform("vl_Vivid.renderMode")->getUniformI();
  }

  // Outline
  // Properties of both the 2D and 3D outlines

  void setOutlineColor(const vl::vec4& color ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.color")->setUniform( color );
  }
  vl::vec4 outlineColor() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.color")->getUniform4F();
  }

  void setOutlineWidth( float width ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.width")->setUniformF( width );
  }
  float outlineWidth() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.width")->getUniformF();
  }

  void setOutlineSlicePlane( const vl::vec4& plane ) {
    actor()->effect()->shader()->getUniform("vl_Vivid.outline.slicePlane")->setUniform( plane );
  }
  vl::vec4 outlineSlicePlane() const {
    return actor()->effect()->shader()->getUniform("vl_Vivid.outline.slicePlane")->getUniform4F();
  }

  // Stencil

  /** Use this Actor as stencil (used when m_VividRendering->setStencilEnabled(true)). */
  void setIsStencil( bool is_stencil ) {
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

  // Material & Opacity
  // Simplified standard OpenGL material properties only difference is they're rendered using high quality per-pixel lighting.

  void setMaterialDiffuseRGBA(const vl::vec4& rgba) {
    actor()->effect()->shader()->getMaterial()->setFrontDiffuse( rgba );
  }
  const vl::vec4& materialDiffuseRGBA() const {
    return actor()->effect()->shader()->getMaterial()->frontDiffuse();
  }

  void setMaterialSpecularColor(const vl::vec4& color ) {
    actor()->effect()->shader()->getMaterial()->setFrontSpecular( color );
  }
  const vl::vec4& materialSpecularColor() const {
    return actor()->effect()->shader()->getMaterial()->frontSpecular();
  }

  void setMaterialSpecularShininess( float shininess ) {
    actor()->effect()->shader()->getMaterial()->setFrontShininess( shininess );
  }
  float materialSpecularShininess() const {
    return actor()->effect()->shader()->getMaterial()->frontShininess();
  }

  // Smart Fog
  // Fog behaves as in standard OpenGL (see red book for settings) except that instead of just targeting the color
  // we can target also alpha and saturation.

  void setFogMode( vl::Vivid::EFogMode mode ) {
    actor()->effect()->shader()->gocUniform("vl_Vivid.smartFog.mode")->setUniformI( mode );
  }
  vl::Vivid::EFogMode fogMode() const {
    return (vl::Vivid::EFogMode)actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.mode")->getUniformI();
  }

  void setFogTarget( vl::Vivid::ESmartTarget target ) {
    actor()->effect()->shader()->gocUniform("vl_Vivid.smartFog.target")->setUniformI( target );
  }
  vl::Vivid::ESmartTarget fogTarget() const {
    return (vl::Vivid::ESmartTarget)actor()->effect()->shader()->getUniform("vl_Vivid.smartFog.target")->getUniformI();
  }

  void setFogColor( const vl::vec4& color ) {
    actor()->effect()->shader()->gocFog()->setColor( color );
  }
  const vl::vec4& fogColor() const {
    return actor()->effect()->shader()->getFog()->color();
  }

  void setFogStart( float start ) {
    actor()->effect()->shader()->gocFog()->setStart( start );
  }
  float fogStart() const {
    return actor()->effect()->shader()->getFog()->start();
  }

  void setFogEnd( float end ) {
    actor()->effect()->shader()->gocFog()->setEnd( end );
  }
  float fogEnd() const {
    return actor()->effect()->shader()->getFog()->end();
  }

  void setFogDensity( float density ) {
    actor()->effect()->shader()->gocFog()->setDensity( density );
  }
  float fogDensity() const {
    return actor()->effect()->shader()->getFog()->density();
  }

  // Smart Clipping
  // We can have up to 4 "clipping units" active: see `i` parameter.
  // We can target color, alpha and saturation -> setClipTarget()
  // We can have various clipping modes: plane, sphere, box -> setClipMode()
  // We can have soft clipping -> setClipFadeRange()
  // We can reverse the clipping effect -> setClipReverse() - by default the negative/outside space is "clipped"

  #define SMARTCLIP(var) (std::string("vl_Vivid.smartClip[") + (char)('0' + i) + "]." + var).c_str()

  void setClipMode( int i, vl::Vivid::EClipMode mode ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("mode"))->setUniformI( mode );
  }
  vl::Vivid::EClipMode clipMode( int i ) const {
    return (vl::Vivid::EClipMode)actor()->effect()->shader()->getUniform(SMARTCLIP("mode"))->getUniformI();
  }

  void setClipTarget( int i, vl::Vivid::ESmartTarget target ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("target"))->setUniformI( target );
  }
  vl::Vivid::ESmartTarget clipTarget( int i ) const {
    return (vl::Vivid::ESmartTarget)actor()->effect()->shader()->getUniform(SMARTCLIP("target"))->getUniformI();
  }

  void setClipFadeRange( int i, float fadeRange ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("fadeRange"))->setUniformF( fadeRange );
  }
  float clipFadeRange( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("fadeRange"))->getUniformF();
  }

  void setClipColor( int i, const vl::vec4& color ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("color"))->setUniform( color );
  }
  vl::vec4 clipColor( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("color"))->getUniform4F();
  }

  void setClipPlane( int i, const vl::vec4& plane ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("plane"))->setUniform( plane );
  }
  vl::vec4 clipPlane( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("plane"))->getUniform4F();
  }

  void setClipSphere( int i, const vl::vec4& sphere ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("sphere"))->setUniform( sphere );
  }
  vl::vec4 clipSphere( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("sphere"))->getUniform4F();
  }

  void setClipBoxMin( int i, const vl::vec3& boxMin ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("boxMin"))->setUniform( boxMin );
  }
  vl::vec3 clipBoxMin( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("boxMin"))->getUniform3F();
  }

  void setClipBoxMax( int i, const vl::vec3& boxMax ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("boxMax"))->setUniform( boxMax );
  }
  vl::vec3 clipBoxMax( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("boxMax"))->getUniform3F();
  }

  void setClipReverse( int i, bool reverse ) {
    actor()->effect()->shader()->gocUniform(SMARTCLIP("reverse"))->setUniformI( reverse );
  }
  bool clipReverse( int i ) const {
    return actor()->effect()->shader()->getUniform(SMARTCLIP("reverse"))->getUniformI();
  }

  #undef SMARTCLIP

  // Texturing

  void setTexture( vl::Texture* tex ) {
    actor()->effect()->shader()->gocTextureSampler( vl::Vivid::UserTexture )->setTexture( tex );
  }
  vl::Texture* texture() {
    return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  }
  const vl::Texture* texture() const {
    return actor()->effect()->shader()->getTextureSampler( vl::Vivid::UserTexture )->texture();
  }

  void setTextureMappingEnabled( bool enable ) {
    actor()->effect()->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( enable );
  }
  bool isTextureMappingEnabled() const {
    return actor()->effect()->shader()->getUniform( "vl_Vivid.enableTextureMapping" )->getUniformI();
  }

  // NOTE: point sprites require texture mapping to be enabled as well.
  // See also pointSize().
  void setPointSpriteEnabled( bool enable ) {
    actor()->effect()->shader()->gocUniform( "vl_Vivid.enablePointSprite" )->setUniformI( enable );
  }
  bool isPointSpriteEnabled() const {
    return actor()->effect()->shader()->getUniform( "vl_Vivid.enablePointSprite" )->getUniformI();
  }

  // Other Vivid supported render states

  vl::PointSize* pointSize() { return actor()->effect()->shader()->getPointSize(); }
  const vl::PointSize* pointSize() const { return actor()->effect()->shader()->getPointSize(); }

  // Useful to render surfaces in wireframe
  vl::PolygonMode* polygonMode() { return actor()->effect()->shader()->getPolygonMode(); }
  const vl::PolygonMode* polygonMode() const { return actor()->effect()->shader()->getPolygonMode(); }

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

class NIFTKVL_EXPORT VLMapperVLGlobalSettings: public VLMapper {
public:
  VLMapperVLGlobalSettings( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

  virtual void updateVLGlobalSettings();
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapperSurface: public VLMapper {
public:
  VLMapperSurface( const mitk::DataNode* node, VLSceneView* sv );

  virtual bool init();

  virtual void update();

protected:
  const mitk::Surface* m_MitkSurf;
};

//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLMapper2DImage: public VLMapper {
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
  const mitk::CoordinateAxesData* m_MitkAxes;
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

class NIFTKVL_EXPORT VLMapperPCL: public VLMapperPoints {
public:
  VLMapperPCL( const mitk::DataNode* node, VLSceneView* sv );

  virtual void updatePoints( const vl::vec4& /*color*/ );

protected:
  const niftk::PCLData* m_NiftkPCL;
};

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA

class NIFTKVL_EXPORT VLMapperCUDAImage: public VLMapper {
public:
  VLMapperCUDAImage( const mitk::DataNode* node, VLSceneView* sv );

  niftk::LightweightCUDAImage getLWCI();

  virtual bool init();

  virtual void update();

  virtual void remove();

protected:
    cudaGraphicsResource_t m_CudaResource;
    vl::ref<Texture> m_Texture;
};

#endif

}

#endif

