/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VLQtWidget_INCLUDE_ONCE
#define VLQtWidget_INCLUDE_ONCE

#include <niftkVLExports.h>

#include <QtGlobal>
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
#include <vlQt5/Qt5Widget.hpp>
#else
#include <vlQt4/Qt4Widget.hpp>
#endif

#include <vlGraphics/OpenGLContext.hpp>
#include <vlVivid/VividRenderer.hpp>
#include <vlVivid/VividRendering.hpp>
#include "TrackballManipulator.h"
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
#include <niftkCUDAImage.h>
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

class VLSceneView;

#ifdef _USE_CUDA

  class CudaTest;

#endif

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

// VLMapper
// - makeCurrent(): when creating, updating and deleting? Or should we do it externally and remove m_OpenGLContext

/**
 * Takes care of managing all VL related aspects with regard to a given mitk::DataNode, ie, maps a mitk::DataNode to VL/Vivid.
 */
class VLMapper : public vl::Object {
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
    m_Actor = NULL;
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
  bool setDataNodeVividUpdateEnabled( bool enable ) { m_DataNodeVividUpdateEnabled = enable; }
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
// VLSceneView
//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLSceneView : public vl::UIEventListener
{
public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLMapper> > DataNodeVLMapperMapType;

public:
  VLSceneView( QGLWidget* qglwidget );
  ~VLSceneView();

  void setDataStorage(const mitk::DataStorage::Pointer& dataStorage);

  bool setCameraTrackingNode(const mitk::DataNode* node);

  void setEyeHandFileName(const std::string& fileName);

  bool setBackgroundNode(const mitk::DataNode* node);

  void setBackgroundColour(float r, float g, float b);

  // Defines the opacity of the 3D renering above the background.
  void setOpacity( float opacity );

  // Number of depth peeling passes to be done.
  void setDepthPeelingPasses( int passes );

  // Positions the camera for optimal visibility of currently selected DataNode
  void reInit(const vl::vec3& dir = vl::vec3(0,0,1), const vl::vec3& up = vl::vec3(0,1,0), float bias=1.0f);
  // Positions the camera for optimal scene visibility
  void globalReInit(const vl::vec3& dir = vl::vec3(0,0,1), const vl::vec3& up = vl::vec3(0,1,0), float bias=1.0f);

  void scheduleTrackballAdjustView(bool schedule = true);
  void scheduleNodeAdd(const mitk::DataNode* node);
  void scheduleNodeRemove(const mitk::DataNode* node);
  void scheduleNodeUpdate(const mitk::DataNode* node);
  void scheduleSceneRebuild();

  mitk::DataStorage* dataStorage() { return m_DataStorage.GetPointer(); }
  const mitk::DataStorage* dataStorage() const { return m_DataStorage.GetPointer(); }

  vl::VividRendering* vividRendering() { return m_VividRendering.get(); }
  const vl::VividRendering* vividRendering() const { return m_VividRendering.get(); }

  VLTrackballManipulator* trackball() { return m_Trackball.get(); }
  const VLTrackballManipulator* trackball() const { return m_Trackball.get(); }

  vl::CalibratedCamera* camera() { return m_Camera.get(); }
  const vl::CalibratedCamera* camera() const { return m_Camera.get(); }

  // Obsolete: called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void setOclResourceService(OclResourceService* oclserv);

protected:
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
  bool contextIsCurrent() { return openglContext() && QGLContext::currentContext() == openglContext()->as<vlQt5::Qt5Widget>()->QGLWidget::context(); }
#else
  bool contextIsCurrent() { return openglContext() && QGLContext::currentContext() == openglContext()->as<vlQt4::Qt4Widget>()->QGLWidget::context(); }
#endif

  void initSceneFromDataStorage();
  void clearScene();
  void updateScene();
  void renderScene();

  // Returned VLMapper can be NULL
  VLMapper* addDataNode(const mitk::DataNode* node);
  void removeDataNode(const mitk::DataNode* node);
  void updateDataNode(const mitk::DataNode* node);

  virtual void addDataStorageListeners();
  virtual void removeDataStorageListeners();

  void updateCameraParameters();

  VLMapper* getVLMapper(const mitk::DataNode* node);

protected:
  // Used by niftk::ScopedOGLContext
  QGLWidget* m_QGLWidget;

  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::SceneManagerActorTree> m_SceneManager;
  vl::ref<vl::CalibratedCamera>      m_Camera;
  vl::ref<VLTrackballManipulator>    m_Trackball;

  mitk::DataStorage::Pointer              m_DataStorage;
  mitk::DataNodePropertyListener::Pointer m_NodeVisibilityListener;
  mitk::DataNodePropertyListener::Pointer m_NodeColorPropertyListener;
  mitk::DataNodePropertyListener::Pointer m_NodeOpacityPropertyListener;

  DataNodeVLMapperMapType                m_DataNodeVLMapperMap;
  std::set<mitk::DataNode::ConstPointer> m_NodesToUpdate;
  std::set<mitk::DataNode::ConstPointer> m_NodesToAdd;
  std::set<mitk::DataNode::ConstPointer> m_NodesToRemove;
  mitk::DataNode::ConstPointer           m_CameraNode;

  mitk::DataNode::ConstPointer m_BackgroundNode;
  mitk::Image::ConstPointer m_BackgroundImage;
#ifdef _USE_CUDA
  niftk::CUDAImage::ConstPointer m_BackgroundCUDAImage;
#endif
  vl::mat4 m_EyeHandMatrix;

  bool m_ScheduleTrackballAdjustView;
  bool m_ScheduleInitScene;
  bool m_RenderingInProgressGuard;

  // Lgacy OpenCL service

  OclResourceService* m_OclService;

  // CUDA support

#ifdef _USE_CUDA
protected:
  CudaTest* m_CudaTest;
#endif

protected:
  // --------------------------------------------------------------------------
  // vl::UIEventListener implementation
  // --------------------------------------------------------------------------

  virtual void initEvent();
  virtual void resizeEvent(int width, int height);
  virtual void updateEvent();
  virtual void destroyEvent();

  virtual void addedListenerEvent(vl::OpenGLContext *) { }
  virtual void removedListenerEvent(vl::OpenGLContext *) { }
  virtual void enableEvent(bool) { }
  virtual void visibilityEvent(bool) { }
  virtual void mouseMoveEvent(int, int) { }
  virtual void mouseUpEvent(vl::EMouseButton, int, int) { }
  virtual void mouseDownEvent(vl::EMouseButton, int, int) { }
  virtual void mouseWheelEvent(int) { }
  virtual void keyPressEvent(unsigned short, vl::EKey) { }
  virtual void keyReleaseEvent(unsigned short, vl::EKey) { }
  virtual void fileDroppedEvent(const std::vector<vl::String>&) { }
};

//-----------------------------------------------------------------------------
// VLQtWidget
//-----------------------------------------------------------------------------
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
class VLQtWidget : public vlQt5::Qt5Widget {
public:
  VLQtWidget(QWidget* parent = NULL, const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0)
    : Qt5Widget(parent, shareWidget, f) {
#else
class VLQtWidget : public vlQt4::Qt4Widget {
public:
  VLQtWidget(QWidget* parent = NULL, const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0)
    : Qt4Widget(parent, shareWidget, f) {
#endif
    m_VLSceneView = new VLSceneView( this );
    addEventListener(m_VLSceneView.get());
    setRefreshRate(1000 / 30); // 30 fps in milliseconds
    setContinuousUpdate(false);
    setMouseTracking(true);
    setAutoBufferSwap(false);
    setAcceptDrops(false);

    // Explicitly request OpenGL 3.2 Compatibility profile.
    QGLContext* glctx = new QGLContext(this->context()->format(), this);
    QGLFormat fmt = this->context()->format();
    fmt.setDoubleBuffer( true );
    #if QT_VERSION >= 0x040700
      fmt.setProfile(QGLFormat::CompatibilityProfile);
      fmt.setVersion(3, 2);
    #endif
    glctx->setFormat(fmt);
    glctx->create(NULL);
    this->setContext(glctx);
    makeCurrent();
    MITK_INFO << "VLQtWidget: created OpenGL context version: " << glGetString(GL_VERSION) << "\n";
  }

  void setVLSceneView(VLSceneView* vl_view) { m_VLSceneView = vl_view; }
  VLSceneView* vlSceneView() { return m_VLSceneView.get(); }
  const VLSceneView* vlSceneView() const { return m_VLSceneView.get(); }

protected:
  vl::ref<VLSceneView> m_VLSceneView;
};

#endif
