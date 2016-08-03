/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLSceneView_h
#define niftkVLSceneView_h

#include <niftkVLExports.h>
#include <niftkVLMapper.h>
#include "niftkVLTrackballManipulator.h"
#include <vlGraphics/UIEventListener.hpp>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkImage.h>

#include <niftkDataNodePropertyListener.h>

namespace mitk
{
  class PointSet;
  class CoordinateAxesData;
  class DataStorage;
  class Surface;
}

namespace niftk
{

class VLWidget;
#ifdef _USE_CUDA
  class CudaTest;
#endif

/**
 * \brief A vl::UIEventListener bound to a QGLWidget (niftk::VLWidget) managing all VL/Vivid rendering and options
 * and listening to mitk::DataStorage events.
 *
 * "Vivid" is the short name of the rendering engine based on Visualization Library developed exclusively for NifTK
 * (see vl::VividRendering, vl::VividRenderer, vl::VividVolume and relative GLSL shaders).
 *
 * VLSceneView listens for the QGLWidget update, resize, mouse, keyboard events and updates the rendering accordingly
 * also allowing for user interaction via trackball. It also listens for mitk::DataStorage events and updates the
 * scene adding, removing and updating objects according to the state of the mitk::DataStorage.
 *
 * VLSceneView keeps track of every mitk::DataNode added and uses a niftk::VLMapper sub-class to render it and
 * update it's visual aspect. The mapping from mitk::DataNode to niftk::VLMapper is done in the factory method niftk::VLMapper::create().
 *
 * Supported objects types:
 *
 * - mitk::Surface -> niftk::VLMapperSurface: render polygonal objects (no lines, no points, no strips), see niftk::VLUtils::getVLGeometry() for details.
 * - mitk::Image:
 *    2D images -> niftk::VLMapper2DImage: can be used as background.
 *    3D images -> niftk::VLMapper3DImage: rendered as a volume using vl::VividVolume
 * - niftk::CUDAImage -> niftk::VLMapperCUDAImage: can be used as background.
 * - mitk::PointSet -> niftk::VLMapperPointSet: a static set of points of the same size and color drawn as 3D spheres or 2D point sprites.
 * - niftk::PCLData -> niftk::VLMapperPCL: a static set of points of the same size but different color drawn as 3D spheres or 2D point sprites.
 * - CoordinateAxesData -> niftk::VLMapperCoordinateAxes: 3 red, green and blue lines perpedicular to one another representing the X, Y and Z axes.
 * - niftk::VLGlobalSettingsDataNode -> niftk::VLMapperVLGlobalSettings: a utility to change the global settings of all active VLWidgets like stencil settings, rendering mode and depth peeling passes.
 *
 * By default if you instantiate multiple VLWidgets all the DataNodes will be displayed in the same way, ie the various niftk::VLMapper will
 * all track the node's properties. If you'd like to override the aspect of an object in a specific VLWidget you can use
 * niftkVLMapper::setDataNodeVividUpdateEnabled(false) and then proceed to set the various options like niftk::VLMapper::setRenderingMode().
 */
class NIFTKVL_EXPORT VLSceneView : public vl::UIEventListener
{
public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLMapper> > DataNodeVLMapperMapType;

public:
  VLSceneView( VLWidget* vlwidget );
  ~VLSceneView();

  /**
   * Sets the mitk::DataStorage to listen from.
   * The scene will be destroyed and recreated and niftk::VLGlobalSettingsDataNode added if not present already.
   */
  void setDataStorage(mitk::DataStorage* ds);

  /**
   * Sets the transform to be used to move the camera around. Passing NULL will disable camera tracking and enable the trackball.
   */
  bool setCameraTrackingNode(const mitk::DataNode* node);

  /**
   * Loads the file storing the eye-hand calibration matrix.
   */
  void setEyeHandFileName(const std::string& fileName);

  /**
   * Sets a node containing an mitk::Image or niftk::CUDAImage to be used as background. Passing NULL will disable the background image.
   */
  bool setBackgroundNode(const mitk::DataNode* node);

  /**
   * The rendering mode, mainly for debugging purposes.
   */
  void setRenderingMode( vl::Vivid::ERenderingMode );
  vl::Vivid::ERenderingMode renderingMode() const;

  /**
   * The rendering background color.
   */
  void setBackgroundColor(float r, float g, float b);
  vl::vec3 backgroundColor() const;

  /**
   * Whether geometry stencil effect is enabled.
   */
  void setStencilEnabled( bool enabled );
  bool isStencilEnabled() const;

  /**
   * Color of the stencil background.
   */
  void setStencilBackgroundColor( const vl::vec4& color );
  const vl::vec4& stencilBackgroundColor() const;

  /**
   * Smoothness in pixels of the stencil background.
   * Defines the size of the smoothing kernel so the bigger the slower the rendering.
   * See also this issue: https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/issues/4717
   */
  void setStencilSmoothness( float smoothness );
  float stencilSmoothness() const;

  /**
   * The opacity of the 3D rendering above the background. The background is always fully opaque.
   */
  void setOpacity( float opacity );
  float opacity() const;

  /**
   * The number of depth peeling passes to be done. The more passes the more transparency layers will be rendered and the slower the rendering.
   * When vl::VividRendering::depthPeelingAutoThrottleEnabled() is false the rendering will always do N passes, else it will try to do only one
   * pass if all objects are *fully* opaque and no fogging or clipping is using the alpha target.
   */
  void setDepthPeelingPasses( int n );
  int depthPeelingPasses() const;

  /**
   * Positions the camera for optimal visibility of the currently selected DataNode.
   */
  void reInit(const vl::vec3& dir = vl::vec3(0,0,1), const vl::vec3& up = vl::vec3(0,1,0), float bias=1.0f);

  /**
   * Positions the camera for optimal scene visibility.
   */
  void globalReInit(const vl::vec3& dir = vl::vec3(0,0,1), const vl::vec3& up = vl::vec3(0,1,0), float bias=1.0f);

  /**
   * Returns the underlying vl::VividRendering, usually you don't need to touch this.
   */
  vl::VividRendering* vividRendering() { return m_VividRendering.get(); }
  const vl::VividRendering* vividRendering() const { return m_VividRendering.get(); }

  /**
   * Returns the underlying niftk::VLTrackballManipulator, usually you don't need to touch this.
   */
  niftk::VLTrackballManipulator* trackball() { return m_Trackball.get(); }
  const niftk::VLTrackballManipulator* trackball() const { return m_Trackball.get(); }

  /**
   * Returns the underlying vl::CalibratedCamera, usually you don't need to touch this.
   */
  vl::CalibratedCamera* camera() { return m_Camera.get(); }
  const vl::CalibratedCamera* camera() const { return m_Camera.get(); }

  /**
   * Returns the underlying mitk::DataStorage, usually you don't need to touch this.
   */
  mitk::DataStorage* dataStorage() { return m_DataStorage.GetPointer(); }
  const mitk::DataStorage* dataStorage() const { return m_DataStorage.GetPointer(); }

  /**
   * Destroys the current scene and schedules a rebuild at the next rendering.
   */
  void scheduleSceneRebuild();

protected:
  void initSceneFromDataStorage();
  void clearScene();
  void updateScene();
  void renderScene();
  void addDataStorageListeners();
  void removeDataStorageListeners();
  void scheduleTrackballAdjustView(bool schedule = true);

  //! Schedules an addDataNode() at the next rendering
  void scheduleNodeAdd(const mitk::DataNode* node);

  //! Schedules a removeDataNode() at the next rendering
  void scheduleNodeRemove(const mitk::DataNode* node);

  //! Schedules an updateDataNode() at the next rendering
  void scheduleNodeUpdate(const mitk::DataNode* node);

  //! Adds to the scene a VLMapper subclass representing the given DataNode
  void addDataNode(const mitk::DataNode* node);

  //! Removes from the scene the VLMapper subclass representing the given DataNode
  void removeDataNode(const mitk::DataNode* node);

  //! Updates the VLMapper subclass representing the given DataNode
  void updateDataNode(const mitk::DataNode* node);

  //! Updates the camera position, projection and viewport
  void updateCameraParameters();

  //! Returns the VLMapper associated to the given DataNode
  VLMapper* getVLMapper(const mitk::DataNode* node);

protected:
  // Used by niftk::ScopedOGLContext
  VLWidget* m_VLWidget;

  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::SceneManagerActorTree> m_SceneManager;
  vl::ref<vl::CalibratedCamera>      m_Camera;
  vl::ref<niftk::VLTrackballManipulator> m_Trackball;

  mitk::DataStorage::Pointer        m_DataStorage;
  DataNodePropertyListener::Pointer m_NodeVisibilityListener;
  DataNodePropertyListener::Pointer m_NodeColorPropertyListener;
  DataNodePropertyListener::Pointer m_NodeOpacityPropertyListener;

  DataNodeVLMapperMapType                m_DataNodeVLMapperMap;
  std::set<mitk::DataNode::ConstPointer> m_NodesToUpdate;
  std::set<mitk::DataNode::ConstPointer> m_NodesToAdd;
  std::set<mitk::DataNode::ConstPointer> m_NodesToRemove;
  mitk::DataNode::ConstPointer           m_CameraNode;

  mitk::DataNode::ConstPointer m_BackgroundNode;
  mitk::Image::ConstPointer m_BackgroundImage;
  vl::mat4 m_EyeHandMatrix;

  bool m_ScheduleTrackballAdjustView;
  bool m_ScheduleInitScene;
  bool m_RenderingInProgressGuard;

  // CUDA support

#ifdef _USE_CUDA
  niftk::CUDAImage::ConstPointer m_BackgroundCUDAImage;
  CudaTest* m_CudaTest;
#endif

protected:

  // vl::UIEventListener implementation

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

}

#endif

