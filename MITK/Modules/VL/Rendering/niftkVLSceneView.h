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
#include <mitkOclResourceService.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkDataNodePropertyListener.h>

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

//-----------------------------------------------------------------------------
// VLSceneView
//-----------------------------------------------------------------------------

class NIFTKVL_EXPORT VLSceneView : public vl::UIEventListener
{
public:
  typedef std::map< mitk::DataNode::ConstPointer, vl::ref<VLMapper> > DataNodeVLMapperMapType;

public:
  VLSceneView( VLWidget* vlwidget );
  ~VLSceneView();

  void setDataStorage(mitk::DataStorage* ds);

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

  niftk::VLTrackballManipulator* trackball() { return m_Trackball.get(); }
  const niftk::VLTrackballManipulator* trackball() const { return m_Trackball.get(); }

  vl::CalibratedCamera* camera() { return m_Camera.get(); }
  const vl::CalibratedCamera* camera() const { return m_Camera.get(); }

  // Obsolete: called by VLRendererView, QmitkIGIVLEditor (via IGIVLEditor)
  void setOclResourceService(OclResourceService* oclserv);

protected:
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

  // Update camera position, projection and viewport
  void updateCameraParameters();

  VLMapper* getVLMapper(const mitk::DataNode* node);

protected:
  // Used by niftk::ScopedOGLContext
  VLWidget* m_VLWidget;

  vl::ref<vl::VividRendering>        m_VividRendering;
  vl::ref<vl::VividRenderer>         m_VividRenderer;
  vl::ref<vl::SceneManagerActorTree> m_SceneManager;
  vl::ref<vl::CalibratedCamera>      m_Camera;
  vl::ref<niftk::VLTrackballManipulator>    m_Trackball;

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

}

#endif

