/**************************************************************************************/
/*                                                                                    */
/*  Visualization Library                                                             */
/*  http://www.visualizationlibrary.org                                               */
/*                                                                                    */
/*  Copyright (c) 2005-2010, Michele Bosi                                             */
/*  All rights reserved.                                                              */
/*                                                                                    */
/*  Redistribution and use in source and binary forms, with or without modification,  */
/*  are permitted provided that the following conditions are met:                     */
/*                                                                                    */
/*  - Redistributions of source code must retain the above copyright notice, this     */
/*  list of conditions and the following disclaimer.                                  */
/*                                                                                    */
/*  - Redistributions in binary form must reproduce the above copyright notice, this  */
/*  list of conditions and the following disclaimer in the documentation and/or       */
/*  other materials provided with the distribution.                                   */
/*                                                                                    */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            */
/*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR  */
/*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    */
/*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      */
/*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON    */
/*  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     */
/*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      */
/*                                                                                    */
/**************************************************************************************/

#ifndef Qt4Window_INCLUDE_ONCE
#define Qt4Window_INCLUDE_ONCE

#include <vlQt4/link_config.hpp>
#include <vlCore/VisualizationLibrary.hpp>
#include <vlGraphics/OpenGLContext.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/Camera.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/RenderingTree.hpp>
#include <vlGraphics/SceneManagerActorTree.hpp>
#include <vlGraphics/TrackballManipulator.hpp>
#include <vlGraphics/Geometry.hpp>
#include <QtGui/QMouseEvent>
#include <QtGui/QWidget>
#include <QtCore/QTimer>
#include <QtCore/QObject>
#include <QtOpenGL/QGLWidget>
//#include <QtOpenGL/QGLFormat>
//#include <QtGui/QApplication>
//#include <QtCore/QUrl>
#include <mitkOclResourceService.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <map>


class VLQt4Widget : public QGLWidget, public vl::OpenGLContext
{
  Q_OBJECT

public:
  using vl::Object::setObjectName;
  using QObject::setObjectName;

  VLQt4Widget(QWidget* parent=NULL, const QGLWidget* shareWidget=NULL, Qt::WindowFlags f=0);

  virtual ~VLQt4Widget();

  //bool initQt4Widget(const vl::String& title/*, const vl::OpenGLContextFormat& info, const QGLContext* shareContext=0*/, int x=0, int y=0, int width=640, int height=480);

  void setRefreshRate(int msec);
  int refreshRate();

  void setOclResourceService(OclResourceService* oclserv);

  void AddDataNode(const mitk::DataNode::Pointer& node);
  void RemoveDataNode(const mitk::DataNode::Pointer& node);
  void UpdateDataNode(const mitk::DataNode::Pointer& node);

  void ClearScene();

  // from vl::OpenGLContext
public:
  virtual void setContinuousUpdate(bool continuous);
  virtual void setWindowTitle(const vl::String& title);
  virtual bool setFullscreen(bool fullscreen);
  virtual void show();
  virtual void hide();
  virtual void setPosition(int x, int y);
  virtual vl::ivec2 position() const;
  virtual void update();                // hides non-virtual QWidget::update()?
  virtual void setSize(int w, int h);
  virtual void swapBuffers();           // in QGLWidget too
  virtual void makeCurrent();           // in QGLWidget too
  virtual void setMousePosition(int x, int y);
  virtual void setMouseVisible(bool visible);
  virtual void getFocus();

  virtual vl::ivec2 size() const;       // BEWARE: not a baseclass method!

protected:
  void translateKeyEvent(QKeyEvent* ev, unsigned short& unicode_out, vl::EKey& key_out);


  // from QGLWidget
protected:
  virtual void initializeGL();
  virtual void resizeGL(int width, int height);
  virtual void paintGL();
  virtual void mouseMoveEvent(QMouseEvent* ev);
  virtual void mousePressEvent(QMouseEvent* ev);
  virtual void mouseReleaseEvent(QMouseEvent* ev);
  virtual void wheelEvent(QWheelEvent* ev);
  virtual void keyPressEvent(QKeyEvent* ev);
  virtual void keyReleaseEvent(QKeyEvent* ev);
  //void dragEnterEvent(QDragEnterEvent *ev);
  //void dropEvent(QDropEvent* ev);


protected:
  void renderScene();

  vl::ref<vl::RenderingTree>            m_RenderingTree;
  vl::ref<vl::Rendering>                m_OpaqueObjectsRendering;
  vl::ref<vl::SceneManagerActorTree>    m_SceneManager;
  vl::ref<vl::Camera>                   m_Camera;
  vl::ref<vl::Light>                    m_Light;
  vl::ref<vl::Transform>                m_LightTr;
  vl::ref<vl::TrackballManipulator>     m_Trackball;

  OclResourceService*                   m_OclService;


  vl::ref<vl::Actor> AddSurfaceActor(const mitk::Surface::Pointer& mitkSurf);
  void ConvertVTKPolyData(vtkPolyData* vtkPoly, vl::ref<vl::Geometry> vlPoly);

  std::map<mitk::DataNode::Pointer, vl::ref<vl::Actor> >    m_NodeToActorMap;
  std::map<vl::ref<vl::Actor>, vl::ref<vl::Renderable> >    m_ActorToRenderableMap;


protected:
  int       m_Refresh;
  QTimer    m_UpdateTimer;
};


#endif
