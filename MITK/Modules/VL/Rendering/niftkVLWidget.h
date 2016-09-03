/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLWidget_h
#define niftkVLWidget_h

#include <QtGlobal>
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
#include <vlQt5/Qt5Widget.hpp>
#else
#include <vlQt4/Qt4Widget.hpp>
#endif

#include <niftkVLExports.h>
#include <niftkVLMapper.h>
#include <niftkVLSceneView.h>

namespace niftk
{

class VLSceneView;

/**
 * \brief A QGLWidget containing a niftk::VLSceneView.
 *
 * Usually you only have to create a VLWidget and use vlSceneView() to set the data store and various Vivid options.
 *
 * Under niftk::VLWidget there are a few layers that implement the OpenGL rendering in decreasing order of abstraction:
 * 1) niftk::VLWidget
 * 2) niftk::VLSceneView
 * 3) niftk::VLMapper
 * 4) vl::VividRendering
 * 5) Visualization Library
 * 6) OpenGL
 *
 * \sa vlQt5::Qt5Widget
 *
 */
#if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
class NIFTKVL_EXPORT VLWidget : public vlQt5::Qt5Widget {
public:
  VLWidget(QWidget* parent = NULL, const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0)
    : Qt5Widget(parent, shareWidget, f) {
#else
class NIFTKVL_EXPORT VLWidget : public vlQt4::Qt4Widget {
public:
  VLWidget(QWidget* parent = NULL, const QGLWidget* shareWidget = NULL, Qt::WindowFlags f = 0)
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
    MITK_INFO << "niftk::VLWidget: created OpenGL context version: " << glGetString(GL_VERSION) << "\n";
  }

  VLSceneView* vlSceneView() { return m_VLSceneView.get(); }

  const VLSceneView* vlSceneView() const { return m_VLSceneView.get(); }

  bool contextIsCurrent() { return QGLContext::currentContext() == QGLWidget::context(); }

protected:
  vl::ref<VLSceneView> m_VLSceneView;
};

}

#endif
