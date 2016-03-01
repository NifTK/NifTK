/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SharedOGLContext_h
#define SharedOGLContext_h

#include <niftkCoreGuiExports.h>
#include <QtOpenGL/QGLWidget>
#include <QMutex>

// forward-decl
class SharedOGLContext;


class NIFTKCOREGUI_EXPORT SharedOGLContext
{

public:
  static QGLWidget* GetShareWidget();


protected:
  SharedOGLContext();
  ~SharedOGLContext();


private:
  QGLWidget*                  m_ShareWidget;

  static SharedOGLContext*    s_Instance;
  static QMutex               s_Lock;
};

#endif // SharedOGLContext_h
