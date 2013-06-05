/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "vtkOpenGLMatrixDrivenCamera.h"

#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLRenderer.h>
#include <vtkOutputWindow.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkgluPickMatrix.h>
#include <vtkOpenGL.h>
#include <math.h>

#ifndef VTK_IMPLEMENT_MESA_CXX
vtkStandardNewMacro(vtkOpenGLMatrixDrivenCamera);
#endif

//----------------------------------------------------------------------------
vtkOpenGLMatrixDrivenCamera::vtkOpenGLMatrixDrivenCamera()
: m_IntrinsicMatrix(NULL)
, DefaultBehaviour(true)
, m_ImageWidth(256)
, m_ImageHeight(256)
, m_WindowWidth(256)
, m_WindowHeight(256)
{
  m_IntrinsicMatrix = vtkMatrix4x4::New();
  m_IntrinsicMatrix->Identity();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetCalibratedImageSize(const int& width, const int& height)
{
  m_ImageWidth = width;
  m_ImageHeight = height;
  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetActualWindowSize(const int& width, const int& height)
{
  m_WindowWidth = width;
  m_WindowHeight = height;
  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetIntrinsicParameters(const double& fx, const double& fy,
                                                         const double &cx, const double& cy
                                                        )
{

  double clippingRange[2];
  this->GetClippingRange(clippingRange);
  double near = clippingRange[0];
  double far = clippingRange[1];

  // Inspired by: http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
  /*
   [2*K00/width, -2*K01/width,    (width - 2*K02 + 2*x0)/width,                            0]
   [          0, 2*K11/height, (-height + 2*K12 + 2*y0)/height,                            0]
   [          0,            0,  (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)]
   [          0,            0,                              -1,                            0]
   */

  m_IntrinsicMatrix->Zero();
  m_IntrinsicMatrix->SetElement(0, 0, 2*fx/m_ImageWidth);
  m_IntrinsicMatrix->SetElement(0, 1, -2*0/m_ImageWidth);
  m_IntrinsicMatrix->SetElement(0, 2, (m_ImageWidth - 2*cx)/m_ImageWidth);
  m_IntrinsicMatrix->SetElement(1, 1, 2*fy/m_ImageHeight);
  m_IntrinsicMatrix->SetElement(1, 2, (-m_ImageHeight + 2*cy)/m_ImageHeight);
  m_IntrinsicMatrix->SetElement(2, 2, (-far-near)/(far-near));
  m_IntrinsicMatrix->SetElement(2, 3, -2*far*near/(far-near));
  m_IntrinsicMatrix->SetElement(3, 2, -1);

  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::Render(vtkRenderer *ren)
{
  // By default we do behaviour in base class.
  if (this->DefaultBehaviour)
  {
    vtkOpenGLCamera::Render(ren);
    return;
  }

  vtkMatrix4x4 *matrix = vtkMatrix4x4::New();
  vtkOpenGLRenderWindow *win=vtkOpenGLRenderWindow::SafeDownCast(ren->GetRenderWindow());

  if (ren->GetRenderWindow()->GetDoubleBuffer())
  {
    glDrawBuffer(static_cast<GLenum>(win->GetBackBuffer()));
    glReadBuffer(static_cast<GLenum>(win->GetBackBuffer()));
  }
  else
  {
    glDrawBuffer(static_cast<GLenum>(win->GetFrontBuffer()));
    glReadBuffer(static_cast<GLenum>(win->GetFrontBuffer()));
  }

  double widthScale  = (double) m_WindowWidth  / (double) m_ImageWidth;
  double heightScale  = (double) m_WindowHeight  / (double) m_ImageHeight;

  int vpw = m_WindowWidth;
  int vph = m_WindowHeight;
  if (widthScale < heightScale)
  {
    vph = (int) ((double) m_ImageHeight * widthScale);
  }
  else
  {
    vpw = (int) ((double) m_ImageWidth * heightScale);
  }

  int vpx = m_WindowWidth  / 2 - vpw / 2;
  int vpy = m_WindowHeight / 2 - vph / 2;

  glViewport(vpx, vpy, vpw, vph);
  glEnable( GL_SCISSOR_TEST );
  glScissor(vpx,vpy, vpw, vph);

  glMatrixMode( GL_PROJECTION);

  matrix->DeepCopy(m_IntrinsicMatrix);
  matrix->Transpose();

  glLoadMatrixd(matrix->Element[0]);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();

  matrix->DeepCopy(this->GetViewTransformMatrix());
  matrix->Transpose();

  glMultMatrixd(matrix->Element[0]);

  if ((ren->GetRenderWindow())->GetErase() && ren->GetErase() && !ren->GetIsPicking())
  {
    ren->Clear();
  }
  matrix->Delete();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "vtkOpenGLMatrixDrivenCamera:DefaultBehaviour=" << DefaultBehaviour << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_ImageWidth=" << m_ImageWidth << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_ImageHeight=" << m_ImageHeight << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_WindowWidth=" << m_WindowWidth << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_WindowHeight=" << m_WindowHeight << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera::m_IntrinsicMatrix" << std::endl;
  m_IntrinsicMatrix->PrintSelf(os, indent);
}
