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
#include <cassert>

#ifndef VTK_IMPLEMENT_MESA_CXX
vtkStandardNewMacro(vtkOpenGLMatrixDrivenCamera);
#endif

//----------------------------------------------------------------------------
vtkOpenGLMatrixDrivenCamera::vtkOpenGLMatrixDrivenCamera()
: m_IntrinsicMatrix(NULL)
, UseCalibratedCamera(false)
, m_ImageWidthInPixels(256)
, m_ImageHeightInPixels(256)
, m_WindowWidthInPixels(256)
, m_WindowHeightInPixels(256)
, m_PixelAspectRatio(1.0)
, m_Fx(1)
, m_Fy(1)
, m_Cx(0)
, m_Cy(0)
{
  m_IntrinsicMatrix = vtkMatrix4x4::New();
  m_IntrinsicMatrix->Identity();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetCalibratedImageSize(const int& width, const int& height, double pixelaspect)
{
  m_ImageWidthInPixels = width;
  m_ImageHeightInPixels = height;
  m_PixelAspectRatio = pixelaspect;
  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetActualWindowSize(const int& width, const int& height)
{
  m_WindowWidthInPixels = width;
  m_WindowHeightInPixels = height;
  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::SetIntrinsicParameters(const double& fx, const double& fy,
                                                         const double &cx, const double& cy
                                                        )
{
  m_Fx = fx;
  m_Fy = fy;
  m_Cx = cx;
  m_Cy = cy;
  this->Modified();
}


//----------------------------------------------------------------------------
void vtkOpenGLMatrixDrivenCamera::Render(vtkRenderer *ren)
{
  // By default we do behaviour in base class.
  if (!this->UseCalibratedCamera)
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

  double clippingRange[2];
  this->GetClippingRange(clippingRange);

  double znear = clippingRange[0];
  double zfar = clippingRange[1];

  // Inspired by: http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
  /*
   [2*K00/width, -2*K01/width,    (width - 2*K02 + 2*x0)/width,                            0]
   [          0, 2*K11/height, (-height + 2*K12 + 2*y0)/height,                            0]
   [          0,            0,  (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)]
   [          0,            0,                              -1,                            0]
   */

  m_IntrinsicMatrix->Zero();
  m_IntrinsicMatrix->SetElement(0, 0, 2*m_Fx/m_ImageWidthInPixels);
  m_IntrinsicMatrix->SetElement(0, 1, -2*0/m_ImageWidthInPixels);
  m_IntrinsicMatrix->SetElement(0, 2, (m_ImageWidthInPixels - 2*m_Cx)/m_ImageWidthInPixels);
  m_IntrinsicMatrix->SetElement(1, 1, 2*(m_Fy / m_PixelAspectRatio) /(m_ImageHeightInPixels / m_PixelAspectRatio));
  m_IntrinsicMatrix->SetElement(1, 2, (-(m_ImageHeightInPixels / m_PixelAspectRatio) + 2*(m_Cy/m_PixelAspectRatio))/(m_ImageHeightInPixels / m_PixelAspectRatio));
  m_IntrinsicMatrix->SetElement(2, 2, (-zfar-znear)/(zfar-znear));
  m_IntrinsicMatrix->SetElement(2, 3, -2*zfar*znear/(zfar-znear));
  m_IntrinsicMatrix->SetElement(3, 2, -1);

  double widthScale  = (double) m_WindowWidthInPixels  / (double) m_ImageWidthInPixels;
  double heightScale  = (double) m_WindowHeightInPixels  / ((double) m_ImageHeightInPixels / m_PixelAspectRatio);

  int vpw = m_WindowWidthInPixels;
  int vph = m_WindowHeightInPixels;

  if (widthScale < heightScale)
  {
    vph = (int) (((double) m_ImageHeightInPixels / m_PixelAspectRatio) * widthScale);
  }
  else
  {
    vpw = (int) ((double) m_ImageWidthInPixels * heightScale);
  }

  int vpx = m_WindowWidthInPixels  / 2 - vpw / 2;
  int vpy = m_WindowHeightInPixels / 2 - vph / 2;

  assert(glGetError() == GL_NO_ERROR);

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

  assert(glGetError() == GL_NO_ERROR);

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
  os << indent << "vtkOpenGLMatrixDrivenCamera:UseCalibratedCamera=" << UseCalibratedCamera << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_ImageWidthInPixels=" << m_ImageWidthInPixels << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_ImageHeightInPixels=" << m_ImageHeightInPixels << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_WindowWidthInPixels=" << m_WindowWidthInPixels << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera:m_WindowHeightInPixels=" << m_WindowHeightInPixels << std::endl;
  os << indent << "vtkOpenGLMatrixDrivenCamera::m_IntrinsicMatrix" << std::endl;
  m_IntrinsicMatrix->PrintSelf(os, indent);
}
