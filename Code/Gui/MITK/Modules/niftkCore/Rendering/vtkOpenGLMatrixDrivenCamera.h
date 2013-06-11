/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef vtkOpenGLMatrixDrivenCamera_h
#define vtkOpenGLMatrixDrivenCamera_h

#include "niftkCoreExports.h"
#include <vtkOpenGLCamera.h>
#include <vtkRenderer.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <vtkSetGet.h>

/**
 * \class vtkOpenGLMatrixDrivenCamera
 * \brief Subclass of vtkCamera so we can just set the relevant intrinsic matrix,
 * the size of the image used in camera calibration, and the size of the actual window
 * to get a calibrated camera view of the rendered scene. So, compared with
 * vtkOpenGLCamera, this class ignores all the stuff to do with orthographic
 * or perspective projection, and just uses the matrix and image/window size.
 *
 * \see http://sightations.wordpress.com/2010/08/03/simulating-calibrated-cameras-in-opengl/
 * \see http://jamesgregson.blogspot.co.uk/2011/11/matching-calibrated-cameras-with-opengl.html
 * \see http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
 *
 * where the last of these links seemed most intuitive to my way of thinking.
 */
class NIFTKCORE_EXPORT vtkOpenGLMatrixDrivenCamera : public vtkOpenGLCamera
{
public:
  static vtkOpenGLMatrixDrivenCamera *New();
  vtkTypeMacro(vtkOpenGLMatrixDrivenCamera, vtkOpenGLCamera);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  void Render(vtkRenderer *ren);

  vtkSetMacro(UseCalibratedCamera, bool);
  vtkGetMacro(UseCalibratedCamera, bool);

  /**
   * \brief Set the size of the image in pixels that was used while calibrating the camera model.
   */
  void SetCalibratedImageSize(const int& width, const int& height);

  /**
   * \brief Set the window/widget size currently used.
   */
  void SetActualWindowSize(const int& width, const int& height);

  /**
   * \brief Set the intrinsic parameters as determined from camera calibration.
   */
  void SetIntrinsicParameters(const double& fx, const double& fy,
                              const double &cx, const double& cy
                              );
protected:  

  vtkOpenGLMatrixDrivenCamera();
  ~vtkOpenGLMatrixDrivenCamera() {};

private:

  vtkOpenGLMatrixDrivenCamera(const vtkOpenGLMatrixDrivenCamera&);  // Purposefully not implemented.
  void operator=(const vtkOpenGLMatrixDrivenCamera&);  // Purposefully not implemented.

  vtkSmartPointer<vtkMatrix4x4> m_IntrinsicMatrix;
  bool UseCalibratedCamera;
  int m_ImageWidth;
  int m_ImageHeight;
  int m_WindowWidth;
  int m_WindowHeight;
  double m_Fx;
  double m_Fy;
  double m_Cx;
  double m_Cy;
};

#endif
