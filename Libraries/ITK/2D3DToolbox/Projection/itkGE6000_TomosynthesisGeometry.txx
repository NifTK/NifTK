/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkGE6000_TomosynthesisGeometry_txx
#define __itkGE6000_TomosynthesisGeometry_txx

#include <iomanip>

#include "itkGE6000_TomosynthesisGeometry.h"

#include <itkLogHelper.h>


namespace itk
{
#define DEGS_TO_RADS 0.01745329

// These are the projection angles for the GE tomosunthesis unit
const double ProjectionAnglesLeftGE6000[15]  = { -18.79, -16.12, -13.46, -10.80, -8.14, -5.48, -2.82, -0.16, 2.50, 5.16, 7.82, 10.48, 13.14, 15.80, 18.47 };
const double ProjectionAnglesRightGE6000[15] = {  18.79, 16.10, 13.42, 10.75, 8.08, 5.41, 2.74, 0.08, -2.59, -5.26, -7.93, -10.60, -13.27, -15.94, -18.61 };


/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */

template<class IntensityType>
GE6000_TomosynthesisGeometry<IntensityType>
::GE6000_TomosynthesisGeometry()
{

}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class IntensityType>
void
GE6000_TomosynthesisGeometry<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


/* --------------------------------------------------------------------------
   CalcNormalPosition()
   -------------------------------------------------------------------------- */

template<class IntensityType>
double
GE6000_TomosynthesisGeometry<IntensityType>
::CalcNormalPosition(double alpha)
{
  double tanAlpha, tanAlphaSqd;

  tanAlpha = tan(alpha*DEGS_TO_RADS);
  tanAlphaSqd = tanAlpha*tanAlpha;

  return -40.*tanAlphaSqd + sqrt(384400. + 382800.*tanAlphaSqd)/(1. + tanAlphaSqd);
}


/* --------------------------------------------------------------------------
   GetPerspectiveTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename GE6000_TomosynthesisGeometry<IntensityType>::PerspectiveProjectionTransformPointerType
GE6000_TomosynthesisGeometry<IntensityType>
::GetPerspectiveTransform(int i)
{
  double offset = 0;		// The offset from rotation of the x-ray source
  double posnNormal = 0;	// The position of the normal in 'x'

  double u0, v0;

  PerspectiveProjectionTransformPointerType perspTransform;

  this->Initialise();

  // Generate the perspective projection matrix

  perspTransform = PerspectiveProjectionTransformType::New();

  offset = CalcNormalPosition(-ProjectionAnglesLeftGE6000[i]);

  posnNormal = (40. + offset)*tan(-ProjectionAnglesLeftGE6000[i]*DEGS_TO_RADS);

  // By default (u0, v0) is the centre of the first pixel
  // hence to set (u0, v0) to the centre of the 2D plane use
  // ((nx-1)*rx/2, (ny-1)*rx/2)

  u0 = this->m_ProjectionSpacing[0]*(this->m_ProjectionSize[0] - 1.)/2. + posnNormal;
  v0 = this->m_ProjectionSpacing[1]*(this->m_ProjectionSize[1] - 1.)/2.;

  perspTransform->SetFocalDistance(40. + offset);
  perspTransform->SetOriginIn2D(u0, v0);

  return perspTransform;
}


/* --------------------------------------------------------------------------
   GetAffineTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename GE6000_TomosynthesisGeometry<IntensityType>::EulerAffineTransformPointerType
GE6000_TomosynthesisGeometry<IntensityType>
::GetAffineTransform(int i)
{
  double posnNormal = 0;	// The position of the normal in 'x'

  EulerAffineTransformPointerType affineTransform;
  typename EulerAffineTransformType::ParametersType parameters;
  typename EulerAffineTransformType::CenterType center;

  this->Initialise();


  // Initialise the affine parameters

  parameters.SetSize(12);

  parameters.Fill(0.);

  parameters[3] = this->m_RotationInX; // Rotation about the 'x' axis
  parameters[4] = this->m_RotationInY; // Rotation about the 'y' axis
  parameters[5] = this->m_RotationInZ; // Rotation about the 'z' axis

  parameters[6] = 1.;		// Scale factor along the 'x' axis
  parameters[7] = 1.;		// Scale factor along the 'y' axis
  parameters[8] = 1.;		// Scale factor along the 'z' axis

  double focalLength = 40. + CalcNormalPosition(-ProjectionAnglesLeftGE6000[i]);

  posnNormal = focalLength*tan(-ProjectionAnglesLeftGE6000[i]*DEGS_TO_RADS);

  // Coordinate system of the the 3D volume is in the corner of the first voxel

  parameters[0] = - this->m_VolumeSpacing[0]*this->m_VolumeSize[0]/2. - posnNormal;
  parameters[1] = - this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2.;
  parameters[2] = focalLength - 19.0 - this->m_VolumeSpacing[2]*this->m_VolumeSize[2];

  affineTransform = EulerAffineTransformType::New();

  center[0] = this->m_VolumeSpacing[0]*this->m_VolumeSize[0]/2.;
  center[1] = this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2.;
  center[2] = this->m_VolumeSpacing[2]*this->m_VolumeSize[2]/2.;

  affineTransform->SetParameters(parameters);

  return affineTransform;
}


} // namespace itk

#endif

