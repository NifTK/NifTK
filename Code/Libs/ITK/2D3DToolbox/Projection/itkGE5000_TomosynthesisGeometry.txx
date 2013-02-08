/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkGE5000_TomosynthesisGeometry_txx
#define __itkGE5000_TomosynthesisGeometry_txx

#include <iomanip>

#include "itkGE5000_TomosynthesisGeometry.h"

#include "itkLogHelper.h"


namespace itk
{

#define DEGS_TO_RADS 0.01745329

// These are the projection angles for the GE tomosunthesis unit
const double ProjectionAnglesLeft[11]  = { -25.00, -20.32, -15.59, -10.82, -5.99, -1.09,  3.90,   8.99,  14.19,  19.52,  24.98 };
const double ProjectionAnglesRight[11] = {  24.99,  19.52,  14.20,   8.99,  3.90, -1.09, -5.99, -10.82, -15.59, -20.31, -25.00 };


/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */

template<class IntensityType>
GE5000_TomosynthesisGeometry<IntensityType>
::GE5000_TomosynthesisGeometry()
{
  
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class IntensityType>
void 
GE5000_TomosynthesisGeometry<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


/* --------------------------------------------------------------------------
   CalcNormalPosition()
   -------------------------------------------------------------------------- */

template<class IntensityType>
double 
GE5000_TomosynthesisGeometry<IntensityType>
::CalcNormalPosition(double alpha)
{
  double tanAlpha, tanAlphaSqd;

  tanAlpha = tan(alpha*DEGS_TO_RADS);
  tanAlphaSqd = tanAlpha*tanAlpha;

  return -217.*tanAlphaSqd + sqrt(196249. + 149160.*tanAlphaSqd)/(1. + tanAlphaSqd); 
}


/* --------------------------------------------------------------------------
   GetPerspectiveTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename GE5000_TomosynthesisGeometry<IntensityType>::PerspectiveProjectionTransformPointerType
GE5000_TomosynthesisGeometry<IntensityType>
::GetPerspectiveTransform(int i)
{
  double offset = 0;		// The offset from rotation of the x-ray source
  double posnNormal = 0;	// The position of the normal in 'x'

  double u0, v0;

  PerspectiveProjectionTransformPointerType perspTransform;

  this->Initialise();

  // Generate the perspective projection matrix

  perspTransform = PerspectiveProjectionTransformType::New();

  offset = CalcNormalPosition(ProjectionAnglesLeft[i]);

  posnNormal = (217. + offset)*tan(ProjectionAnglesLeft[i]*DEGS_TO_RADS); 

  // By default (u0, v0) is the centre of the first pixel
  // hence to set (u0, v0) to the centre of the 2D plane use
  // ((nx-1)*rx/2, (ny-1)*rx/2)

  u0 = this->m_ProjectionSpacing[0]*(this->m_ProjectionSize[0] - 1.)/2. + posnNormal;
  v0 = this->m_ProjectionSpacing[1]*(this->m_ProjectionSize[1] - 1.)/2.;
  
  perspTransform->SetFocalDistance(217. + offset);
  perspTransform->SetOriginIn2D(u0, v0);

  return perspTransform;
}


/* --------------------------------------------------------------------------
   GetAffineTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename GE5000_TomosynthesisGeometry<IntensityType>::EulerAffineTransformPointerType 
GE5000_TomosynthesisGeometry<IntensityType>
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

  double focalLength = 217. + CalcNormalPosition(ProjectionAnglesLeft[i]);

  posnNormal = focalLength*tan(ProjectionAnglesLeft[i]*DEGS_TO_RADS); 

  // Coordinate system of the the 3D volume is in the corner of the first voxel

  parameters[0] = - this->m_VolumeSpacing[0]*this->m_VolumeSize[0]/2. - posnNormal;
  parameters[1] = - this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2.;
  parameters[2] = focalLength - this->m_VolumeSpacing[2]*this->m_VolumeSize[2];

  affineTransform = EulerAffineTransformType::New();

  center[0] = this->m_VolumeSpacing[0]*this->m_VolumeSize[0]/2.;
  center[1] = this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2.;
  center[2] = this->m_VolumeSpacing[2]*this->m_VolumeSize[2]/2.;

  affineTransform->SetCenter(center);

  affineTransform->SetParameters(parameters);

  return affineTransform;
}


} // namespace itk

#endif
