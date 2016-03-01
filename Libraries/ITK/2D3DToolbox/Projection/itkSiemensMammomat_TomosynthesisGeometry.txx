/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSiemensMammomat_TomosynthesisGeometry_txx
#define __itkSiemensMammomat_TomosynthesisGeometry_txx

#include <iomanip>

#include "itkSiemensMammomat_TomosynthesisGeometry.h"

#include "itkLogHelper.h"


namespace itk
{
#define DEGS_TO_RADS 0.01745329

// These are the projection angles for the Siemens tomosynthesis unit

const double ProjectionAnglesSiemensMammomat_LMLO[25] = { 25.00, 24.06, 22.03, 20.02, 18.25, 16.37, 14.41, 12.58, 10.56, 08.71, 06.75, 04.86, 03.00, 01.11, -00.82, -02.78, -04.74, -06.56, -08.47, -10.39, -12.32, -14.19, -16.02, -18.01, -19.89 };

#if 0
const double ProjectionAnglesSiemensMammomat_RMLO[25] = { 25.00, 24.06, 22.03, 20.02, 18.25, 16.37, 14.41, 12.58, 10.56, 08.71, 06.75, 04.86, 03.00, 01.11, -00.82, -02.78, -04.74, -06.56, -08.47, -10.39, -12.32, -14.19, -16.02, -18.01, -19.89 };
#else
const double ProjectionAnglesSiemensMammomat_RMLO[25] = { -25.28, -24.41, -22.42, -20.51, -18.6, -16.71, -14.71, -12.84, -10.93, -9.02, -7, -5.14, -3.26, -1.29, 0.64, 2.65, 4.58, 6.52, 8.39, 10.26, 12.21, 14.13, 16.06, 17.97, 19.92 };
#endif

#if 0
const double ProjectionAnglesSiemensMammomat_LCC[25] = { 25.00, 24.06, 22.03, 20.02, 18.25, 16.37, 14.41, 12.58, 10.56, 08.71, 06.75, 04.86, 03.00, 01.11, -00.82, -02.78, -04.74, -06.56, -08.47, -10.39, -12.32, -14.19, -16.02, -18.01, -19.89 };
#else
const double ProjectionAnglesSiemensMammomat_LCC[25] = { -25.25, -24.26, -22.28, -20.39, -18.46, -16.54, -14.66, -12.74, -10.77, -8.87, -6.93, -4.96, -3.12, -1.05, 0.7, 2.65, 4.61, 6.49, 8.37, 10.3, 12.23, 14.2, 16.14, 18.06, 19.99 };
#endif

#if 1
const double ProjectionAnglesSiemensMammomat_RCC[25] = { -25.16, -24.24, -22.23, -20.3, -18.45, -16.48, -14.61, -12.68, -10.71, -8.83, -6.88, -4.96, -3.02, -1.1, 0.88, 2.82, 4.75, 6.55, 8.46, 10.45, 12.29, 14.17, 16.13, 18.07, 19.98 };
#endif


/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */

template<class IntensityType>
SiemensMammomat_TomosynthesisGeometry<IntensityType>
::SiemensMammomat_TomosynthesisGeometry() 
  : nProjections( 25 ), 
    heightOfTable( 17 ), 
    heightOfIsoCenterFromTable( 30 ), 
    distSourceToIsoCenter( 608.5 ),
    sizeOfDetectorInX( 240 ),
    sizeOfDetectorInY( 305 )
{

}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class IntensityType>
void
SiemensMammomat_TomosynthesisGeometry<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


/* --------------------------------------------------------------------------
   GetAngles()
   -------------------------------------------------------------------------- */

template<class IntensityType>
const double*
SiemensMammomat_TomosynthesisGeometry<IntensityType>
::GetAngles( void )
{
  if ( (this->m_FlagSide == Superclass::LEFT_SIDE) &&
       (this->m_FlagView == Superclass::MLO_VIEW) )
  {
    return ProjectionAnglesSiemensMammomat_LMLO;
  }
  else if ( (this->m_FlagSide == Superclass::RIGHT_SIDE) &&
            (this->m_FlagView == Superclass::MLO_VIEW) )
  {
    return ProjectionAnglesSiemensMammomat_RMLO;
  }

  if ( (this->m_FlagSide == Superclass::LEFT_SIDE) &&
       (this->m_FlagView == Superclass::CC_VIEW) )
  {
    return ProjectionAnglesSiemensMammomat_LCC;
  }
  else if ( (this->m_FlagSide == Superclass::RIGHT_SIDE) &&
            (this->m_FlagView == Superclass::CC_VIEW) )
  {
    return ProjectionAnglesSiemensMammomat_RCC;
  }

  
  else
  {
    return 0;
  }
}


/* --------------------------------------------------------------------------
   GetPerspectiveTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename SiemensMammomat_TomosynthesisGeometry<IntensityType>::PerspectiveProjectionTransformPointerType
SiemensMammomat_TomosynthesisGeometry<IntensityType>
::GetPerspectiveTransform(int i)
{
  double posnNormal = 0;	// The position of the normal in 'x'
  double u0, v0;

  const double *ProjectionAnglesSiemensMammomat = 0;

  PerspectiveProjectionTransformPointerType perspTransform;

  this->Initialise();

  // Generate the perspective projection matrix

  perspTransform = PerspectiveProjectionTransformType::New();

  ProjectionAnglesSiemensMammomat = GetAngles();

  posnNormal = distSourceToIsoCenter*sin( ProjectionAnglesSiemensMammomat[i]*DEGS_TO_RADS );

  // By default (u0, v0) is the centre of the first pixel
  // hence to set (u0, v0) to the centre of the 2D plane use
  // ((nx-1)*rx/2, (ny-1)*rx/2)

  if ( this->m_FlagSide == Superclass::LEFT_SIDE )
  {
    u0 = 0.;
  }
  else if ( this->m_FlagSide == Superclass::RIGHT_SIDE )
  {
    u0 = sizeOfDetectorInX;
  }

  v0 = this->m_ProjectionSpacing[1]*(this->m_ProjectionSize[1] - 1.)/2. + posnNormal;

  perspTransform->SetFocalDistance( distSourceToIsoCenter
                                    *cos( ProjectionAnglesSiemensMammomat[i]*DEGS_TO_RADS ) + 
                                    heightOfIsoCenterFromTable + heightOfTable );

  perspTransform->SetOriginIn2D(u0, v0);

  return perspTransform;
}


/* --------------------------------------------------------------------------
   GetAffineTransform(int)
   -------------------------------------------------------------------------- */

template<class IntensityType>
typename SiemensMammomat_TomosynthesisGeometry<IntensityType>::EulerAffineTransformPointerType
SiemensMammomat_TomosynthesisGeometry<IntensityType>
::GetAffineTransform(int i)
{
  double posnNormal = 0;	// The position of the normal in 'x'
  double focalLength = 0;       // The focal length

  const double *ProjectionAnglesSiemensMammomat = 0;

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

  ProjectionAnglesSiemensMammomat = GetAngles();

  focalLength = 
    distSourceToIsoCenter*cos(ProjectionAnglesSiemensMammomat[i]*DEGS_TO_RADS) + 
    heightOfIsoCenterFromTable + heightOfTable;
  
  posnNormal = distSourceToIsoCenter*sin(ProjectionAnglesSiemensMammomat[i]*DEGS_TO_RADS);


  // Coordinate system of the the 3D volume is in the corner of the first voxel

  if ( this->m_FlagSide == Superclass::LEFT_SIDE )
  {
    parameters[0] = 0.;
  }
  else if ( this->m_FlagSide == Superclass::RIGHT_SIDE )
  {
    parameters[0] = - this->m_VolumeSpacing[0]*this->m_VolumeSize[0];
  }

  parameters[1] = - this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2. - posnNormal;
  parameters[2] = focalLength - heightOfTable - this->m_VolumeSpacing[2]*this->m_VolumeSize[2];

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

