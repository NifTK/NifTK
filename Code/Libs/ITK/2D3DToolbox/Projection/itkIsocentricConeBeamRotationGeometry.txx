/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkIsocentricConeBeamRotationGeometry_txx
#define __itkIsocentricConeBeamRotationGeometry_txx

#include <iomanip>

#include "itkIsocentricConeBeamRotationGeometry.h"

#include "itkLogHelper.h"


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */

template <class IntensityType>
IsocentricConeBeamRotationGeometry<IntensityType>
::IsocentricConeBeamRotationGeometry() 
{
  m_ProjectionAngles = 0;
  m_RotationType = ISOCENTRIC_CONE_BEAM_ROTATION_IN_Y;

  m_Translation[0] = 0.;
  m_Translation[1] = 0.;
  m_Translation[2] = 0.;
}


/* -----------------------------------------------------------------------
   Initialise()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void 
IsocentricConeBeamRotationGeometry<IntensityType>
::Initialise(void)
{
  if (! this->m_FlagInitialised) {
    
    if ( m_NumberOfProjections <= 0 ) {
      niftkitkErrorMacro("Number of projections must be greater than zero." );
    }
    
    if ( m_FocalLength <= 0 ) {
      niftkitkErrorMacro("Focal length must be greater than zero." );
    }
    
    if (! m_ProjectionAngles) {
      m_ProjectionAngles = new double[m_NumberOfProjections];
    }
    
    ProjectionGeometry<IntensityType>::Initialise();
    
    for (size_t i=0; i<m_NumberOfProjections; i++) {
      
      m_ProjectionAngles[i] = m_FirstAngle + ((double) i)*(m_AngularRange/(((double) m_NumberOfProjections) - 1.));
    } 

    this->m_FlagInitialised = true;
  }
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template <class IntensityType>
void 
IsocentricConeBeamRotationGeometry<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  unsigned int i;

  os << indent << "Number of projections: " << m_NumberOfProjections << std::endl;
  os << indent << "First angle: " << m_FirstAngle << std::endl;
  os << indent << "Angular range: " << m_AngularRange << std::endl;
  os << indent << "Focal length: " << m_FocalLength << std::endl;

  if (m_ProjectionAngles) {
    os << indent << "Projection angles: ";
    for (i=0; i<m_NumberOfProjections; i++)
      os << m_ProjectionAngles[i] << " ";
    os << std::endl;
  }
  else 
    os << indent << "Projection angles: NULL" << std::endl;
}


/* --------------------------------------------------------------------------
   GetPerspectiveTransform(int)
   -------------------------------------------------------------------------- */

template <class IntensityType>
typename IsocentricConeBeamRotationGeometry<IntensityType>::PerspectiveProjectionTransformPointerType
IsocentricConeBeamRotationGeometry<IntensityType>
::GetPerspectiveTransform(int i)
{

  double u0, v0;

  PerspectiveProjectionTransformPointerType perspTransform;

  this->Initialise();

  // Generate the perspective projection matrix

  perspTransform = PerspectiveProjectionTransformType::New();

  // By default (u0, v0) is the centre of the first pixel
  // hence to set (u0, v0) to the centre of the 2D plane use
  // ((nx-1)*rx/2, (ny-1)*rx/2)

  u0 = this->m_ProjectionSpacing[0]*(this->m_ProjectionSize[0] - 1.)/2.;
  v0 = this->m_ProjectionSpacing[1]*(this->m_ProjectionSize[1] - 1.)/2.;
  
  perspTransform->SetFocalDistance(m_FocalLength);
  perspTransform->SetOriginIn2D(u0, v0);

  return perspTransform;
}


/* --------------------------------------------------------------------------
   GetAffineTransform(int)
   -------------------------------------------------------------------------- */

template <class IntensityType>
typename IsocentricConeBeamRotationGeometry<IntensityType>::EulerAffineTransformPointerType 
IsocentricConeBeamRotationGeometry<IntensityType>
::GetAffineTransform(int i)
{
  EulerAffineTransformPointerType affineTransform;
  typename EulerAffineTransformType::ParametersType parameters;
  typename EulerAffineTransformType::InputPointType center;

  this->Initialise();

  affineTransform = EulerAffineTransformType::New();

  // Set the center of rotation

  center[0] = this->m_VolumeSpacing[0]*this->m_VolumeSize[0]/2.;
  center[1] = this->m_VolumeSpacing[1]*this->m_VolumeSize[1]/2.;
  center[2] = this->m_VolumeSpacing[2]*this->m_VolumeSize[2]/2.;

  affineTransform->SetCenter(center);

  // Initialise the affine parameters
  
  parameters.SetSize(12);

  parameters.Fill(0.);


  switch (m_RotationType)
    {

    case ISOCENTRIC_CONE_BEAM_ROTATION_IN_X: {
      
      parameters[3] = m_ProjectionAngles[i] + this->m_RotationInX; // Rotation about the 'x' axis
      parameters[4] = this->m_RotationInY;                         // Rotation about the 'y' axis
      parameters[5] = this->m_RotationInZ;                         // Rotation about the 'z' axis

      break;
    }

    case ISOCENTRIC_CONE_BEAM_ROTATION_IN_Y: {

      parameters[3] = this->m_RotationInX;                         // Rotation about the 'x' axis
      parameters[4] = m_ProjectionAngles[i] + this->m_RotationInY; // Rotation about the 'y' axis
      parameters[5] = this->m_RotationInZ;                         // Rotation about the 'z' axis

      break;
    }

    case ISOCENTRIC_CONE_BEAM_ROTATION_IN_Z: {

      parameters[3] = this->m_RotationInX;                         // Rotation about the 'x' axis
      parameters[4] = this->m_RotationInY;                         // Rotation about the 'y' axis
      parameters[5] = m_ProjectionAngles[i] + this->m_RotationInZ; // Rotation about the 'z' axis

      break;
    }
    }

  parameters[6] = 1.;		// Scale factor along the 'x' axis
  parameters[7] = 1.;		// Scale factor along the 'y' axis
  parameters[8] = 1.;		// Scale factor along the 'z' axis

 // Coordinate system of the the 3D volume is in the corner of the first voxel

  parameters[0] = m_Translation[0] - center[0];
  parameters[1] = m_Translation[1] - center[1];
  parameters[2] = m_Translation[2] - center[2] + m_FocalLength/2.;

  affineTransform->SetParameters(parameters);

  return affineTransform;
}


} // namespace itk

#endif
