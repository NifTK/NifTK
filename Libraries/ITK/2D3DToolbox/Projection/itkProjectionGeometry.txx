/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkProjectionGeometry_txx
#define __itkProjectionGeometry_txx

#include "itkProjectionGeometry.h"

#include "itkLogHelper.h"



namespace itk
{
 
/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */

template<class IntensityType>
ProjectionGeometry<IntensityType>
::ProjectionGeometry()
{
  m_FlagInitialised = false;

  m_FlagSide = UNDEFINED_SIDE;
  m_FlagView = UNDEFINED_VIEW;

  m_RotationInX = 0.;
  m_RotationInY = 0.;
  m_RotationInZ = 0.;

  m_VolumeSize[0]    = m_VolumeSize[1]    = m_VolumeSize[2]    = 0;
  m_VolumeSpacing[0] = m_VolumeSpacing[1] = m_VolumeSpacing[2] = 0.;

  m_ProjectionSize[0]    = m_ProjectionSize[1]    = 0;
  m_ProjectionSpacing[0] = m_ProjectionSpacing[1] = 0.;
}

     
/* -----------------------------------------------------------------------
   Initialise()
   ----------------------------------------------------------------------- */

template<class IntensityType>
void 
ProjectionGeometry<IntensityType>
::Initialise(void)
{
  if (! m_FlagInitialised) {

    if ( (m_FlagSide != LEFT_SIDE) && (m_FlagSide != RIGHT_SIDE) ) {
      niftkitkErrorMacro( "Projection geometry side (L or R) must be defined." );
    }

    if ( (m_FlagView != CC_VIEW) && (m_FlagView != MLO_VIEW) ) {
      niftkitkErrorMacro( "Projection geometry view (CC or MLO) must be defined." );
    }

    niftkitkDebugMacro(<<"Projection geometry rotation in 'x': " << m_RotationInX);
    niftkitkDebugMacro(<<"Projection geometry rotation in 'y': " << m_RotationInY);
    niftkitkDebugMacro(<<"Projection geometry rotation in 'z': " << m_RotationInZ);
    niftkitkDebugMacro(<<"Projection geometry volume size: " << m_VolumeSize);
    niftkitkDebugMacro(<<"Projection geometry volume resolution: " << m_VolumeSpacing);
    niftkitkDebugMacro(<<"Projection geometry projection size: " << m_ProjectionSize);
    niftkitkDebugMacro(<<"Projection geometry projection resolution: " << m_ProjectionSpacing);

    if ( (m_VolumeSize[0] == 0) || (m_VolumeSize[1] == 0) || (m_VolumeSize[2] == 0) ) {
      niftkitkErrorMacro( "3D volume region must be defined." );
    }
    if ( (m_VolumeSpacing[0] == 0) || (m_VolumeSpacing[1] == 0) || (m_VolumeSpacing[2] == 0) ) {
      niftkitkErrorMacro( "3D volume resolution must be defined." );
    }

    if ( (m_ProjectionSize[0] == 0) || (m_ProjectionSize[1] == 0) ) {
      niftkitkErrorMacro( "2D projection image region must be defined." );
    }
    if ( (m_ProjectionSpacing[0] == 0) || (m_ProjectionSpacing[1] == 0) ) {
      niftkitkErrorMacro( "2D projection image resolution must be defined." );
    }

    m_FlagInitialised = true;
  }
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class IntensityType>
void
ProjectionGeometry<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  if ( m_FlagSide == UNDEFINED_SIDE )
    os << indent << "Side is undefined" << std::endl;
  else if ( m_FlagSide == LEFT_SIDE )
    os << indent << "Side is left" << std::endl;
  else if ( m_FlagSide == RIGHT_SIDE )
    os << indent << "Side is right" << std::endl;
  else
    os << indent << "Side value is unrecognised" << std::endl << std::endl;


  if ( m_FlagView == UNDEFINED_VIEW )
    os << indent << "View is undefined" << std::endl;
  else if ( m_FlagView == CC_VIEW )
    os << indent << "View is CC" << std::endl;
  else if ( m_FlagView == MLO_VIEW )
    os << indent << "View is MLO" << std::endl;
  else
    os << indent << "View value is unrecognised" << std::endl << std::endl;

  
  os << indent << "Rotation in 'x': " << m_RotationInX << std::endl;
  os << indent << "Rotation in 'y': " << m_RotationInY << std::endl;
  os << indent << "Rotation in 'z': " << m_RotationInZ << std::endl << std::endl;

  os << indent << "3D volume size: " << m_VolumeSize << std::endl;
  os << indent << "3D volume spacing: " << m_VolumeSpacing << std::endl << std::endl;

  os << indent << "3D projection size: " << m_ProjectionSize << std::endl;
  os << indent << "3D projection spacing: " << m_ProjectionSpacing << std::endl << std::endl;
}


} // namespace itk

#endif
