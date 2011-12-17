/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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


} // namespace itk

#endif
