/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMakeGeometry_h
#define mitkMakeGeometry_h

#include "niftkIGIExports.h"
#include <mitkSurface.h>
#include <QString>
#include <vtkMatrix4x4.h>
/**
 * \brief For visualisation purposes, creates a representation of the laparoscope.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeLaparoscope(std::string rigidBodyFilename, std::string handeyeFilename );

/**
 * \brief For visualisation purposes, creates a representation of the pointer.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakePointer(std::string rigidBodyFilename, std::string handeyeFilename );

/**
 * \brief For visualisation purposes, creates a representation of the reference.
 * \param the rigid body filename to define the location of the tracking markers
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeReference(std::string rigidBodyFilename, std::string handeyeFilename );

/**
 * \brief For visualisation purposes, make a wall of a cube
 * \param the size of the cube in mm 
 * \param which wall to make 
 * \param the xoffset, room will be centred at x= size * xOffset 
 * */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeAWall( const int& whichwall, const float& size = 4000,
    const float& xOffset = 0.0 , const float& yOffset = 0.0, const float& zOffset = -0.3, 
    const float& thickness = 10.0);

/**
 * \brief For visualisation purposes, make a nice big axes
 * \param the length of the axis
 * \param whether or not the axis is symmetric 
 * */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeXAxes( const float& length = 4000,const bool& symmetric = false);
/**
 * \brief For visualisation purposes, make a nice big axes
 * \param the length of the axis
 * \param whether or not the axis is symmetric 
 * */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeYAxes( const float& length = 4000,const bool& symmetric = false);
/**
 * \brief For visualisation purposes, make a nice big axes
 * \param the length of the axis
 * \param whether or not the axis is symmetric 
 * */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeZAxes( const float& length = 4000,const bool& symmetric = true);

/** 
 * \brief a special type of axis useful for cameras
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeLapLensAxes();

/**
 * \brief for visualisation purposes, make a representation of an Optotrak Certus 
 * camera unit
 * \param the width of the camera unit
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeOptotrak( const float & width = 500);

/**
 * \brief For visualisation purposes, creates a representation of a transrectal ultrasound probe
 * \param the handeye calibration to define the tool origin
 */
NIFTKIGI_EXPORT mitk::Surface::Pointer MakeTransrectalUSProbe(std::string handeyeFilename );


#endif
