/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkIGITestDataUtils_h
#define mitkIGITestDataUtils_h

#include "niftkIGIExports.h"
#include <mitkVector.h>
#include <mitkDataNode.h>
#include <mitkSurface.h>
#include <mitkProperties.h>

/**
 * \file mitkIGITestDataUtils.h
 * \brief Some useful functions to create bits of test data for IGI purposes.
 */
namespace mitk
{

/**
 * \brief Creates a cone, radius 7.5mm, height 15.0mm, centred at centrePoint, facing the given direction.
 * \param label the name as it appears in DataStorage
 * \param centerPoint the centre of the cone (in VTK terms, see SetCenter).
 * \param direction the directionof the cone (in VTK terms, see SetDirection).
 */
NIFTKIGI_EXPORT
mitk::DataNode::Pointer CreateConeRepresentation(
    const char* label,
    const mitk::Vector3D& centerPoint,
    const mitk::Vector3D& direction);

/**
 * \brief Calls CreateConeRepresentation, centering the cone at (0,0,7.5).
 * \param label the name as it appears in DataStorage
 * \param direction the directionof the cone (in VTK terms, see SetDirection).
 */
NIFTKIGI_EXPORT
mitk::DataNode::Pointer CreateConeRepresentation(
    const char* label,
    mitk::Vector3D& direction
    );

} // end namespace

#endif
