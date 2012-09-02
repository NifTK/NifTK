/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGITESTDATAUTILS_H
#define MITKIGITESTDATAUTILS_H

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

#endif // MITKIGITESTDATAUTILS_H

