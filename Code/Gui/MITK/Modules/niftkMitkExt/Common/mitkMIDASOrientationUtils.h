/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASORIENTATIONUTILS_H
#define MITKMIDASORIENTATIONUTILS_H

#include "niftkMitkExtExports.h"
#include <mitkImage.h>
#include "mitkMIDASEnums.h"

/**
 * \file mitkMIDASOrientationUtils.h
 * \brief Some utilities to help with MIDAS conventions on orientation.
 */
namespace mitk
{

/**
 * \brief Returns either +1, or -1 to indicate in which direction you should change the slice number to go "up".
 * \param image An MITK image, not NULL.
 * \param orientation a MIDAS itk::ORIENTATION_ENUM corresponding to Axial, Coronal or Sagittal.
 * \return -1 or +1 telling you to either increase of decrease the slice number.
 *
 * So, the spec is: Shortcut key A=Up, Z=Down which means:
 * <pre>
 * Axial: A=Superior, Z=Inferior
 * Coronal: A=Anterior, Z=Posterior
 * Sagittal: A=Right, Z=Left
 * </pre>
 */
NIFTKMITKEXT_EXPORT int GetUpDirection(const mitk::Image* image, const MIDASOrientation& orientation);

} // end namespace

#endif
