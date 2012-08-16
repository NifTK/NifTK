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
#include "itkMIDASHelper.h"

/**
 * \file mitkMIDASOrientationUtils.h
 * \brief Some utilities to help with MIDAS conventions on orientation.
 */
namespace mitk
{

/**
 * \brief Converts an MITK orientation enum to an ITK orientation enum, and ideally these types should be merged.
 */
NIFTKMITKEXT_EXPORT itk::ORIENTATION_ENUM GetItkOrientation(const MIDASOrientation& orientation);


/*
 * \brief Converts an ITK orientation enum to an MITK orientation enum, and ideally these types should be merged.
 */
NIFTKMITKEXT_EXPORT MIDASOrientation GetMitkOrientation(const itk::ORIENTATION_ENUM& orientation);


/**
 * \brief See GetUpDirection as in effect, we are only using the direction cosines from the geometry.
 */
NIFTKMITKEXT_EXPORT int GetUpDirection(const mitk::Image* image, const MIDASOrientation& orientation);


/**
 * \brief Returns either +1, or -1 to indicate in which direction you should change the slice number to go "up".
 * \param geometry An MITK geometry, not NULL.
 * \param orientation a MIDASOrientation corresponding to Axial, Coronal or Sagittal.
 * \return -1 or +1 telling you to either increase of decrease the slice number or 0 for "unknown".
 *
 * So, the MIDAS spec is: Shortcut key A=Up, Z=Down which means:
 * <pre>
 * Axial: A=Superior, Z=Inferior
 * Coronal: A=Anterior, Z=Posterior
 * Sagittal: A=Right, Z=Left
 * </pre>
 */

NIFTKMITKEXT_EXPORT int GetUpDirection(const mitk::Geometry3D* geometry, const MIDASOrientation& orientation);


/**
 * \brief Returns either -1 (unknown), or [0,1,2] for the x, y, or z axis corresponding to the through plane direction for the specified orientation.
 * \param image An MITK image, not NULL.
 * \param orientation a MIDASOrientation corresponding to Axial, Coronal or Sagittal.
 * \return -1=unknown, or the axis number [0,1,2].
 */
NIFTKMITKEXT_EXPORT int GetThroughPlaneAxis(const mitk::Image* image, const MIDASOrientation& orientation);


/**
 * \brief Returns the Orientation String (RPI, RAS etc).
 * \param image An MITK image, not NULL.
 *
 * NOTE: MIDAS Analyze are flipped in the MITK GUI. This means if you use the default ITK reader
 * which is used for example in the command line app niftkImageInfo, you will get a different answer to
 * this method, as this method will be run from within the MITK GUI, and hence will be using itkDRCAnalyzeImageIO.
 */
NIFTKMITKEXT_EXPORT std::string GetOrientationString(const mitk::Image* image);

} // end namespace

#endif
