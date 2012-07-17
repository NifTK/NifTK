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

#ifndef MITKMIDASIMAGEUTILS_H
#define MITKMIDASIMAGEUTILS_H

#include "niftkMitkExtExports.h"
#include "itkConversionUtils.h"
#include "itkSpatialOrientationAdapter.h"
#include "mitkImage.h"
#include "mitkDataNode.h"
#include "mitkMIDASEnums.h"

/**
 * \file mitkMIDASImageUtils.h
 * \brief Some useful MIDAS related image utilities, such as working out the As Acquired orientation.
 */
namespace mitk
{
  /**
   * \brief Given an image, returns the MIDASOrientation for the XY plane.
   * \param itkImage an ITK image
   * \param outputOrientation the output MIDASOrientation as either axial, sagittal, coronal,
   * and defaults to coronal if the as Acquired orientation cannot be found.
   */
  template<typename TPixel, unsigned int VImageDimension>
  void
  GetAsAcquiredOrientation(
    itk::Image<TPixel, VImageDimension>* itkImage,
    MIDASOrientation &outputOrientation
  );

  /**
   * \brief Checks if the supplied node is an image, and returns the MIDASView corresponding to the XY plane, or else returns the supplied default.
   * \param defaultView A default MIDASView that will be returned if we can't work out the As Acquired view.
   * \param node A node to check.
   * \return MIDASView the As Acquired view, or the defaultView.
   */
  NIFTKMITKEXT_EXPORT MIDASView GetAsAcquiredView(const MIDASView& defaultView, const mitk::DataNode* node);
}

#endif // MITKMIDASIMAGEUTILS_H
