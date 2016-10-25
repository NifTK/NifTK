/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBinaryMaskUtils_h
#define niftkBinaryMaskUtils_h

#include "niftkCoreExports.h"
#include <mitkImage.h>
#include <mitkDataNode.h>

namespace niftk
{

/**
* \brief Checks that the image is in fact single channel, unsigned char.
*/
NIFTKCORE_EXPORT bool IsBinaryMask(const mitk::Image::Pointer& input);

/**
* \brief Checks for a binary property, and the fact that a node contains an image.
*/
NIFTKCORE_EXPORT bool IsBinaryMask(const mitk::DataNode::Pointer& input);

/**
* \brief Performs logical AND of two unsigned char, greyscale, 8 bit images.
*
* Anything non-zero counts as positive, and output is [0|255].
* Throws mitk::Exception if not a binary mask, and not the same size image.
*/
NIFTKCORE_EXPORT void BinaryMaskAndOperator(const mitk::Image::Pointer& input1,
                                            const mitk::Image::Pointer& input2,
                                            mitk::Image::Pointer& output
                                           );

/**
* \brief Performs logical OR of two unsigned char, greyscale, 8 bit images.
*
* Anything non-zero counts as positive, and output is [0|255].
* Throws mitk::Exception if not a binary mask, and not the same size image.
*/
NIFTKCORE_EXPORT void BinaryMaskOrOperator(const mitk::Image::Pointer& input1,
                                           const mitk::Image::Pointer& input2,
                                           mitk::Image::Pointer& output
                                          );

} // end namespace

#endif
