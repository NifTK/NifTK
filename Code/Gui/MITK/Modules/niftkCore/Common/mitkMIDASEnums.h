/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASEnums_h
#define mitkMIDASEnums_h

/*!
 * \file mitkMIDASEnums.h
 * \brief Contains MIDAS enums, which we move out of the classes, so
 * they are independent, which makes manually analysing classes for
 * their include dependencies a bit easier.
 */

/*!
 * \enum MIDASOrientation
 * \brief Describes the different types of orientation, axial, sagittal, coronal,
 * that can be achieved in the MIDAS style Display window. This is different from
 * the MIDASLayout. The orientation might be used to refer to the axis of an image,
 * so an image can ONLY be sampled in AXIAL, SAGITTAL and CORONAL direction.
 */
enum MIDASOrientation
{
  MIDAS_ORIENTATION_AXIAL = 0,
  MIDAS_ORIENTATION_SAGITTAL = 1,
  MIDAS_ORIENTATION_CORONAL = 2,
  MIDAS_ORIENTATION_UNKNOWN = 3
};

/*!
 * \brief The number of the possible orientations.
 */
const int MIDAS_ORIENTATION_NUMBER = 4;

#endif
