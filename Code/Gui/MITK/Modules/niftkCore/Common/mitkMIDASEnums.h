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

/*!
 * \enum MIDASLayout
 * \brief Describes the different window layouts that can be achieved in
 * the MIDAS style Display window. So one MIDASLayout could have
 * multiple MIDASOrientations, but most often will contain either Axial,
 * Coronal or Sagittal. This is different to the MIDASLayout as a layout
 * can contain multiple orientations.
 */
enum MIDASLayout
{
  MIDAS_LAYOUT_AXIAL = 0,
  MIDAS_LAYOUT_SAGITTAL = 1,
  MIDAS_LAYOUT_CORONAL = 2,
  MIDAS_LAYOUT_ORTHO = 3,
  MIDAS_LAYOUT_3D = 4,
  MIDAS_LAYOUT_3H = 5,
  MIDAS_LAYOUT_3V = 6,
  MIDAS_LAYOUT_AS_ACQUIRED = 7,
  MIDAS_LAYOUT_UNKNOWN = 8,
  MIDAS_LAYOUT_COR_SAG_H = 9,
  MIDAS_LAYOUT_COR_SAG_V = 10,
  MIDAS_LAYOUT_COR_AX_H = 11,
  MIDAS_LAYOUT_COR_AX_V = 12,
  MIDAS_LAYOUT_SAG_AX_H = 13,
  MIDAS_LAYOUT_SAG_AX_V = 14
};


/*!
 * \brief Returns true if the layout contains only one window, otherwise false.
 */
inline bool IsSingleWindowLayout(MIDASLayout layout)
{
  return layout == MIDAS_LAYOUT_AXIAL
      || layout == MIDAS_LAYOUT_SAGITTAL
      || layout == MIDAS_LAYOUT_CORONAL
      || layout == MIDAS_LAYOUT_3D;
}


/*!
 * \brief Returns true if the layout contains multiple windows, otherwise false.
 */
inline bool IsMultiWindowLayout(MIDASLayout layout)
{
  return !IsSingleWindowLayout(layout);
}

/*!
 * \brief The number of the possible layouts.
 */
const int MIDAS_LAYOUT_NUMBER = 15;

/*!
 * \enum MIDASDropType
 * \brief Describes the different modes that can be used when drag and dropping
 * into the MIDAS style Display window.
 */
enum MIDASDropType
{
  MIDAS_DROP_TYPE_SINGLE = 0,   /** This means that multiple nodes are dropped into a single window. */
  MIDAS_DROP_TYPE_MULTIPLE = 1, /** This means that multiple nodes are dropped across multiple windows. */
  MIDAS_DROP_TYPE_ALL = 2       /** This means that multiple nodes are dropped across all windows for a thumnail effect. */
};

/*!
 * \enum MIDASDefaultInterpolationType
 * \brief Describes what the interpolation type should be set to when an image is dropped.
 */
enum MIDASDefaultInterpolationType
{
  MIDAS_INTERPOLATION_NONE,
  MIDAS_INTERPOLATION_LINEAR,
  MIDAS_INTERPOLATION_CUBIC
};

/*!
 * \enum MIDASBindType
 * \brief Describes valid bind modes.
 */
enum MIDASBindType
{
  MIDAS_BIND_NONE = 0,
  MIDAS_BIND_LAYOUT = 1,
  MIDAS_BIND_CURSORS = 2,
  MIDAS_BIND_MAGNIFICATION = 4,
  MIDAS_BIND_GEOMETRY = 8
};

#endif
