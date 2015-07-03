/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TrackballManipulator_h
#define TrackballManipulator_h

#include <niftkVLExports.h>
#include <vlGraphics/TrackballManipulator.hpp>


/**
 * The only difference to the standard vl::TrackballManipulator is handling
 * mouse-wheel events and pretending they originated from the zoom-button-press.
 */
class NIFTKVL_EXPORT TrackballManipulator : public vl::TrackballManipulator
{

public:
  virtual void mouseWheelEvent(int n);
};


#endif // TrackballManipulator_h
