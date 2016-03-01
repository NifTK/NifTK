/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "TrackballManipulator.h"


//-----------------------------------------------------------------------------
void TrackballManipulator::mouseWheelEvent(int n)
{
  // default wheel delta is 120, i think.
  n *= 30;

  // fake a zoom button event.
  vl::TrackballManipulator::mouseDownEvent((vl::EMouseButton) zoomButton(), 100, 100);
  vl::TrackballManipulator::mouseMoveEvent(100, 100 + n);
  vl::TrackballManipulator::mouseUpEvent((vl::EMouseButton) zoomButton(), 100, 100 + n);
}
