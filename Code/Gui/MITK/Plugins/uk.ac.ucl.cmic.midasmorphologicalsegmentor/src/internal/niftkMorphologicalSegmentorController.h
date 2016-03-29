/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMorphologicalSegmentorController_h
#define __niftkMorphologicalSegmentorController_h

#include <niftkBaseSegmentorController.h>


class niftkMorphologicalSegmentorView;

/**
 * \class niftkMorphologicalSegmentorController
 */
class niftkMorphologicalSegmentorController : public niftkBaseSegmentorController
{

  Q_OBJECT

public:

  niftkMorphologicalSegmentorController(niftkMorphologicalSegmentorView* segmentorView);
  virtual ~niftkMorphologicalSegmentorController();

private:

  niftkMorphologicalSegmentorView* m_MorphologicalSegmentorView;

friend class niftkMorphologicalSegmentorView;

};

#endif
