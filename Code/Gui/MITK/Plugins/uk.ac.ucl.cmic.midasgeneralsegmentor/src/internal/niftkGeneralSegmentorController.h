/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorController_h
#define __niftkGeneralSegmentorController_h

#include <niftkBaseSegmentorController.h>


class niftkGeneralSegmentorView;

/**
 * \class niftkGeneralSegmentorController
 */
class niftkGeneralSegmentorController : public niftkBaseSegmentorController
{

  Q_OBJECT

public:

  niftkGeneralSegmentorController(niftkGeneralSegmentorView* segmentorView);
  virtual ~niftkGeneralSegmentorController();

protected:

  /// \brief Registers the segmentation tools provided by this segmentor.
  /// Registers the draw, seed, poly and posn tools.
  void RegisterTools() override;

private:

  niftkGeneralSegmentorView* m_GeneralSegmentorView;

friend class niftkGeneralSegmentorView;

};

#endif
