/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseSegmentorController_h
#define __niftkBaseSegmentorController_h

#include <uk_ac_ucl_cmic_gui_qt_commonmidas_Export.h>

#include <QObject>

class niftkBaseSegmentorView;

/**
 * \class niftkBaseSegmentorController
 */
class CMIC_QT_COMMONMIDAS niftkBaseSegmentorController : public QObject
{

  Q_OBJECT

public:

  niftkBaseSegmentorController(niftkBaseSegmentorView* segmentorView);
  virtual ~niftkBaseSegmentorController();

private:

  niftkBaseSegmentorView* m_SegmentorView;

friend class niftkBaseSegmentorView;

};

#endif
