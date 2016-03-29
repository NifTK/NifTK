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

#include <mitkToolManager.h>

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

  /// \brief Returns the segmentation tool manager used by the segmentor.
  mitk::ToolManager* GetToolManager() const;

private:

  mitk::ToolManager::Pointer m_ToolManager;

  niftkBaseSegmentorView* m_SegmentorView;

friend class niftkBaseSegmentorView;

};

#endif
