/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkMaskMergerGUI_h
#define niftkMaskMergerGUI_h

#include <QWidget>
#include "ui_niftkMaskMergerGUI.h"
#include <niftkBaseGUI.h>
#include <mitkDataStorage.h>

namespace niftk
{

/// \class MaskMergerGUI
/// \brief Widgets for Mask Merger plugin.
class MaskMergerGUI : public BaseGUI,
                      private Ui::MaskMergerGUI
{

  Q_OBJECT

public:

  MaskMergerGUI(QWidget* parent);
  virtual ~MaskMergerGUI();

  void SetDataStorage(mitk::DataStorage* storage);
  void ResetLeft();
  void ResetRight();

signals:

  void LeftMask1SelectionChanged(const mitk::DataNode*);
  void LeftMask2SelectionChanged(const mitk::DataNode*);
  void RightMask1SelectionChanged(const mitk::DataNode*);
  void RightMask2SelectionChanged(const mitk::DataNode*);

private:

  MaskMergerGUI(const MaskMergerGUI&);  // Purposefully not implemented.
  void operator=(const MaskMergerGUI&);  // Purposefully not implemented.
};

} // end namespace

#endif
