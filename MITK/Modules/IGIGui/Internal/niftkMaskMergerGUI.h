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
/// \brief Widgets for Maske Merger plugin.
class MaskMergerGUI : public BaseGUI,
                      private Ui::MaskMergerGUI
{

  Q_OBJECT

public:

  MaskMergerGUI(QWidget* parent);
  virtual ~MaskMergerGUI();

  void SetDataStorage(mitk::DataStorage* storage);

signals:

  void OnLeftSelectionChanged(const mitk::DataNode*);
  void OnRightSelectionChanged(const mitk::DataNode*);
  void OnDoItNowPressed();
  void OnManualUpdateClicked(bool);
  void OnAutomaticUpdateClicked(bool);

private:

  MaskMergerGUI(const MaskMergerGUI&);  // Purposefully not implemented.
  void operator=(const MaskMergerGUI&);  // Purposefully not implemented.
};

} // end namespace

#endif
