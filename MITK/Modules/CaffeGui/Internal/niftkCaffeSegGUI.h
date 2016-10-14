/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkCaffeSegGUI_h
#define niftkCaffeSegGUI_h

#include <QWidget>
#include "ui_niftkCaffeSegGUI.h"
#include <niftkBaseGUI.h>
#include <mitkDataStorage.h>

namespace niftk
{

/// \class CaffeSegGUI
/// \brief Widgets for Caffe segmentation tester plugin.
class CaffeSegGUI : public BaseGUI,
                    private Ui::CaffeSegGUI
{

  Q_OBJECT

public:

  CaffeSegGUI(QWidget* parent);
  virtual ~CaffeSegGUI();

  void SetDataStorage(mitk::DataStorage* storage);

signals:

  void OnLeftSelectionChanged(const mitk::DataNode*);
  void OnRightSelectionChanged(const mitk::DataNode*);
  void OnDoItNowPressed();
  void OnManualUpdateClicked(bool);
  void OnAutomaticUpdateClicked(bool);

private:

  CaffeSegGUI(const CaffeSegGUI&);  // Purposefully not implemented.
  void operator=(const CaffeSegGUI&);  // Purposefully not implemented.
};

} // end namespace

#endif
