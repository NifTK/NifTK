/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCaffeSegmentorPreferencePage_h
#define niftkCaffeSegmentorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include "niftkCaffePreferencesWidget.h"

class QWidget;
class QPushButton;
class ctkPathLineEdit;
class QCheckBox;
class QLineEdit;
class QSpinBox;

namespace niftk
{

/// \class CaffeSegmentorPreferencePage
/// \brief Preferences page for this plugin.
/// \ingroup uk_ac_ucl_cmic_caffesegmentor_internal
class CaffeSegmentorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  CaffeSegmentorPreferencePage();
  CaffeSegmentorPreferencePage(const CaffeSegmentorPreferencePage& other);
  ~CaffeSegmentorPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  ///
  /// \see IPreferencePage::PerformOk()
  ///
  virtual bool PerformOk();

  ///
  /// \see IPreferencePage::PerformCancel()
  ///
  virtual void PerformCancel();

  ///
  /// \see IPreferencePage::Update()
  ///
  virtual void Update();

protected slots:

protected:

  QWidget*                    m_MainControl;
  bool                        m_Initializing;
  CaffePreferencesWidget*     m_CaffeSegPrefs;

  berry::IPreferences::Pointer m_CaffeSegmentorPreferencesNode;
};

}

#endif
