/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGeneralSegmentorViewPreferencePage_h
#define MIDASGeneralSegmentorViewPreferencePage_h

#include "berryIQtPreferencePage.h"
#include <berryIPreferences.h>

class QWidget;
class QPushButton;

/**
 * \class MIDASGeneralSegmentorViewPreferencePage
 * \brief Preferences page for this plugin, enabling the choice of the default colour of the segmentation.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor
 *
 */
class MIDASGeneralSegmentorViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  MIDASGeneralSegmentorViewPreferencePage();
  MIDASGeneralSegmentorViewPreferencePage(const MIDASGeneralSegmentorViewPreferencePage& other);
  ~MIDASGeneralSegmentorViewPreferencePage();

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

  void OnDefaultColourChanged();
  void OnResetDefaultColour();

protected:

  QWidget        *m_MainControl;
  QPushButton    *m_DefaultColorPushButton;
  QString         m_DefauleColorStyleSheet;
  std::string     m_DefaultColor;

  bool m_Initializing;

  berry::IPreferences::Pointer m_MIDASGeneralSegmentorViewPreferencesNode;
};

#endif
