/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIVLVideoOverlayEditorPreferencePage_h
#define niftkIGIVLVideoOverlayEditorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;
class ctkPathLineEdit;

namespace niftk
{

/**
 * \class IGIVLVideoOverlayEditorPreferencePage
 * \brief Preference page for IGIVLVideoOverlayEditor, eg. setting the gradient background.
 * \ingroup uk_ac_ucl_cmic_IGIVLVideoOverlayeditor_internal
 */
struct IGIVLVideoOverlayEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  IGIVLVideoOverlayEditorPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench) override;
  void CreateQtControl(QWidget* widget) override;

  QWidget* GetQtControl() const override;

  /**
   * \see IPreferencePage::PerformOk()
   */
  virtual bool PerformOk() override;

  /**
   * \see IPreferencePage::PerformCancel()
   */
  virtual void PerformCancel() override;

  /**
   * \see IPreferencePage::Update()
   */
  virtual void Update() override;

  /**
   * \brief Stores the name of the preference node that contains the name of the calibration file.
   */
  static const QString CALIBRATION_FILE_NAME;

  /**
   * \brief Stores the name of the preference node that contains the background colour.
   */
  static const QString BACKGROUND_COLOR_PREFSKEY;
  static const unsigned int DEFAULT_BACKGROUND_COLOR;

public slots:

  void OnBackgroundColourClicked();

private:

  QWidget                      *m_MainControl;
  ctkPathLineEdit              *m_CalibrationFileName;
  QPushButton                  *m_BackgroundColourButton;
  unsigned int                  m_BackgroundColour;
  berry::IPreferences::Pointer  m_IGIVLVideoOverlayEditorPreferencesNode;
};

} // end namespace

#endif
