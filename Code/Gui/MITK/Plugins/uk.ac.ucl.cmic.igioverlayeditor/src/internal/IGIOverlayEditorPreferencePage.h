/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditorPreferencePage_h
#define IGIOverlayEditorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QCheckBox;
class QPushButton;
class QWidgetAction;
class ctkPathLineEdit;

/**
 * \class IGIOverlayEditorPreferencePage
 * \brief Preference page for IGIOverlayEditor, currently setting the gradient background.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor
 */
struct IGIOverlayEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  IGIOverlayEditorPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);
  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  /**
   * \see IPreferencePage::PerformOk()
   */
  virtual bool PerformOk();

  /**
   * \see IPreferencePage::PerformCancel()
   */
  virtual void PerformCancel();

  /**
   * \see IPreferencePage::Update()
   */
  virtual void Update();

  /**
   * \brief Stores the name of the preference node that contains the stylesheet of the first background colour.
   */
  static const std::string FIRST_BACKGROUND_STYLE_SHEET;

  /**
   * \brief Stores the name of the preference node that contains the stylesheet of the second background colour.
   */
  static const std::string SECOND_BACKGROUND_STYLE_SHEET;

  /**
   * \brief Stores the name of the preference node that contains the first background colour.
   */
  static const std::string FIRST_BACKGROUND_COLOUR;

  /**
   * \brief Stores the name of the preference node that contains the second background colour.
   */
  static const std::string SECOND_BACKGROUND_COLOUR;

  /**
   * \brief Stores the name of the preference node containing the filename of the calibration (eg. hand-eye for a laparoscope).
   */
  static const std::string CALIBRATION_FILE_NAME;

public slots:

  void FirstColorChanged();
  void SecondColorChanged();
  void ResetColors();

protected:

  QWidget*   m_MainControl;
  QPushButton* m_ColorButton1;
  QPushButton* m_ColorButton2;
  ctkPathLineEdit *m_CalibrationFileName;
  std::string m_FirstColor;
  std::string m_SecondColor;
  QString m_FirstColorStyleSheet;
  QString m_SecondColorStyleSheet;

  berry::IPreferences::Pointer m_IGIOverlayEditorPreferencesNode;
};

#endif /* IGIOverlayEditorPreferencePage_h */
