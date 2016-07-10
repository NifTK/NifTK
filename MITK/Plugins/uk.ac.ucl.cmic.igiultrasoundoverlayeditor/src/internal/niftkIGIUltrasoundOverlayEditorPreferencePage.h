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
class QRadioButton;
class QPushButton;
class QWidgetAction;
class QCheckBox;
class ctkPathLineEdit;

namespace niftk
{

/**
 * \class IGIOverlayEditorPreferencePage
 * \brief Preference page for IGIUltrasoundOverlayEditor.
 * \ingroup uk_ac_ucl_cmic_igiultrasoundoverlayeditor_internal
 */
struct IGIUltrasoundOverlayEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  IGIUltrasoundOverlayEditorPreferencePage();

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
   * \brief Stores the name of the preference node that contains the stylesheet of the first background colour.
   */
  static const QString FIRST_BACKGROUND_STYLE_SHEET;

  /**
   * \brief Stores the name of the preference node that contains the stylesheet of the second background colour.
   */
  static const QString SECOND_BACKGROUND_STYLE_SHEET;

  /**
   * \brief Stores the name of the preference node that contains the first background colour.
   */
  static const QString FIRST_BACKGROUND_COLOUR;

  /**
   * \brief Stores the name of the preference node that contains the second background colour.
   */
  static const QString SECOND_BACKGROUND_COLOUR;

  /**
   * \brief Stores the name of the preference node containing whether we are using clipping planes in Image Tracking mode.
   */  
  static const QString CLIP_TO_IMAGE_PLANE;
  
public slots:

  void FirstColorChanged();
  void SecondColorChanged();
  void ResetColors();

protected:

  QWidget         *m_MainControl;
  QCheckBox       *m_ClipToImagePlane;
  QPushButton     *m_ColorButton1;
  QPushButton     *m_ColorButton2;
  QString          m_FirstColor;
  QString          m_SecondColor;
  QString          m_FirstColorStyleSheet;
  QString          m_SecondColorStyleSheet;

  berry::IPreferences::Pointer m_IGIUltrasoundOverlayEditorPreferencesNode;
};

} // end namespace

#endif
