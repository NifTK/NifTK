/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-02 06:46:00 +0000 (Fri, 02 Dec 2011) $
 Revision          : $Revision: 7905 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMULTIVIEWEDITORPREFERENCEPAGE_H
#define QMITKMIDASMULTIVIEWEDITORPREFERENCEPAGE_H

#include "berryIQtPreferencePage.h"
#include <uk_ac_ucl_cmic_midaseditor_Export.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;
class QComboBox;
class QSpinBox;
class QCheckBox;

/**
 * \class QmitkMIDASMultiViewEditorPreferencePage
 * \brief Provides a preferences page for the CMIC MIDAS Display, including default number of rows,
 * default number of columns, image interpolation, default view and background colour.
 * \ingroup uk_ac_ucl_cmic_midaseditor
 */
struct MIDASEDITOR_EXPORT QmitkMIDASMultiViewEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  QmitkMIDASMultiViewEditorPreferencePage();
  QmitkMIDASMultiViewEditorPreferencePage(const QmitkMIDASMultiViewEditorPreferencePage& other);

  void CreateQtControl(QWidget* widget);
  QWidget* GetQtControl() const;

  /// \brief Nothing to do.
  void Init(berry::IWorkbench::Pointer workbench);

  /// \see IPreferencePage::PerformOk()
  virtual bool PerformOk();

  /// \see IPreferencePage::PerformCancel()
  virtual void PerformCancel();

  /// \see IPreferencePage::Update()
  virtual void Update();

  /// \brief Stores the preference name for the default number of rows in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_NUMBER_ROWS;

  /// \brief Stores the preference name for the default number of columns in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_NUMBER_COLUMNS;

  /// \brief Stores the preference name for the default view in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_VIEW;

  /// \brief Stores the preference name for the default image interpolation in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_IMAGE_INTERPOLATION;

  /// \brief Stores the preference name for the default drop type (single, multiple, all).
  static const std::string MIDAS_DEFAULT_DROP_TYPE;

  /// \brief Stores the preference name for a simple on/off preference for whether we show the single, multiple, all checkbox.
  static const std::string MIDAS_SHOW_DROP_TYPE_WIDGETS;

  /// \brief Stores the preference name for whether we show the 3D view in orthoview, as screen can get a bit cluttered.
  static const std::string MIDAS_SHOW_3D_VIEW_IN_ORTHOVIEW;

  /// \brief Stores the preference name for whether we show the 2D cursors as people may prefer them to always be off.
  static const std::string MIDAS_SHOW_2D_CURSORS;

  /// \brief Stores the preference name for whether we show the magnification slider, as most people wont need it.
  static const std::string MIDAS_SHOW_MAGNIFICATION_SLIDER;

  /// \brief Stores the preference name for the default background colour in the CMIC MIDAS Display.
  static const std::string MIDAS_BACKGROUND_COLOUR;

  /// \brief Stores the preference name for the default background colour stylesheet in the CMIC MIDAS Display.
  static const std::string MIDAS_BACKGROUND_COLOUR_STYLESHEET;

  /// \brief Stores the preference name for whether we show the layout buttons.
  static const std::string MIDAS_SHOW_LAYOUT_BUTTONS;

  /// \brief Stores the preference name for whether we adopt MIDAS behaviour when switching orientation to revert to last remembered slice, timestep, magnification.
  static const std::string MIDAS_REMEMBER_VIEW_SETTINGS_PER_ORIENTATION;

public slots:

  void OnBackgroundColourChanged();
  void OnResetBackgroundColour();
  void OnResetMIDASBackgroundColour();

private:

  QWidget* m_MainControl;

  QString      m_BackgroundColorStyleSheet;
  std::string  m_BackgroundColor;

  QSpinBox    *m_DefaultNumberOfRowsSpinBox;
  QSpinBox    *m_DefaultNumberOfColumnsSpinBox;
  QComboBox   *m_DefaultViewComboBox;
  QComboBox   *m_ImageInterpolationComboBox;
  QComboBox   *m_DefaultDropType;
  QCheckBox   *m_ShowDropTypeWidgetsCheckBox;
  QCheckBox   *m_ShowLayoutButtonsCheckBox;
  QCheckBox   *m_ShowMagnificationSliderCheckBox;
  QCheckBox   *m_Show3DInOrthoCheckBox;
  QCheckBox   *m_Show2DCursorsCheckBox;
  QCheckBox   *m_RememberEachOrientationsViewSettings;
  QPushButton *m_BackgroundColourButton;

  berry::IPreferences::Pointer m_MIDASMultiViewEditorPreferencesNode;
};

#endif /* QMITKMIDASMULTIVIEWEDITORPREFERENCEPAGE_H */
