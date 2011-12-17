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
#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;
class QComboBox;
class QSpinBox;

/**
 * \class QmitkMIDASMultiViewEditorPreferencePage
 * \brief Provides a preferences page for the CMIC MIDAS Display, including default number of rows,
 * default number of columns, image interpolation, default orientation and background colour.
 */
struct CMIC_QT_COMMON QmitkMIDASMultiViewEditorPreferencePage : public QObject, public berry::IQtPreferencePage
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

  /// \brief Stores the preference name for the default orientation in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_ORIENTATION;

  /// \brief Stores the preference name for the default image interpolation in the CMIC MIDAS Display.
  static const std::string MIDAS_DEFAULT_IMAGE_INTERPOLATION;

  /// \brief Stores the preference name for the default background colour in the CMIC MIDAS Display.
  static const std::string MIDAS_BACKGROUND_COLOUR;

  /// \brief Stores the preference name for the default background colour stylesheet in the CMIC MIDAS Display.
  static const std::string MIDAS_BACKGROUND_COLOUR_STYLESHEET;

public slots:

  void OnBackgroundColourChanged();
  void OnResetBackgroundColour();

private:

  QWidget* m_MainControl;

  QString      m_BackgroundColorStyleSheet;
  std::string  m_BackgroundColor;

  QSpinBox    *m_DefaultNumberOfRowsSpinBox;
  QSpinBox    *m_DefaultNumberOfColumnsSpinBox;
  QComboBox   *m_DefaultOrientationComboBox;
  QComboBox   *m_ImageInterpolationComboBox;
  QPushButton *m_BackgroundColourButton;

  berry::IPreferences::Pointer m_MIDASMultiViewEditorPreferencesNode;
};

#endif /* QMITKMIDASMULTIVIEWEDITORPREFERENCEPAGE_H */
