/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerEditorPreferencePage_h
#define niftkMultiViewerEditorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <uk_ac_ucl_cmic_dnddisplay_Export.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;
class QComboBox;
class QSpinBox;
class QCheckBox;

/**
 * \class niftkMultiViewerEditorPreferencePage
 * \brief Provides a preferences page for the NifTK DnD Display, including default number of rows,
 * default number of columns, image interpolation, default view and background colour.
 * \ingroup uk_ac_ucl_cmic_dnddisplay
 */
struct DNDDISPLAY_EXPORT niftkMultiViewerEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  niftkMultiViewerEditorPreferencePage();
  niftkMultiViewerEditorPreferencePage(const niftkMultiViewerEditorPreferencePage& other);

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

  /// \brief Stores the preference name for the default image interpolation in the DnD Display.
  static const std::string DEFAULT_INTERPOLATION_TYPE;

  /// \brief Stores the preference name for the default background colour in the DnD Display.
  static const std::string MIDAS_BACKGROUND_COLOUR;

  /// \brief Stores the preference name for the default background colour stylesheet in the NifTK DnD Display.
  static const std::string MIDAS_BACKGROUND_COLOUR_STYLESHEET;

  /// \brief Stores the preference name for slice select tracking
  static const std::string MIDAS_SLICE_SELECT_TRACKING;

  /// \brief Stores the preference name for time select tracking
  static const std::string MIDAS_TIME_SELECT_TRACKING;

  /// \brief Stores the preference name for magnification select tracking
  static const std::string MIDAS_MAGNIFICATION_SELECT_TRACKING;

  /// \brief Stores the preference name for whether we show the 2D cursors as people may prefer them to always be off.
  static const std::string MIDAS_SHOW_2D_CURSORS;

  /// \brief Stores the preference name for whether we show the direction annotations.
  static const std::string MIDAS_SHOW_DIRECTION_ANNOTATIONS;

  /// \brief Stores the preference name for whether we show the 3D window in multiple window layout, as screen can get a bit cluttered.
  static const std::string MIDAS_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT;

  /// \brief Stores the preference name for the default view in the NifTK DnD Display.
  static const std::string MIDAS_DEFAULT_WINDOW_LAYOUT;

  /// \brief Stores the preference name for whether we adopt MIDAS behaviour when switching orientation to revert to last remembered slice, timestep, magnification.
  static const std::string MIDAS_REMEMBER_VIEW_SETTINGS_PER_WINDOW_LAYOUT;

  /// \brief Stores the preference name for the default number of rows in the NifTK DnD Display.
  static const std::string MIDAS_DEFAULT_VIEW_ROW_NUMBER;

  /// \brief Stores the preference name for the default number of columns in the NifTK DnD Display.
  static const std::string MIDAS_DEFAULT_VIEW_COLUMN_NUMBER;

  /// \brief Stores the preference name for the default drop type (single, multiple, all).
  static const std::string DEFAULT_DROP_TYPE;

  /// \brief Stores the preference name for whether we show the magnification slider, as most people wont need it.
  static const std::string MIDAS_SHOW_MAGNIFICATION_SLIDER;

  /// \brief Stores the preference name for whether we show the the options to show/hide cursor,
  /// direction annotations and 3D window in multi window layout.
  static const std::string MIDAS_SHOW_SHOWING_OPTIONS;

  /// \brief Stores the preference name for whether we show the window layout controls.
  static const std::string MIDAS_SHOW_WINDOW_LAYOUT_CONTROLS;

  /// \brief Stores the preference name for whether we show the view number controls.
  static const std::string MIDAS_SHOW_VIEW_NUMBER_CONTROLS;

  /// \brief Stores the preference name for a simple on/off preference for whether we show the single, multiple, all checkbox.
  static const std::string MIDAS_SHOW_DROP_TYPE_CONTROLS;

public slots:

  void OnBackgroundColourChanged();
  void OnResetBackgroundColour();
  void OnResetMIDASBackgroundColour();

private:

  QWidget* m_MainControl;

  QString m_BackgroundColorStyleSheet;
  std::string m_BackgroundColor;

  QComboBox* m_ImageInterpolationComboBox;
  QCheckBox* m_SliceSelectTracking;
  QCheckBox* m_TimeSelectTracking;
  QCheckBox* m_MagnificationSelectTracking;
  QCheckBox* m_Show2DCursorsCheckBox;
  QCheckBox* m_ShowDirectionAnnotationsCheckBox;
  QCheckBox* m_Show3DWindowInMultiWindowLayoutCheckBox;
  QComboBox* m_DefaultWindowLayoutComboBox;
  QCheckBox* m_RememberEachWindowLayoutsViewSettings;
  QSpinBox* m_DefaultNumberOfViewRowsSpinBox;
  QSpinBox* m_DefaultNumberOfViewColumnsSpinBox;
  QComboBox* m_DefaultDropType;
  QCheckBox* m_ShowMagnificationSliderCheckBox;
  QCheckBox* m_ShowShowingOptionsCheckBox;
  QCheckBox* m_ShowWindowLayoutControlsCheckBox;
  QCheckBox* m_ShowViewNumberControlsCheckBox;
  QCheckBox* m_ShowDropTypeControlsCheckBox;
  QPushButton* m_BackgroundColourButton;

  berry::IPreferences::Pointer m_MultiViewerEditorPreferencesNode;
};

#endif
