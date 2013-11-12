/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDnDDisplayPreferencePage_h
#define niftkDnDDisplayPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <uk_ac_ucl_cmic_dnddisplay_Export.h>
#include <berryIPreferences.h>

class QWidget;
class QPushButton;
class QComboBox;
class QSpinBox;
class QCheckBox;

/**
 * \class niftkDnDDisplayPreferencePage
 * \brief Provides a preferences page for the NifTK DnD Display, including default number of rows,
 * default number of columns, image interpolation, default window layout and background colour.
 * \ingroup uk_ac_ucl_cmic_dnddisplay
 */
struct DNDDISPLAY_EXPORT niftkDnDDisplayPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  niftkDnDDisplayPreferencePage();
  niftkDnDDisplayPreferencePage(const niftkDnDDisplayPreferencePage& other);

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
  static const std::string DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE;

  /// \brief Stores the preference name for the default background colour in the DnD Display.
  static const std::string DNDDISPLAY_BACKGROUND_COLOUR;

  /// \brief Stores the preference name for the default background colour stylesheet in the NifTK DnD Display.
  static const std::string DNDDISPLAY_BACKGROUND_COLOUR_STYLESHEET;

  /// \brief Stores the preference name for slice select tracking
  static const std::string DNDDISPLAY_SLICE_SELECT_TRACKING;

  /// \brief Stores the preference name for time select tracking
  static const std::string DNDDISPLAY_TIME_SELECT_TRACKING;

  /// \brief Stores the preference name for magnification select tracking
  static const std::string DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING;

  /// \brief Stores the preference name for whether we show the 2D cursors as people may prefer them to always be off.
  static const std::string DNDDISPLAY_SHOW_2D_CURSORS;

  /// \brief Stores the preference name for whether we show the direction annotations.
  static const std::string DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS;

  /// \brief Stores the preference name for whether we show the 3D window in multiple window layout, as screen can get a bit cluttered.
  static const std::string DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT;

  /// \brief Stores the preference name for the default window layout in the NifTK DnD Display.
  static const std::string DNDDISPLAY_DEFAULT_WINDOW_LAYOUT;

  /// \brief Stores the preference name for whether to revert to last remembered slice, timestep and magnification when switching window layout.
  static const std::string DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT;

  /// \brief Stores the preference name for the default number of rows in the DnD Display.
  static const std::string DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER;

  /// \brief Stores the preference name for the default number of columns in the DnD Display.
  static const std::string DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER;

  /// \brief Stores the preference name for the default drop type (single, multiple, all).
  static const std::string DNDDISPLAY_DEFAULT_DROP_TYPE;

  /// \brief Stores the preference name for whether we show the magnification slider, as most people wont need it.
  static const std::string DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER;

  /// \brief Stores the preference name for whether we show the the options to show/hide cursor,
  /// direction annotations and 3D window in multi window layout.
  static const std::string DNDDISPLAY_SHOW_SHOWING_OPTIONS;

  /// \brief Stores the preference name for whether we show the window layout controls.
  static const std::string DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS;

  /// \brief Stores the preference name for whether we show the viewer number controls.
  static const std::string DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS;

  /// \brief Stores the preference name for a simple on/off preference for whether we show the single, multiple, all checkbox.
  static const std::string DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS;

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
  QCheckBox* m_RememberEachWindowLayoutsViewerSettings;
  QSpinBox* m_DefaultNumberOfViewerRowsSpinBox;
  QSpinBox* m_DefaultNumberOfViewerColumnsSpinBox;
  QComboBox* m_DefaultDropType;
  QCheckBox* m_ShowMagnificationSliderCheckBox;
  QCheckBox* m_ShowShowingOptionsCheckBox;
  QCheckBox* m_ShowWindowLayoutControlsCheckBox;
  QCheckBox* m_ShowViewerNumberControlsCheckBox;
  QCheckBox* m_ShowDropTypeControlsCheckBox;
  QPushButton* m_BackgroundColourButton;

  berry::IPreferences::Pointer m_DnDDisplayPreferencesNode;
};

#endif
