/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TagTrackerViewPreferencePage_h
#define TagTrackerViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QCheckBox;

/**
 * \class TagTrackerViewPreferencePage
 * \brief Preferences page for the Tag Tracker View plugin.
 * \ingroup uk_ac_ucl_cmic_igitagtracker_internal
 */
class TagTrackerViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  TagTrackerViewPreferencePage();
  TagTrackerViewPreferencePage(const TagTrackerViewPreferencePage& other);
  ~TagTrackerViewPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);
  void CreateQtControl(QWidget* widget);
  QWidget* GetQtControl() const;

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  /**
   * \brief Stores the minimum size of the tag as a fraction [0..1] of the maximum of the number of rows and columns.
   */
  static const float MIN_SIZE;

  /**
   * \brief Stores the name of the preferences node used to store the MIN_SIZE.
   */
  static const std::string MIN_SIZE_NAME;

  /**
   * \brief Stores the maximum size of the tag as a fraction [0..1] of the maximum of the number of rows and columns.
   */
  static const float MAX_SIZE;

  /**
   * \brief Stores the name of the preferences node used to store the MAX_SIZE.
   */
  static const std::string MAX_SIZE_NAME;

  /**
   * \brief Stores the boolean as to whether we listen to event bus.
   */
  static const bool LISTEN_TO_EVENT_BUS;

  /**
   * \brief Stores the name of the preferences node used to store LISTEN_TO_EVENT_BUS.
   */
  static const std::string LISTEN_TO_EVENT_BUS_NAME;

  /**
   * \brief Stores a boolean to force mono in left camera.
   */
  static const bool DO_MONO_LEFT_CAMERA;

  /**
   * \brief Stores the name of the preferences node used to store DO_MONO_LEFT_CAMERA.
   */
  static const std::string DO_MONO_LEFT_CAMERA_NAME;

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

private slots:

private:

  QWidget        *m_MainControl;
  QRadioButton   *m_ListenToEventBusPulse;
  QRadioButton   *m_ManualUpdate;
  QDoubleSpinBox *m_MinSize;
  QDoubleSpinBox *m_MaxSize;
  QCheckBox      *m_DoMonoLeftCamera;

  bool            m_Initializing;

  berry::IPreferences::Pointer m_TagTrackerViewPreferencesNode;
};

#endif // TagTrackerViewPreferencePage_h

