/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _IMAGESTATISTICSVIEWPREFERENCESPAGE_H_INCLUDED
#define _IMAGESTATISTICSVIEWPREFERENCESPAGE_H_INCLUDED

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QCheckBox;
class QSpinBox;

/**
 * \class ImageStatisticsViewPreferencesPage
 * \brief Preference page for Image Statistics view, providing checkboxes for "automatic update", "assume binary image",
 * and "require same size image" etc.
 *
 * \ingroup uk_ac_ucl_cmic_imagestatistics_internal
 *
 */
class ImageStatisticsViewPreferencesPage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  ImageStatisticsViewPreferencesPage();
  ImageStatisticsViewPreferencesPage(const ImageStatisticsViewPreferencesPage& other);
  ~ImageStatisticsViewPreferencesPage();

  static const std::string AUTO_UPDATE_NAME;
  static const std::string ASSUME_BINARY_NAME;
  static const std::string REQUIRE_SAME_SIZE_IMAGE_NAME;
  static const std::string BACKGROUND_VALUE_NAME;

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

  QWidget*   m_MainControl;
  QCheckBox* m_AutoUpdate;
  QCheckBox* m_AssumeBinary;
  QCheckBox* m_RequireSameSizeImage;
  QSpinBox*  m_BackgroundValue;
  berry::IPreferences::Pointer m_ImageStatisticsPreferencesNode;
};

#endif /* QMITKDATAMANAGERPREFERENCEPAGE_H_ */

