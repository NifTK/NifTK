/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkImageLookupTablesPreferencePage_h
#define niftkImageLookupTablesPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QSpinBox;

namespace niftk
{

/**
 * \class ImageLookupTablesPreferencePage
 * \brief Preferences page for this plugin, enabling choice of spin box precision.
 * \ingroup uk_ac_ucl_cmic_gui_imagelookuptables
 *
 */
class ImageLookupTablesPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  ImageLookupTablesPreferencePage();
  ImageLookupTablesPreferencePage(const ImageLookupTablesPreferencePage& other);
  ~ImageLookupTablesPreferencePage();

  static const QString PRECISION_NAME;

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

protected slots:

protected:

  QWidget* m_MainControl;
  QSpinBox* m_Precision;
  bool m_Initializing;

  berry::IPreferences::Pointer m_ImageLookupTablesPreferencesNode;

};

}

#endif

