/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkImageLookupTablesPreferencePage_h
#define QmitkImageLookupTablesPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QSpinBox;

/**
 * \class QmitkImageLookupTablesPreferencePage
 * \brief Preferences page for this plugin, enabling choice of spin box precision.
 * \ingroup uk_ac_ucl_cmic_gui_imagelookuptables
 *
 */
class QmitkImageLookupTablesPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  QmitkImageLookupTablesPreferencePage();
  QmitkImageLookupTablesPreferencePage(const QmitkImageLookupTablesPreferencePage& other);
  ~QmitkImageLookupTablesPreferencePage();

  static const std::string PRECISION_NAME;

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

protected slots:

protected:

  QWidget* m_MainControl;
  QSpinBox* m_Precision;
  bool m_Initializing;

  berry::IPreferences::Pointer m_ImageLookupTablesPreferencesNode;
};

#endif

