/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED

#include "berryIQtPreferencePage.h"
#include "uk_ac_ucl_cmic_midasgeneralsegmentor_Export.h"
#include <berryIPreferences.h>

class QWidget;
class QPushButton;

/**
 * \class MIDASGeneralSegmentorViewPreferencePage
 * \brief Preferences page for this plugin, enabling the choice of the default colour of the segmentation.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor
 *
 */
class MIDASGENERALSEGMENTOR_EXPORTS MIDASGeneralSegmentorViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  MIDASGeneralSegmentorViewPreferencePage();
  MIDASGeneralSegmentorViewPreferencePage(const MIDASGeneralSegmentorViewPreferencePage& other);
  ~MIDASGeneralSegmentorViewPreferencePage();

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

  void OnDefaultColourChanged();
  void OnResetDefaultColour();

protected:

  QWidget        *m_MainControl;
  QPushButton    *m_DefaultColorPushButton;
  QString         m_DefauleColorStyleSheet;
  std::string     m_DefaultColor;

  bool m_Initializing;

  berry::IPreferences::Pointer m_MIDASGeneralSegmentorViewPreferencesNode;
};

#endif /* _MIDASGENERALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED */

