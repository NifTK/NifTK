/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASMORPHOLOGICALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED
#define _MIDASMORPHOLOGICALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED

#include "berryIQtPreferencePage.h"
#include "uk_ac_ucl_cmic_midasmorphologicalsegmentor_Export.h"
#include <berryIPreferences.h>

class QWidget;
class QPushButton;

/**
 * \class MIDASMorphologicalSegmentorViewPreferencePage
 * \brief Preferences page for this plugin, enabling choice volume rendering on/off.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor
 *
 */
class MIDASMORPHOLOGICALSEGMENTOR_EXPORTS MIDASMorphologicalSegmentorViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  MIDASMorphologicalSegmentorViewPreferencePage();
  MIDASMorphologicalSegmentorViewPreferencePage(const MIDASMorphologicalSegmentorViewPreferencePage& other);
  ~MIDASMorphologicalSegmentorViewPreferencePage();

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

  berry::IPreferences::Pointer m_MIDASMorphologicalSegmentorViewPreferencesNode;
};

#endif /* _MIDASMORPHOLOGICALSEGMENTORVIEWPREFERENCEPAGE_H_INCLUDED */

