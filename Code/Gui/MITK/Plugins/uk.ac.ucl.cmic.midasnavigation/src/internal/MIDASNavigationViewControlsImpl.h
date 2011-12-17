/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 18:07:46 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7922 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MIDASNAVIGATIONVIEWCONTROLSIMPL_H
#define MIDASNAVIGATIONVIEWCONTROLSIMPL_H

#include "ui_MIDASNavigationViewControls.h"
#include "MagnificationFactorWidget.h"
#include "IntegerSpinBoxAndSliderWidget.h"

class QAbstractButton;
class QGridLayout;
class QVBoxLayout;

/**
 * \class MIDASNavigationViewControlsImpl
 * \brief Implements a few Qt specific things that are of no interest to the view class.
 * \ingroup uk_ac_ucl_cmic_midasnavigation_internal
 */
class MIDASNavigationViewControlsImpl : public QWidget, public Ui_MIDASNavigationViewControls
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  MIDASNavigationViewControlsImpl();

  /** Destructor. */
  ~MIDASNavigationViewControlsImpl();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

signals:

protected slots:

protected:

private:

};

#endif
