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


#ifndef QMITKNIFTYVIEWMITKPERSPECTIVE_H_
#define QMITKNIFTYVIEWMITKPERSPECTIVE_H_

#include <berryIPerspectiveFactory.h>

/**
 * \class QmitkNiftyViewMITKPerspective
 * \brief Perspective to arrange widgets, primarily for MITK style segmentation.
 * \ingroup uk_ac_ucl_cmic_gui_qt_niftyview_internal
 *
 * Note: We have to load at least one view component, to get an editor created.
 */
class QmitkNiftyViewMITKPerspective : public QObject, public berry::IPerspectiveFactory
{
  Q_OBJECT
  Q_INTERFACES(berry::IPerspectiveFactory)

public:

  QmitkNiftyViewMITKPerspective();
  QmitkNiftyViewMITKPerspective(const QmitkNiftyViewMITKPerspective& other);

  void CreateInitialLayout(berry::IPageLayout::Pointer layout);

};

#endif /* QMITKNIFTYVIEWMITKPERSPECTIVE_H_ */
