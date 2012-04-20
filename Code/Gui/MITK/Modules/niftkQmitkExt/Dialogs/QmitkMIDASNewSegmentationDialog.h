/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASNEWSEGMENTATIONDIALOG_H_
#define QMITKMIDASNEWSEGMENTATIONDIALOG_H_

#include "niftkQmitkExtExports.h"
#include "QmitkNewSegmentationDialog.h"
#include <QColor>

/**
 * \class QmitkMIDASNewSegmentationDialog
 * \brief Derives from QmitkNewSegmentationDialog, to simply set the default colour to pure green.
 * \sa QmitkNewSegmentationDialog
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASNewSegmentationDialog : public QmitkNewSegmentationDialog
{
  Q_OBJECT

public:

  /// \brief Constructor, which sets the default button colour to that given by defaultColor.
  QmitkMIDASNewSegmentationDialog(const QColor &defaultColor, QWidget* parent = 0);
  ~QmitkMIDASNewSegmentationDialog() {}
};
#endif /*QMITKMIDASNEWSEGMENTATIONDIALOG_H_*/
