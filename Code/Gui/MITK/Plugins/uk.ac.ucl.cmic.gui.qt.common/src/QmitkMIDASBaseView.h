/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak <espakm@gmail.com>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __QmitkMIDASBaseView_h
#define __QmitkMIDASBaseView_h

#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>

#include <QmitkAbstractView.h>

#include "internal/VisibilityChangeObserver.h"

class QmitkMIDASBaseViewPrivate;

namespace mitk {
class DataNode;
}


/**
 * \class QmitkMIDASBaseView
 * \brief Base view component for plugins listening to visibility change events.
 *
 * This is a legacy code from the KMaps project. It should be merged into the QmitkMIDASBaseFunctionality class.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkMIDASBaseView : public QmitkAbstractView, public VisibilityChangeObserver
{
  Q_OBJECT

public:
  typedef QmitkAbstractView SuperClass;

  explicit QmitkMIDASBaseView();
  virtual ~QmitkMIDASBaseView();

  ///
  /// Called when the visibility of a node in the data storage changed
  ///
  virtual void onVisibilityChanged(const mitk::DataNode* node);

  virtual bool IsExclusiveFunctionality() const;

private:
  void onNodeAddedInternal(const mitk::DataNode*);
  void onNodeRemovedInternal(const mitk::DataNode*);

  QScopedPointer<QmitkMIDASBaseViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(QmitkMIDASBaseView);
  Q_DISABLE_COPY(QmitkMIDASBaseView);
};

#endif
