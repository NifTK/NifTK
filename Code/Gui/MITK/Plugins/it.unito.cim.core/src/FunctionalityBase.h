/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __FunctionalityBase_h
#define __FunctionalityBase_h

#include "it_unito_cim_core_Export.h"

#include <QmitkAbstractView.h>

#include "internal/VisibilityChangeObserver.h"

class FunctionalityBasePrivate;

namespace mitk {
class DataNode;
}

class CIM_CORE_EXPORT FunctionalityBase : public QmitkAbstractView, public VisibilityChangeObserver
{
  Q_OBJECT

public:
  typedef QmitkAbstractView SuperClass;

  explicit FunctionalityBase();
  virtual ~FunctionalityBase();

  ///
  /// Called when the visibility of a node in the data storage changed
  ///
  virtual void onVisibilityChanged(const mitk::DataNode* node);

  virtual bool IsExclusiveFunctionality() const;

//protected:
//  void AfterCreateQtPartControl();

private:
  void onNodeAddedInternal(const mitk::DataNode*);
  void onNodeRemovedInternal(const mitk::DataNode*);

  QScopedPointer<FunctionalityBasePrivate> d_ptr;

  Q_DECLARE_PRIVATE(FunctionalityBase);
  Q_DISABLE_COPY(FunctionalityBase);
};

#endif
