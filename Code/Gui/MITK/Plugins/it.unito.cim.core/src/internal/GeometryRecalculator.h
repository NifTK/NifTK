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

#ifndef __GeometryRecalculator_h
#define __GeometryRecalculator_h

#include <QObject>
#include "PluginCore.h"

class GeometryRecalculatorPrivate;

namespace mitk {
class DataNode;
}

class GeometryRecalculator : public PluginCore
{
public:

  explicit GeometryRecalculator();
  virtual ~GeometryRecalculator();

  virtual void onNodeAdded(const mitk::DataNode* node);
  virtual void onVisibilityChanged(const mitk::DataNode* node);

  void recalculateGeometry(const mitk::DataNode* node);

protected:
  virtual void init();

private:
  QScopedPointer<GeometryRecalculatorPrivate> d_ptr;

  Q_DECLARE_PRIVATE(GeometryRecalculator);
  Q_DISABLE_COPY(GeometryRecalculator);
};

#endif
