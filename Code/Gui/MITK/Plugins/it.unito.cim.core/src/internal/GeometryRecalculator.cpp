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

#include "GeometryRecalculator.h"

#include <itkSimpleDataObjectDecorator.h>

#include <mitkRenderingManager.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateSource.h>
#include <mitkGenericProperty.h>

#include <QmitkStdMultiWidget.h>

class GeometryRecalculatorPrivate
{
public:
  mitk::DataStorage::Pointer dataStorage;
  mitk::NodePredicateNot::Pointer pred;
  mitk::RenderingManager* renderingManager;

  QmitkStdMultiWidget* display;

  std::vector<mitk::DataNode*> nodes;
};


GeometryRecalculator::GeometryRecalculator()
: d_ptr(new GeometryRecalculatorPrivate)
{
  Q_D(GeometryRecalculator);

  d->renderingManager = mitk::RenderingManager::GetInstance();
  d->dataStorage = GetDataStorage();

  d->display = 0;

  d->pred =
    mitk::NodePredicateNot::New(
        mitk::NodePredicateProperty::New("includeInBoundingBox",
            mitk::BoolProperty::New(false)));
}

GeometryRecalculator::~GeometryRecalculator()
{
}

void
GeometryRecalculator::init()
{
  PluginCore::init();
}

void
GeometryRecalculator::onNodeAdded(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    recalculateGeometry(node);
  }
}

void
GeometryRecalculator::onVisibilityChanged(const mitk::DataNode* node)
{
  recalculateGeometry(node);
}

void
GeometryRecalculator::recalculateGeometry(const mitk::DataNode* node)
{
  static unsigned lastTimeSteps = 0;

  // Skip if the added/removed/changed node is not visible, it is
  // not to be included in the bounding box calculation (special
  // attribute used inside MITK) or it is a helper object
  if (!node->IsVisible(0) ||
      !node->GetData() ||
      !node->IsOn("includeInBoundingBox", 0, true) ||
      node->IsOn("helper object", 0, false))
  {
    return;
  }

  Q_D(GeometryRecalculator);

  // get all nodes that have not set "includeInBoundingBox" to false
  mitk::DataStorage::SetOfObjects::ConstPointer rs = d->dataStorage->GetSubset(d->pred);
  // calculate bounding geometry of these nodes
  mitk::TimeSlicedGeometry::Pointer bounds = d->dataStorage->ComputeBoundingGeometry3D(rs, "visible");
  if (bounds.IsNull()) {
    MITK_INFO << __FILE__ << ":" << __LINE__ << ": "
        "No nodes were included in bounding box computation." << std::endl;
    return;
  }

  // This is a bad hack. We expect that the other dimensions remain the same,
  // and recalculate the geometry only if the number of time steps differ.
  // This should be fixed inside the Image Navigator, anyway.
  unsigned timeSteps = bounds->GetTimeSteps();
  if (timeSteps != lastTimeSteps) {
    // initialize the views to the bounding geometry
    mitk::RenderingManager::GetInstance()->InitializeViews(bounds);
    lastTimeSteps = timeSteps;
  }
}
