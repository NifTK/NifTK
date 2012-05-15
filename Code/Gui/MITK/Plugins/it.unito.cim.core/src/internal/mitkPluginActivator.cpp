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

#include "mitkPluginActivator.h"

#include <QtPlugin>

#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateAnd.h>
#include <QmitkNodeDescriptorManager.h>

//#include "ImageInfoRenderer.h"
//#include "GeometryRecalculator.h"
//#include "NodeVisibilityManager.h"

namespace mitk {

class PluginActivatorPrivate
{
public:
//  ImageInfoRenderer* imageInfoRenderer;
//  GeometryRecalculator* geometryRecalculator;
//  NodeVisibilityManager* nodeVisibilityManager;
};

PluginActivator::PluginActivator()
: d_ptr(new PluginActivatorPrivate)
{
}

PluginActivator::~PluginActivator()
{
}

void
PluginActivator::start(ctkPluginContext* context)
{
  Q_UNUSED(context);
  MITK_INFO << "Core plugin activated";

//  registerNodeDescriptors();

//  Q_D(PluginActivator);

//  d->imageInfoRenderer = new ImageInfoRenderer();
//  d->geometryRecalculator = new GeometryRecalculator();
//  d->nodeVisibilityManager = new NodeVisibilityManager();
}

void
PluginActivator::stop(ctkPluginContext* context)
{
  Q_UNUSED(context);
//  Q_D(PluginActivator);
//  delete d->nodeVisibilityManager;
//  delete d->geometryRecalculator;
//  delete d->imageInfoRenderer;
}

//void
//PluginActivator::registerNodeDescriptors()
//{
//  // Adding "Patient nodes"
//  QmitkNodeDescriptorManager* nodeDescriptorManager = QmitkNodeDescriptorManager::GetInstance();
//  mitk::NodePredicateData::Pointer isGroupNode = mitk::NodePredicateData::New(0);
//
//  mitk::NodePredicateProperty::Pointer hasPatientProperty = mitk::NodePredicateProperty::New("Patient");
//  mitk::NodePredicateAnd::Pointer isPatientNode = mitk::NodePredicateAnd::New(hasPatientProperty, isGroupNode);
//  QmitkNodeDescriptor* patientNodeDescriptor =
//      new QmitkNodeDescriptor(
//          tr("Patient"),
//          QString(":/it.unito.cim.core/patient-icon.png"),
//          isPatientNode,
//          this);
//  nodeDescriptorManager->AddDescriptor(patientNodeDescriptor);
//
//  mitk::NodePredicateProperty::Pointer hasStudyProperty = mitk::NodePredicateProperty::New("Study");
//  mitk::NodePredicateAnd::Pointer isStudyNode = mitk::NodePredicateAnd::New(hasStudyProperty, isGroupNode);
//  QmitkNodeDescriptor* studyNodeDescriptor =
//      new QmitkNodeDescriptor(
//          tr("Study"),
//          QString(":/it.unito.cim.core/study-icon.png"),
//          isStudyNode,
//          this);
//  nodeDescriptorManager->AddDescriptor(studyNodeDescriptor);
//
//  mitk::NodePredicateProperty::Pointer hasSequenceProperty = mitk::NodePredicateProperty::New("Sequence");
//  mitk::NodePredicateAnd::Pointer isSequenceNode = mitk::NodePredicateAnd::New(hasSequenceProperty, isGroupNode);
//  QmitkNodeDescriptor* sequenceNodeDescriptor =
//      new QmitkNodeDescriptor(
//          tr("Sequence"),
//          QString(":/it.unito.cim.core/sequence-icon.png"),
//          isSequenceNode,
//          this);
//  nodeDescriptorManager->AddDescriptor(sequenceNodeDescriptor);
//
//  mitk::NodePredicateProperty::Pointer hasFittingProperty = mitk::NodePredicateProperty::New("Fitting");
//  mitk::NodePredicateAnd::Pointer isFittingNode = mitk::NodePredicateAnd::New(hasFittingProperty, isGroupNode);
//  QmitkNodeDescriptor* fittingNodeDescriptor =
//      new QmitkNodeDescriptor(
//          tr("Fitting"),
//          QString(":/it.unito.cim.core/fitting-icon.png"),
//          isFittingNode,
//          this);
//  nodeDescriptorManager->AddDescriptor(fittingNodeDescriptor);
//
//  mitk::NodePredicateProperty::Pointer hasModelProperty = mitk::NodePredicateProperty::New("Model");
//  mitk::NodePredicateAnd::Pointer isModelNode = mitk::NodePredicateAnd::New(hasModelProperty, isGroupNode);
//  QmitkNodeDescriptor* modelNodeDescriptor =
//      new QmitkNodeDescriptor(
//          tr("Model"),
//          QString(":/it.unito.cim.core/model-icon.png"),
//          isModelNode,
//          this);
//  nodeDescriptorManager->AddDescriptor(modelNodeDescriptor);
//}

}

Q_EXPORT_PLUGIN2(it_unito_cim_core, mitk::PluginActivator)
