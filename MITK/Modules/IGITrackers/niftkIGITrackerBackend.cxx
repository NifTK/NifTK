/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITrackerBackend.h"
#include <niftkCoordinateAxesData.h>
#include <niftkMITKMathsUtils.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGITrackerBackend::IGITrackerBackend(QString name,
                                     mitk::DataStorage::Pointer dataStorage)
: m_Name(name)
, m_DataStorage(dataStorage)
, m_FrameId(0)
, m_Lag(0)
, m_ExpectedFramesPerSecond(0)
{
  m_CachedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  m_CachedTransform->Identity();
}


//-----------------------------------------------------------------------------
IGITrackerBackend::~IGITrackerBackend()
{
  if (m_DataStorage.IsNotNull())
  {
    std::set<mitk::DataNode::Pointer>::iterator iter;
    for (iter = m_DataNodes.begin(); iter != m_DataNodes.end(); ++iter)
    {
      m_DataStorage->Remove(*iter);
    }
  }
}


//-----------------------------------------------------------------------------
void IGITrackerBackend::SetProperties(const IGIDataSourceProperties& properties)
{
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Lag = milliseconds;

    MITK_INFO << "IGITrackerBackend(" << m_Name.toStdString()
              << "): set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties IGITrackerBackend::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Lag);

  MITK_INFO << "IGITrackerBackend:(" << m_Name.toStdString()
            << "): Retrieved current value of lag as " << m_Lag << " ms.";

  return props;
}


//-----------------------------------------------------------------------------
void IGITrackerBackend::WriteToDataStorage(const std::string& name,
                                           const niftk::IGITrackerDataType& transform)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "DataStorage is NULL!";
  }

  if (name.empty())
  {
    mitkThrow() << "Empty name.";
  }

  mitk::DataNode::Pointer node = m_DataStorage->GetNamedNode(name);
  if (node.IsNull())
  {
    node = mitk::DataNode::New();
    node->SetVisibility(true);
    node->SetOpacity(1);
    node->SetName(name);
    m_DataStorage->Add(node);
    m_DataNodes.insert(node);
  }

  CoordinateAxesData::Pointer coords = dynamic_cast<CoordinateAxesData*>(node->GetData());
  if (coords.IsNull())
  {
    coords = CoordinateAxesData::New();

    // We remove and add to trigger the NodeAdded event,
    // which is not emmitted if the node was added with no data.
    m_DataStorage->Remove(node);
    node->SetData(coords);
    m_DataStorage->Add(node);
  }

  mitk::Point4D rotation;
  mitk::Vector3D translation;
  transform.GetTransform(rotation, translation);
  niftk::ConvertRotationAndTranslationToMatrix(rotation, translation, *m_CachedTransform);
  coords->SetVtkMatrix(*m_CachedTransform);

  // We tell the node that it is modified so the next rendering event
  // will redraw it. Triggering this does not in itself guarantee a re-rendering.
  coords->Modified();
  node->Modified();
}

} // end namespace
