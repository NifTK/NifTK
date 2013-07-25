/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIUtils.h"
#include <QFile>
#include <QMessageBox>
#include <mitkSTLFileReader.h>
#include <mitkFileIOUtils.h>
#include <mitkDataStorageUtils.h>
#include <igtlStringMessage.h>
#include <NiftyLinkSocketObject.h>
#include <Common/NiftyLinkXMLBuilder.h>


//-----------------------------------------------------------------------------
mitk::Surface::Pointer LoadSurfaceFromSTLFile(const QString& surfaceFilename)
{
  mitk::Surface::Pointer toolSurface;

  QFile surfaceFile(surfaceFilename);

  if(surfaceFile.exists())
  {
    mitk::STLFileReader::Pointer stlReader = mitk::STLFileReader::New();

    try
    {
      stlReader->SetFileName(surfaceFilename.toStdString().c_str());
      stlReader->Update();//load surface
      toolSurface = stlReader->GetOutput();
    }
    catch (std::exception& e)
    {
      MBI_ERROR<<"Could not load surface for tool!";
      MBI_ERROR<< e.what();
      throw e;
    }
  }

  return toolSurface;
}


//-----------------------------------------------------------------------------
QString CreateTestDeviceDescriptor()
{
  TrackerClientDescriptor tcld;
  tcld.SetDeviceName("NDI Polaris Vicra");
  tcld.SetDeviceType("Tracker");
  tcld.SetCommunicationType("Serial");
  tcld.SetPortName("Tracker not connected");
  tcld.SetClientIP(GetLocalHostAddress());
  tcld.SetClientPort(QString::number(3200));
  //tcld.AddTrackerTool("8700302.rom");
  tcld.AddTrackerTool("8700338.rom");
  //tcld.AddTrackerTool("8700339.rom");
  tcld.AddTrackerTool("8700340.rom");

  return tcld.GetXMLAsString();
}


//-----------------------------------------------------------------------------
bool SaveMatrixToFile(const vtkMatrix4x4& matrix, const QString& fileName)
{
  bool isSuccessful = false;

  if (fileName.length() == 0)
  {
    QMessageBox msgBox;
    msgBox.setText("The file name is empty.");
    msgBox.setInformativeText("Please select a file name.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return isSuccessful;
  }

  isSuccessful = mitk::SaveVtkMatrix4x4ToFile(fileName.toStdString(), matrix);

  if (!isSuccessful)
  {
    QMessageBox msgBox;
    msgBox.setText("The file failed to save.");
    msgBox.setInformativeText("Please check the file location.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return isSuccessful;
  }

  return isSuccessful;
}


//-----------------------------------------------------------------------------
void ApplyMatrixToNodes(const vtkMatrix4x4& matrix, const QmitkDataStorageCheckableComboBox& comboBox)
{
  std::vector<mitk::DataNode*> nodes = comboBox.GetSelectedNodes();
  mitk::DataNode::Pointer node = NULL;
  mitk::BaseData::Pointer data = NULL;

  if (nodes.size() == 0)
  {
    QMessageBox msgBox;
    msgBox.setText("There are no items selected.");
    msgBox.setInformativeText("Please select a valid data item.");
    msgBox.setStandardButtons(QMessageBox::Ok);
    msgBox.setDefaultButton(QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  for (unsigned int i = 0; i < nodes.size(); ++i)
  {
    node = nodes[i];

    if (node.IsNotNull())
    {
      data = dynamic_cast<mitk::BaseData*>(node->GetData());
    }

    if (data.IsNull())
    {
      QMessageBox msgBox;
      msgBox.setText(QString("The data set for item ") + QString::fromStdString(node->GetName()) + QString("%1 is non-existent or does not contain data."));
      msgBox.setInformativeText("Please select a valid data set.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
    }

    bool successful = mitk::ApplyToNode(node, &matrix, true);

    if (!successful)
    {
      QMessageBox msgBox;
      msgBox.setText(QString("Failed to apply transform to item ") + QString::fromStdString(node->GetName()));
      msgBox.setInformativeText("Please check the console.");
      msgBox.setStandardButtons(QMessageBox::Ok);
      msgBox.setDefaultButton(QMessageBox::Ok);
      msgBox.exec();
    }
  } // end foreach node
}


