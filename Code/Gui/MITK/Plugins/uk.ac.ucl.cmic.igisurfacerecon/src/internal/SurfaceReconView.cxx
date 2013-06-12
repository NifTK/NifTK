/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "SurfaceReconView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "SurfaceReconViewActivator.h"
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkNodePredicateDataType.h>
#include <QFileDialog>
#include <mitkCoordinateAxesData.h>


const std::string SurfaceReconView::VIEW_ID = "uk.ac.ucl.cmic.igisurfacerecon";

//-----------------------------------------------------------------------------
SurfaceReconView::SurfaceReconView()
{
  m_SurfaceReconstruction = niftk::SurfaceReconstruction::New();
}


//-----------------------------------------------------------------------------
SurfaceReconView::~SurfaceReconView()
{
}


//-----------------------------------------------------------------------------
std::string SurfaceReconView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void SurfaceReconView::LeftBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Intrinsic Camera Calibration");
  if (!file.isEmpty())
  {
    LeftIntrinsicPathLineEdit->setText(file);
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RightBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Intrinsic Camera Calibration");
  if (!file.isEmpty())
  {
    RightIntrinsicPathLineEdit->setText(file);
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::StereoRigBrowseButtonClicked()
{
  // FIXME: this blocks timer delivery?
  QString   file = QFileDialog::getOpenFileName(GetParent(), "Stereo Rig Calibration");
  if (!file.isEmpty())
  {
    StereoRigTransformationPathLineEdit->setText(file);
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CreateQtPartControl( QWidget *parent )
{
  setupUi(parent);
  connect(DoItButton, SIGNAL(clicked()), this, SLOT(DoSurfaceReconstruction()));

  connect(LeftIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(LeftBrowseButtonClicked()));
  connect(RightIntrinsicBrowseButton, SIGNAL(clicked()), this, SLOT(RightBrowseButtonClicked()));
  connect(StereoRigTransformBrowseButton, SIGNAL(clicked()), this, SLOT(StereoRigBrowseButtonClicked()));

  ctkServiceReference ref = mitk::SurfaceReconViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::SurfaceReconViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
  this->RetrievePreferenceValues();
  this->LeftChannelNodeNameComboBox->SetDataStorage(this->GetDataStorage());
  this->RightChannelNodeNameComboBox->SetDataStorage(this->GetDataStorage());
  CameraNodeComboBox->SetDataStorage(GetDataStorage());

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  this->LeftChannelNodeNameComboBox->SetAutoSelectNewItems(false);
  this->LeftChannelNodeNameComboBox->SetPredicate(isImage);
  this->RightChannelNodeNameComboBox->SetAutoSelectNewItems(false);
  this->RightChannelNodeNameComboBox->SetPredicate(isImage);
  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isCoords = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
  CameraNodeComboBox->SetAutoSelectNewItems(false);
  CameraNodeComboBox->SetPredicate(isCoords);

  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void SurfaceReconView::OnUpdate(const ctkEvent& event)
{
  // Optional. This gets called everytime the data sources are updated.
  // If the surface reconstruction was as fast as the GUI update, we could trigger it here.

  // not sure if enum'ing the storage here is a good idea
  // FIXME: we should register a listener on the data-storage instead?
  UpdateNodeNameComboBox();
}


//-----------------------------------------------------------------------------
void SurfaceReconView::UpdateNodeNameComboBox()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    // leave the editable string part intact!
    // it's extremely annoying having that reset all the time while trying to input something.
    QString leftText  = LeftChannelNodeNameComboBox->currentText();
    QString rightText = RightChannelNodeNameComboBox->currentText();

    bool  wasModified = false;

    std::set<std::string>   nodeNamesLeftToAdd;

    mitk::DataStorage::SetOfObjects::ConstPointer allNodes = storage->GetAll();
    for (mitk::DataStorage::SetOfObjects::ConstIterator i = allNodes->Begin(); i != allNodes->End(); ++i)
    {
      const mitk::DataNode::Pointer node = i->Value();
      assert(node.IsNotNull());

      std::string nodeName = node->GetName();
      if (!nodeName.empty())
      {
        mitk::BaseData::Pointer data = node->GetData();
        if (data.IsNotNull())
        {
          mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(data.GetPointer());
          if (imageInNode.IsNotNull())
          {
            nodeNamesLeftToAdd.insert(nodeName);
          }
        }
      }
    }

    // for all elements that currently are in the combo box
    // check whether there still is a node with that name.
    for (int i = 0; i < LeftChannelNodeNameComboBox->count(); ++i)
    {
      QString itemName = LeftChannelNodeNameComboBox->itemText(i);
      // for now, both left and right have to have the same node names.
      assert(RightChannelNodeNameComboBox->itemText(i) == itemName);

      std::set<std::string>::iterator ni = nodeNamesLeftToAdd.find(itemName.toStdString());
      if (ni == nodeNamesLeftToAdd.end())
      {
        // the node name currently in the combobox is not in data storage
        // so we need to drop it from the combobox
        LeftChannelNodeNameComboBox->removeItem(i);
        RightChannelNodeNameComboBox->removeItem(i);
        wasModified = true;
      }
      else
      {
        // name is still in data-storage
        // so remove it from the to-be-added list
        nodeNamesLeftToAdd.erase(ni);
      }
    }

    for (std::set<std::string>::const_iterator i = nodeNamesLeftToAdd.begin(); i != nodeNamesLeftToAdd.end(); ++i)
    {
      QString s = QString::fromStdString(*i);
      LeftChannelNodeNameComboBox->addItem(s);
      RightChannelNodeNameComboBox->addItem(s);
      wasModified = true;
    }

    // put original text in only if we modified the combobox.
    // otherwise the edit control is reset all the time.
    if (wasModified)
    {
      LeftChannelNodeNameComboBox->setEditText(leftText);
      RightChannelNodeNameComboBox->setEditText(rightText);
    }

    assert(LeftChannelNodeNameComboBox->count() == RightChannelNodeNameComboBox->count());
  }
}


//-----------------------------------------------------------------------------
template <typename PropType>
static void CopyProp(const mitk::DataNode::Pointer source, mitk::Image::Pointer target, const char* name)
{
  mitk::BaseProperty::Pointer baseProp = target->GetProperty(name);
  if (baseProp.IsNull())
  {
    // none there yet, try to pull it from the datanode
    baseProp = source->GetProperty(name);
    if (baseProp.IsNotNull())
    {
      // check that it's the correct type
      typename PropType::Pointer   prop = dynamic_cast<PropType*>(baseProp.GetPointer());
      if (prop.IsNotNull())
      {
        // FIXME: copy? or simply ref the same object?
        target->SetProperty(name, prop);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void SurfaceReconView::CopyImagePropsIfNecessary(const mitk::DataNode::Pointer source, mitk::Image::Pointer target)
{
  // we copy known meta-data properties to the images, but only if they dont exist yet.
  // we'll not ever change the value of an existing property!

  // calibration data
  CopyProp<mitk::CameraIntrinsicsProperty>(source, target, niftk::Undistortion::s_CameraCalibrationPropertyName);

  // has the image been rectified?
  CopyProp<mitk::BoolProperty>(source, target, niftk::SurfaceReconstruction::s_ImageIsRectifiedPropertyName);

  // has the image been undistorted?
  CopyProp<mitk::BoolProperty>(source, target, niftk::Undistortion::s_ImageIsUndistortedPropertyName);
}


//-----------------------------------------------------------------------------
void SurfaceReconView::LoadStereoRig(const std::string& filename, mitk::Image::Pointer img)
{
  assert(img.IsNotNull());

  itk::Matrix<float, 4, 4>    txf;
  txf.SetIdentity();

  if (!filename.empty())
  {
    std::ifstream   file(filename.c_str());
    if (!file.good())
    {
      throw std::runtime_error("Cannot open stereo-rig file " + filename);
    }
    float   values[3 * 4];
    for (unsigned int i = 0; i < (sizeof(values) / sizeof(values[0])); ++i)
    {
      if (!file.good())
      {
        throw std::runtime_error("Cannot read enough data from stereo-rig file " + filename);
      }
      file >> values[i];
    }
    file.close();

    // set rotation
    for (int i = 0; i < 9; ++i)
    {
      txf.GetVnlMatrix()(i / 3, i % 3) = values[i];
    }

    // set translation
    for (int i = 0; i < 3; ++i)
    {
      txf.GetVnlMatrix()(i, 3) = values[9 + i];
    }
  }
  else
  {
    // no idea what to invent here...
  }

  niftk::MatrixProperty::Pointer  prop = niftk::MatrixProperty::New(txf);
  img->SetProperty(niftk::SurfaceReconstruction::s_StereoRigTransformationPropertyName, prop);
}


//-----------------------------------------------------------------------------
void SurfaceReconView::DoSurfaceReconstruction()
{
  mitk::DataStorage::Pointer storage = GetDataStorage();
  if (storage.IsNotNull())
  {
    std::string leftText  = LeftChannelNodeNameComboBox->currentText().toStdString();
    std::string rightText = RightChannelNodeNameComboBox->currentText().toStdString();

    const mitk::DataNode::Pointer leftNode  = storage->GetNamedNode(leftText);
    const mitk::DataNode::Pointer rightNode = storage->GetNamedNode(rightText);

    if (leftNode.IsNotNull() && rightNode.IsNotNull())
    {
      mitk::BaseData::Pointer leftData  = leftNode->GetData();
      mitk::BaseData::Pointer rightData = rightNode->GetData();

      if (leftData.IsNotNull() && rightData.IsNotNull())
      {
        mitk::Image::Pointer leftImage  = dynamic_cast<mitk::Image*>(leftData.GetPointer());
        mitk::Image::Pointer rightImage = dynamic_cast<mitk::Image*>(rightData.GetPointer());

        if (leftImage.IsNotNull() && rightImage.IsNotNull())
        {
          // if our output node exists already then we recycle it, of course.
          // it may not be tagged as "derived" from the correct source nodes
          // but that shouldn't be a problem here.

          std::string               outputName = OutputNodeNameLineEdit->text().toStdString();
          mitk::DataNode::Pointer   outputNode = storage->GetNamedNode(outputName);
          if (outputNode.IsNull())
          {
            outputNode = mitk::DataNode::New();
            outputNode->SetName(outputName);

            mitk::DataStorage::SetOfObjects::Pointer   nodeParents = mitk::DataStorage::SetOfObjects::New();
            nodeParents->push_back(leftNode);
            nodeParents->push_back(rightNode);

            storage->Add(outputNode, nodeParents);
          }

          // FIXME: this is here temporarily only. calibration should come from a calibration-plugin instead!
          niftk::Undistortion::LoadCalibration(LeftIntrinsicPathLineEdit->text().toStdString(), leftImage);
          niftk::Undistortion::LoadCalibration(RightIntrinsicPathLineEdit->text().toStdString(), rightImage);
          LoadStereoRig(StereoRigTransformationPathLineEdit->text().toStdString(), rightImage);

          CopyImagePropsIfNecessary(leftNode,  leftImage);
          CopyImagePropsIfNecessary(rightNode, rightImage);

          niftk::SurfaceReconstruction::OutputType  outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
          if (GenerateDisparityImageRadioBox->isChecked())
          {
            assert(!GeneratePointCloudRadioBox->isChecked());
            outputtype = niftk::SurfaceReconstruction::DISPARITY_IMAGE;
          }
          if (GeneratePointCloudRadioBox->isChecked())
          {
            assert(!GenerateDisparityImageRadioBox->isChecked());
            outputtype = niftk::SurfaceReconstruction::POINT_CLOUD;
          }

          // where to place the point cloud in 3d space
          mitk::DataNode::Pointer camNode;
          std::string             camNodeName  = CameraNodeComboBox->currentText().toStdString();
          if (!camNodeName.empty())
          {
            // is ok if node doesnt exist, SurfaceReconstruction will deal with that.
            camNode = storage->GetNamedNode(leftText);
          }

          try
          {
            // Then delagate everything to class outside of plugin, so we can unit test it.
            m_SurfaceReconstruction->Run(storage, outputNode, leftImage, rightImage, 
                niftk::SurfaceReconstruction::SEQUENTIAL_CPU, outputtype, camNode);
          }
          catch (const std::exception& e)
          {
            std::cerr << "Whoops... something went wrong with surface reconstruction: " << e.what() << std::endl;
            // FIXME: show an error message on the plugin panel somewhere?
          }
        }
      }
    }
  }
}
