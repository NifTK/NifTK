/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

// Qmitk
#include "ImageStatisticsView.h"
#include "ImageStatisticsViewPreferencesPage.h"

// Qt
#include <QMessageBox>
#include <QTableWidgetItem>

// ITK
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>

// MITK
#include <mitkImageAccessByItk.h>

const std::string ImageStatisticsView::VIEW_ID = "uk.ac.ucl.cmic.imagestatistics";

ImageStatisticsView::ImageStatisticsView()
: m_AutoUpdate(false)
, m_RequireSameSizeImage(true)
, m_AssumeBinary(true)
, m_BackgroundValue(0)
, m_MaskNode(NULL)
, m_ImageNode(NULL)
{
}

ImageStatisticsView::~ImageStatisticsView()
{
}

void ImageStatisticsView::SetFocus()
{
}

void ImageStatisticsView::CreateQtPartControl( QWidget *parent )
{
  // Create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  // Retrieve up-to-date preference values.
  this->RetrievePreferenceValues();

  // Connect slots, so we are ready for action.
  connect( m_Controls.m_UpdateButton, SIGNAL(clicked()), this, SLOT(TryUpdate()) );
}

void ImageStatisticsView::EnableControls(bool enabled)
{
  m_Controls.m_UpdateButton->setEnabled(enabled && !m_AutoUpdate);
  m_Controls.m_Table->setEnabled(enabled);
  m_Controls.m_MaskNameLabel->setEnabled(enabled);
  m_Controls.m_MaskLabel->setEnabled(enabled);
  m_Controls.m_ImageNameLabel->setEnabled(enabled);
  m_Controls.m_ImageLabel->setEnabled(enabled);
}

void ImageStatisticsView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  // Retrieve up-to-date preference values.
  this->RetrievePreferenceValues();
}

void ImageStatisticsView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(VIEW_ID))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );

  m_AutoUpdate = prefs->GetBool(ImageStatisticsViewPreferencesPage::AUTO_UPDATE_NAME, false);
  m_AssumeBinary = prefs->GetBool(ImageStatisticsViewPreferencesPage::ASSUME_BINARY_NAME, true);
  m_RequireSameSizeImage = prefs->GetBool(ImageStatisticsViewPreferencesPage::REQUIRE_SAME_SIZE_IMAGE_NAME, true);
  m_BackgroundValue = prefs->GetInt(ImageStatisticsViewPreferencesPage::BACKGROUND_VALUE_NAME, 0);
}

void ImageStatisticsView::OnSelectionChanged( berry::IWorkbenchPart::Pointer /*source*/,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{
  bool isValid = this->IsSelectionValid(nodes);

  // If the selection is valid, we enable the controls.
  this->EnableControls(isValid);

  // If the selection is value, we update the image labels to indicate we have selected.
  if (isValid)
  {
    m_ImageNode = NULL;
    if (nodes.count() > 0)
    {
      m_ImageNode = nodes[0];
      m_Controls.m_ImageNameLabel->setText(QString(m_ImageNode->GetName().c_str()));
    }
    else
    {
      m_Controls.m_ImageNameLabel->setText("please select an image");
    }

    m_MaskNode = NULL;
    if (nodes.size() > 1)
    {
      m_MaskNode = nodes[1];
      m_Controls.m_MaskNameLabel->setText(QString(m_MaskNode->GetName().c_str()));
    }
    else
    {
      m_Controls.m_MaskNameLabel->setText("please select an image");
    }
  }
  else
  {
    m_MaskNode = NULL;
    m_ImageNode = NULL;
  }

  // Optionally (depending on m_AutoUpdate preference), trigger the update.
  // The alternative, is for the user to hit the Update button.
  if (isValid && m_AutoUpdate)
  {
    this->Update(nodes);
  }
}

bool ImageStatisticsView::IsSelectionValid(const QList<mitk::DataNode::Pointer>& nodes)
{
  bool isValid = true;

  // We must have either 1, or 2 nodes selected.
  if (nodes.count() == 0 || nodes.count() > 2)
  {
    isValid = false;
  }

  // All nodes must be non null images.
  foreach( mitk::DataNode::Pointer node, nodes )
  {
    if(node.IsNull())
    {
      isValid = false;
    }

    if (node.IsNotNull() && dynamic_cast<mitk::Image*>(node->GetData()) == NULL)
    {
      isValid = false;
    }
  }

  // Give up now, if the input is invalid.
  if (!isValid)
  {
    return isValid;
  }

  // If we have 2 images and m_RequireSameSizeImage is true, they must be the same size.
  if (nodes.count() == 2 && m_RequireSameSizeImage)
  {
    mitk::Image::Pointer images[2] = {NULL, NULL};
    for(int i = 0; i < nodes.count(); i++)
    {
      images[i] = dynamic_cast<mitk::Image*>(nodes[i]->GetData());
    }

    if (images[0]->GetDimension() != images[1]->GetDimension())
    {
      isValid = false;
    }
    else
    {
      for (unsigned int i = 0; i < images[0]->GetDimension(); i++)
      {
        if (images[0]->GetDimension(i) != images[1]->GetDimension(i))
        {
          isValid = false;
          break;
        }
      }
    }
  } // end if nodes.count()
  return isValid;
}

void ImageStatisticsView::TryUpdate()
{

  // This does not work, as when you select the Update button, you lose the selection in the DataManager as it loses focus.
  // const QList<mitk::DataNode::Pointer>& nodes = this->GetDataManagerSelection();

  // So create a list.
  QList<mitk::DataNode::Pointer> nodes;

  if (m_ImageNode.IsNotNull())
  {
    nodes.push_back(m_ImageNode);
  }
  if (m_MaskNode.IsNotNull())
  {
    nodes.push_back(m_MaskNode);
  }

  // Check nodes.
  bool isValid = this->IsSelectionValid(nodes);
  if (isValid)
  {
    this->Update(nodes);
  }
}


void ImageStatisticsView::Update(const QList<mitk::DataNode::Pointer>& nodes)
{
  // We are assuming nodes is valid input, and not checking it any further.
  try
  {
    mitk::DataNode::Pointer imageNode = NULL;
    if (nodes.count() > 0)
    {
      imageNode = nodes[0];
    }
    mitk::DataNode::Pointer maskNode = NULL;
    if (nodes.size() > 1)
    {
      maskNode = nodes[1];
    }

    mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
    mitk::Image::Pointer mask = NULL;

    if (maskNode.IsNotNull())
    {
      mask = dynamic_cast<mitk::Image*>(maskNode->GetData());
    }

    if (image.IsNotNull() && mask.IsNull())
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessFixedDimensionByItk(image, UpdateTable, 2);
        break;
      case 3:
        AccessFixedDimensionByItk(image, UpdateTable, 3);
        break;
      case 4:
        AccessFixedDimensionByItk(image, UpdateTable, 4);
        break;
      default:
        MITK_ERROR << "During ImageStatisticsView::UpdateTable, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
    else if (image.IsNotNull() && mask.IsNotNull())
    {
      int dimensions = image->GetDimension();
      switch(dimensions)
      {
      case 2:
        AccessTwoImagesFixedDimensionByItk(image, mask, UpdateTableWithMask, 2);
        break;
      case 3:
        AccessTwoImagesFixedDimensionByItk(image, mask, UpdateTableWithMask, 3);
        break;
      case 4:
        AccessTwoImagesFixedDimensionByItk(image, mask, UpdateTableWithMask, 4);
        break;
      default:
        MITK_ERROR << "During ImageStatisticsView::UpdateTableWithMask, unsupported number of dimensions:" << dimensions << std::endl;
      }
    }
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "During ImageStatisticsView::Update, caught mitk::AccessByItkException caused by:" << e.what() << std::endl;
  }
  catch( itk::ExceptionObject &err )
  {
    MITK_ERROR << "During ImageStatisticsView::Update, caught itk::ExceptionObject caused by:" << err.what() << std::endl;
  }
}

void ImageStatisticsView::InitializeTable()
{
  m_Controls.m_Table->clear();
  m_Controls.m_Table->setColumnCount(9);

  // The order of these columns must match the order in AddTableRow.
  QStringList headers;
  headers << "value";
  headers << "volume (ml)";
  headers << "mean";
  headers << "mean 60%";
  headers << "mean 70%";
  headers << "std dev";
  headers << "min";
  headers << "max";
  headers << "count";
  m_Controls.m_Table->setHorizontalHeaderLabels(headers);
}

template <typename PixelType>
void
ImageStatisticsView
::AddTableRow(int row,
    QString &value, PixelType &min, PixelType &max, double &mean,
    double &stdDev, unsigned long int &count, double &volume)
{
  QTableWidgetItem *valueItem = new QTableWidgetItem(tr("%1").arg(value));
  m_Controls.m_Table->setItem(row, 0, valueItem);

  QTableWidgetItem *volumeItem = new QTableWidgetItem(tr("%1").arg(volume/1000.0)); // convert cubic millimetres to cubic centimetres (ml).
  m_Controls.m_Table->setItem(row, 1, volumeItem);

  QTableWidgetItem *meanItem = new QTableWidgetItem(tr("%1").arg(mean));
  m_Controls.m_Table->setItem(row, 2, meanItem);

  QTableWidgetItem *mean60Item = new QTableWidgetItem(tr("%1").arg(mean * 0.6));
  m_Controls.m_Table->setItem(row, 3, mean60Item);

  QTableWidgetItem *mean70Item = new QTableWidgetItem(tr("%1").arg(mean * 0.7));
  m_Controls.m_Table->setItem(row, 4, mean70Item);

  QTableWidgetItem *stdDevItem = new QTableWidgetItem(tr("%1").arg(stdDev));
  m_Controls.m_Table->setItem(row, 5, stdDevItem);

  QTableWidgetItem *minItem = new QTableWidgetItem(tr("%1").arg(min));
  m_Controls.m_Table->setItem(row, 6, minItem);

  QTableWidgetItem *maxItem = new QTableWidgetItem(tr("%1").arg(max));
  m_Controls.m_Table->setItem(row, 7, maxItem);

  QTableWidgetItem *countItem = new QTableWidgetItem(tr("%1").arg(count));
  m_Controls.m_Table->setItem(row, 8, countItem);

}

template <typename PixelType, unsigned int VImageDimension>
void
ImageStatisticsView
::GetLabelValues(
    itk::Image<PixelType, VImageDimension>* itkImage,
    std::set<PixelType> &labels)
{
  labels.clear();

  itk::ImageRegionConstIterator< itk::Image<PixelType, VImageDimension> >
    iterator(itkImage, itkImage->GetLargestPossibleRegion());

  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    labels.insert(iterator.Get());
  }
}

template <typename PixelType, unsigned int VImageDimension>
void
ImageStatisticsView
::GetVoxelVolume(
    itk::Image<PixelType, VImageDimension>* itkImage,
    double &volume
    )
{

  typedef itk::Image<PixelType, VImageDimension> ImageType;
  typedef typename ImageType::SpacingType SpacingType;

  SpacingType spacing = itkImage->GetSpacing();

  volume = 1;
  for (unsigned int i = 0; i < SpacingType::GetVectorDimension(); i++)
  {
    volume *= spacing[i];
  }
}

template <typename TPixel>
void
ImageStatisticsView
::TestMinAndMax(
    TPixel &imageValue,
    TPixel &min,
    TPixel &max
    )
{
  if (imageValue < min)
  {
    min = imageValue;
  }
  if (imageValue > max)
  {
    max = imageValue;
  }
}

template <typename TPixel>
void
ImageStatisticsView
::InitializeData(
    TPixel &min,
    TPixel &max,
    double &mean,
    double &s0,
    double &s1,
    double &s2,
    double &stdDev,
    unsigned long int &counter
    )
{
  min = std::numeric_limits<TPixel>::max();
  max = std::numeric_limits<TPixel>::min();
  mean = 0;
  s0 = 0;
  s1 = 0;
  s2 = 0;
  stdDev = 0;
  counter = 0;
}

template <typename TPixel>
void
ImageStatisticsView
::AccumulateData(
    TPixel &imageValue,
    double &mean,
    double &s0,
    double &s1,
    double &s2,
    unsigned long int &counter
    )
{
  mean += imageValue;
  s0 += 1;
  s1 += imageValue;
  s2 += imageValue*imageValue;
  counter++;
}

template <typename TPixel>
void
ImageStatisticsView
::AccumulateValue(
    TPixel &imageValue,
    TPixel &min,
    TPixel &max,
    double &mean,
    double &s0,
    double &s1,
    double &s2,
    unsigned long int &counter
    )
{
  if (imageValue != (TPixel)m_BackgroundValue)
  {
    this->TestMinAndMax<TPixel>(imageValue, min, max);
    this->AccumulateData<TPixel>(imageValue, mean, s0, s1, s2, counter);
  }
}

template <typename TPixel1, typename TPixel2, typename LabelType>
void
ImageStatisticsView
::AccumulateValue(
    bool &invert,
    LabelType &valueToCompareMaskAgainst,
    TPixel1 &imageValue,
    TPixel2 &maskValue,
    TPixel1 &min,
    TPixel1 &max,
    double  &mean,
    double  &s0,
    double  &s1,
    double  &s2,
    unsigned long int &counter
    )
{
  if (   (!invert && maskValue == (TPixel2)valueToCompareMaskAgainst)
      || (invert &&  maskValue != (TPixel2)valueToCompareMaskAgainst)
      )
  {
    this->TestMinAndMax<TPixel1>(imageValue, min, max);
    this->AccumulateData<TPixel1>(imageValue, mean, s0, s1, s2, counter);
  }
}

void ImageStatisticsView::CalculateMeanAndStdDev(
    double &mean,
    double &s0,
    double &s1,
    double &s2,
    double &stdDev,
    unsigned long int &counter
    )
{
  if (counter > 0)
  {
    mean /= (double)counter;
    stdDev = sqrt( (double)((s0*s2 - s1*s1) / (s0*(s0 - 1))) );
  }
  else
  {
    mean = 0;
    stdDev = 0;
  }
}

template <typename TPixel, unsigned int VImageDimension>
void
ImageStatisticsView
::UpdateTable(
    itk::Image<TPixel, VImageDimension>* itkImage
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> GreyImageType;
  double mean, s0, s1, s2, stdDev;
  unsigned long int counter;
  TPixel greyPixel, min, max;

  // Initialize table.
  this->InitializeTable();
  m_Controls.m_Table->setRowCount(1);

  // Calculate Stats.
  this->InitializeData(min, max, mean, s0, s1, s2, stdDev, counter);

  // Iterate through image, calculating stats for anything != background value.
  itk::ImageRegionConstIterator<GreyImageType> iter(itkImage, itkImage->GetLargestPossibleRegion());
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    greyPixel = iter.Get();
    this->AccumulateValue<TPixel>(greyPixel, min, max, mean, s0, s1, s2, counter);
  }
  this->CalculateMeanAndStdDev(mean, s0, s1, s2, stdDev, counter);

  // Get voxel volume.
  double volume;
  this->GetVoxelVolume<TPixel, VImageDimension>(itkImage, volume);
  volume *= (double)counter;

  QString value = tr("All except %1").arg(m_BackgroundValue);
  this->AddTableRow(0, value, min, max, mean, stdDev, counter, volume);
}

template <typename TPixel1, unsigned int VImageDimension1, typename TPixel2, unsigned int VImageDimension2>
void
ImageStatisticsView
::UpdateTableWithMask(
    itk::Image<TPixel1, VImageDimension1>* itkImage,
    itk::Image<TPixel2, VImageDimension2>* itkMask
    )
{
  typedef typename itk::Image<TPixel1, VImageDimension1> GreyImageType;
  typedef typename itk::Image<TPixel2, VImageDimension2> MaskImageType;
  double mean, s0, s1, s2, stdDev;
  unsigned long int counter;
  bool invert;
  TPixel1 greyPixel, min, max;
  TPixel2 maskPixel;

  // Get a list of values in itkMask.
  std::set<TPixel2> labels;
  if (!m_AssumeBinary)
  {
    this->GetLabelValues<TPixel2, VImageDimension2>(itkMask, labels);
  }

  // Initialize table.
  this->InitializeTable();
  if (labels.size() > 1)
  {
    m_Controls.m_Table->setRowCount(labels.size());
  }
  else
  {
    m_Controls.m_Table->setRowCount(1);
  }

  if (m_AssumeBinary)
  {
    invert = true;

    // Initialize variables
    this->InitializeData(min, max, mean, s0, s1, s2, stdDev, counter);

    // We iterate over the image, calculating stats for any voxel where the mask value is NOT the background value.
    // i.e. we are using the mask image, but if the mask has multiple labels, we treat any label except the background label as foreground, and accumulate stats.

    itk::ImageRegionConstIterator<GreyImageType> greyIter(itkImage, itkImage->GetLargestPossibleRegion());
    itk::ImageRegionConstIterator<MaskImageType> binaryIter(itkMask, itkMask->GetLargestPossibleRegion());
    for (greyIter.GoToBegin(), binaryIter.GoToBegin(); !greyIter.IsAtEnd() && !binaryIter.IsAtEnd(); ++greyIter, ++binaryIter)
    {
      greyPixel = greyIter.Get();
      maskPixel = binaryIter.Get();

      this->AccumulateValue<TPixel1, TPixel2, int>
        (invert, m_BackgroundValue, greyPixel, maskPixel, min, max, mean, s0, s1, s2, counter);
    }
    this->CalculateMeanAndStdDev(mean, s0, s1, s2, stdDev, counter);

    // Get voxel volume.
    double volume;
    this->GetVoxelVolume<TPixel1, VImageDimension2>(itkImage, volume);
    volume *= (double)counter;

    QString value = tr("All except %1").arg(m_BackgroundValue);
    this->AddTableRow(0, value, min, max, mean, stdDev, counter, volume);
  }
  else
  {
    invert = false;

    // We compute stats for EACH label.
    // This is a bit slow, as we repeatedly iterate over the image.

    typename std::set<TPixel2>::iterator iterator;
    unsigned int rowCounter = 0;
    for (iterator = labels.begin(); iterator != labels.end(); iterator++)
    {
      TPixel2 labelValue = *iterator;

      // Initialize variables
      this->InitializeData(min, max, mean, s0, s1, s2, stdDev, counter);

      itk::ImageRegionConstIterator<GreyImageType> greyIter(itkImage, itkImage->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<MaskImageType> binaryIter(itkMask, itkMask->GetLargestPossibleRegion());
      for (greyIter.GoToBegin(), binaryIter.GoToBegin(); !greyIter.IsAtEnd() && !binaryIter.IsAtEnd(); ++greyIter, ++binaryIter)
      {
        greyPixel = greyIter.Get();
        maskPixel = binaryIter.Get();

        this->AccumulateValue<TPixel1, TPixel2, TPixel2>
          (invert, labelValue, greyPixel, maskPixel, min, max, mean, s0, s1, s2, counter);
      }
      this->CalculateMeanAndStdDev(mean, s0, s1, s2, stdDev, counter);

      // Get voxel volume.
      double volume;
      this->GetVoxelVolume<TPixel1, VImageDimension1>(itkImage, volume);
      volume *= (double)counter;

      QString value = tr("%1").arg(labelValue);
      this->AddTableRow(rowCounter++, value, min, max, mean, stdDev, counter, volume);
    }
  }
}
