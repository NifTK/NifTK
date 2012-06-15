/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-30 22:53:06 +0100 (Fri, 30 Sep 2011) $
 Revision          : $Revision: 7847 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkMIDASRegionGrowingProcessor.h"
#include "itkImageFileWriter.h"
 
namespace itk
{
 
template<class TInputImage, class TOutputImage, class TPointSet>
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>
::MIDASRegionGrowingProcessor()
: m_ExtractGreySliceFromReferenceImageFilter(NULL)
, m_CastGreySliceToSegmentationSliceFilter(NULL)
, m_RegionGrowingBySliceFilter(NULL)
, m_PasteRegionFilter(NULL)
, m_RegionOfInterestCalculator(NULL)
, m_GreyScaleImage(NULL)
, m_DestinationImage(NULL)
, m_Seeds(NULL)
, m_LowerThreshold(0)
, m_UpperThreshold(0)
{
  m_ExtractGreySliceFromReferenceImageFilter = ExtractGreySliceFromGreyImageFilterType::New();
  m_CastGreySliceToSegmentationSliceFilter = CastGreySliceToSegmentationSliceFilterType::New();
  m_RegionGrowingBySliceFilter = RegionGrowingBySliceFilterType::New();
  m_PasteRegionFilter = PasteRegionFilterType::New();
  m_RegionOfInterestCalculator = CalculatorType::New();
  m_GreyScaleImage = GreyImageType::New();
  m_DestinationImage = SegmentationImageType::New();
  m_Seeds = PointSetType::New();
}

template<class TInputImage, class TOutputImage, class TPointSet>
void
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "m_Orientation=" << m_Orientation << std::endl;
  os << indent << "m_SliceNumber=" << m_SliceNumber << std::endl;
  os << indent << "m_LowerThreshold=" << m_LowerThreshold << std::endl;
  os << indent << "m_UpperThreshold=" << m_UpperThreshold << std::endl;
  os << indent << "m_RegionOfInterest=\n" << m_RegionOfInterest << std::endl;
  if (m_GreyScaleImage.IsNotNull())
  {
    os << indent << "m_GreyScaleImage=" << m_GreyScaleImage << std::endl;
  }
  else
  {
    os << indent << "m_GreyScaleImage=NULL" << std::endl;
  }
  if (m_DestinationImage.IsNotNull())
  {
    os << indent << "m_DestinationImage=" << m_DestinationImage << std::endl;
  }
  else
  {
    os << indent << "m_DestinationImage=NULL" << std::endl;
  }
  if (m_Seeds.IsNotNull())
  {
    os << indent << "m_Seeds=" << m_Seeds << std::endl;
  }
  else
  {
    os << indent << "m_Seeds=NULL" << std::endl;
  }
}

template<class TInputImage, class TOutputImage, class TPointSet> 
bool
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet> 
::RegionOK(RegionType region)
{
  bool isRegionOK = true;
  SizeType requestedSize = region.GetSize();
  unsigned long int volumeSize = requestedSize[0];
  
  for (unsigned int i = 1; i < TInputImage::ImageDimension; i++)
  {
    volumeSize *= requestedSize[i];
  } 
  
  if (volumeSize < 1)
  {
    isRegionOK = false;
  }
  
  return isRegionOK;
}

template<class TInputImage, class TOutputImage, class TPointSet>
void 
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>
::PropagatePointList(
  int axis, 
  typename MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>::RegionType currentRegion,
  typename MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>::PointSetType::Pointer pointsIn, 
  typename MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>::PointSetType::Pointer pointsOut)
{
  IndexType voxelIndex;
  IndexType currentRegionIndex = currentRegion.GetIndex();
  PointType propagatedPoint;
  
  unsigned long int counter = 0;
  for (unsigned long int j = 0; j < pointsIn->GetPoints()->Size(); j++)
  {
    m_DestinationImage->TransformPhysicalPointToIndex(pointsIn->GetPoints()->GetElement(j), voxelIndex);

    // Only accept points that are for the current slice (just to be safe).
    if (voxelIndex[axis] == m_SliceNumber)
    {
      // But if we are propogating through multiple slices, we then move them to the right slice.
      voxelIndex[axis] =  currentRegionIndex[axis];
          
      // Again, double check we are not outside the ROI.
      if (currentRegion.IsInside(voxelIndex))
      {
        m_DestinationImage->TransformIndexToPhysicalPoint(voxelIndex, propagatedPoint);
        pointsOut->GetPoints()->InsertElement(counter, propagatedPoint);
        counter++;
      }
    } // end if
  } // end for
}

template<class TInputImage, class TOutputImage, class TPointSet>
void 
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>
::SetContours(ParametricPathVectorType& contours)
{
  m_Contours = contours;
  this->Modified();
}

template<class TInputImage, class TOutputImage, class TPointSet>
void
MIDASRegionGrowingProcessor<TInputImage, TOutputImage, TPointSet>
::Execute()
{
  SegmentationImagePixelType foreground = 1;
  SegmentationImagePixelType background = 0;
  
  if (m_GreyScaleImage.IsNull())
  {
    itkExceptionMacro(<< "Grey scale image has not been provided.");
  }
  
  if (m_DestinationImage.IsNull())
  {
    itkExceptionMacro(<< "Destination image has not been provided.");
  }

  if (m_Seeds.IsNull())
  {
    itkExceptionMacro(<< "Seeds have not been provided.");
  }
  
  RegionType largestPossibleDestinationRegion = m_DestinationImage->GetLargestPossibleRegion(); 
  if (!largestPossibleDestinationRegion.IsInside(m_RegionOfInterest))
  {
    itkExceptionMacro(<< "Region of interest=\n" << m_RegionOfInterest << " is not within the destination image=\n" <<  largestPossibleDestinationRegion); 
  }
  
  RegionType largestPossibleGreyScaleRegion = m_GreyScaleImage->GetLargestPossibleRegion(); 
  if (!largestPossibleGreyScaleRegion.IsInside(m_RegionOfInterest))
  {
    itkExceptionMacro(<< "Region of interest=\n" << m_RegionOfInterest << " is not within the grey scale image=\n" <<  largestPossibleGreyScaleRegion); 
  }
  
  if (!RegionOK(m_RegionOfInterest))
  {
    itkExceptionMacro(<< "Region of interest=\n" << m_RegionOfInterest << " is invalid");
  }

  int axis = m_RegionOfInterestCalculator->GetAxis(m_DestinationImage, m_Orientation);
  std::vector<RegionType> listOfSlicesToProcess = m_RegionOfInterestCalculator->SplitRegionBySlices(m_RegionOfInterest, m_DestinationImage, m_Orientation);
  
  typename SegmentationImageType::Pointer outputImage = m_DestinationImage;
  IndexType                               voxelIndex;
  ContinuousIndexType                     continuousIndex;
  ParametricPathVertexType                vertex;
  
  if (listOfSlicesToProcess.size() > 0)
  {
  
    for (unsigned int i = 0; i < listOfSlicesToProcess.size(); i++)
    {
      RegionType currentRegion = listOfSlicesToProcess[i];
      IndexType  currentRegionIndex = currentRegion.GetIndex();
      SizeType   currentRegionSize = currentRegion.GetSize();
      
      m_ExtractGreySliceFromReferenceImageFilter->SetInput(m_GreyScaleImage);
      m_ExtractGreySliceFromReferenceImageFilter->SetExtractionRegion(currentRegion);
      m_CastGreySliceToSegmentationSliceFilter->SetInput(m_ExtractGreySliceFromReferenceImageFilter->GetOutput());
      m_CastGreySliceToSegmentationSliceFilter->UpdateLargestPossibleRegion();
    
      typename SegmentationImageType::Pointer contoursImage = m_CastGreySliceToSegmentationSliceFilter->GetOutput();
      contoursImage->FillBuffer(background);
      
      if (m_Contours.size() > 0)
      {
        // Basically, we need to draw all contours into image.
        // Each contour is a set of points that run "between" voxels.
        // (i.e. at exactly the half way point between voxels).
        // So we need to paint either side of the contour line.

        for (unsigned int j = 0; j < m_Contours.size(); j++)
        {
          ParametricPathPointer path = m_Contours[j];
          const ParametricPathVertexListType* list = path->GetVertexList();

          for (unsigned int k = 0; k < list->Size(); k++)
          {
            vertex = list->ElementAt(k);
            
            m_DestinationImage->TransformPhysicalPointToContinuousIndex(vertex, continuousIndex);
            
            for (unsigned int a = 0; a < currentRegionSize.GetSizeDimension(); a++)
            {
              voxelIndex[a] = continuousIndex[a];
            }
            
            IndexType  paintingRegionIndex = voxelIndex;
            SizeType   paintingRegionSize;
            paintingRegionSize.Fill(2);
            paintingRegionSize[axis] = 1;
            
            RegionType paintingRegion;
            paintingRegion.SetSize(paintingRegionSize);
            paintingRegion.SetIndex(paintingRegionIndex);

            ImageRegionIterator<SegmentationImageType> iterator(contoursImage, paintingRegion);
            for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
            {
              iterator.Set(foreground);
            }
          } 
        }
      }
      
      typename PointSetType::Pointer propagatedSeeds = PointSetType::New();
      this->PropagatePointList(axis, currentRegion, m_Seeds, propagatedSeeds);
            
      m_RegionGrowingBySliceFilter->SetLowerThreshold(m_LowerThreshold);
      m_RegionGrowingBySliceFilter->SetUpperThreshold(m_UpperThreshold);
      m_RegionGrowingBySliceFilter->SetForegroundValue(foreground);
      m_RegionGrowingBySliceFilter->SetBackgroundValue(background);
      m_RegionGrowingBySliceFilter->SetSeedPoints(*(propagatedSeeds.GetPointer()));
      m_RegionGrowingBySliceFilter->SetContourImage(contoursImage);
      m_RegionGrowingBySliceFilter->SetInput(m_ExtractGreySliceFromReferenceImageFilter->GetOutput());
      m_RegionGrowingBySliceFilter->UpdateLargestPossibleRegion();
     
      m_PasteRegionFilter->InPlaceOn();      
      m_PasteRegionFilter->SetSourceImage(m_RegionGrowingBySliceFilter->GetOutput());
      m_PasteRegionFilter->SetSourceRegion(m_RegionGrowingBySliceFilter->GetOutput()->GetLargestPossibleRegion());
      m_PasteRegionFilter->SetDestinationImage(outputImage);
      m_PasteRegionFilter->SetDestinationIndex(currentRegionIndex);
      m_PasteRegionFilter->Update();
      
      outputImage = m_PasteRegionFilter->GetOutput();
      outputImage->DisconnectPipeline();
      
      /*
      typename itk::ImageFileWriter<TOutputImage>::Pointer writer = itk::ImageFileWriter<TOutputImage>::New();
      writer->SetInput(contoursImage);
      writer->SetFileName("matt.contours.nii");
      writer->Update();
      */
    }
  }
  else
  {
    itkWarningMacro(<< "No slices to process, is the region of interest correct?");
  }  
  
  m_DestinationImage = outputImage;
}
 
} // end namespace
