/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-19 11:45:45 +0100 (Wed, 19 Oct 2011) $
 Revision          : $Revision: 7553 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED

template<typename TPixel, unsigned int VImageDimension>
GeneralSegmentorPipeline<TPixel, VImageDimension>
::GeneralSegmentorPipeline()
{
  m_SliceNumber = 0;
  m_AxisNumber = -1;
  m_Orientation = itk::ORIENTATION_UNKNOWN;
  m_LowerThreshold = 0;
  m_UpperThreshold = 0;
  m_RegionGrowingProcessor = MIDASRegionGrowingProcessorType::New();
  m_AllSeeds = PointSetType::New();
  m_AllContours = PointSetType::New();
}

template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::SetParam(GeneralSegmentorPipelineParams& p)
{
  if (p.m_SliceNumber != m_SliceNumber || p.m_Orientation != m_AxisNumber || p.m_Orientation != m_Orientation)
  {
    RegionType region3D = m_RegionGrowingProcessor->GetGreyScaleImage()->GetLargestPossibleRegion();
    SizeType   sliceSize3D = region3D.GetSize();
    IndexType  sliceIndex3D = region3D.GetIndex();

    sliceSize3D[p.m_AxisNumber] = 1;
    sliceIndex3D[p.m_AxisNumber] = p.m_SliceNumber;

    region3D.SetSize(sliceSize3D);
    region3D.SetIndex(sliceIndex3D);

    m_RegionGrowingProcessor->SetRegionOfInterest(region3D);
    m_RegionGrowingProcessor->SetSliceNumber(p.m_SliceNumber);
    m_RegionGrowingProcessor->SetOrientation(p.m_Orientation);
    m_SliceNumber = p.m_SliceNumber;
    m_AxisNumber = p.m_AxisNumber;
    m_Orientation = p.m_Orientation;
  }

  if (p.m_LowerThreshold != m_LowerThreshold)
  {
    m_RegionGrowingProcessor->SetLowerThreshold(p.m_LowerThreshold);
    m_LowerThreshold = p.m_LowerThreshold;
  }

  if (p.m_UpperThreshold != m_UpperThreshold)
  {
    m_RegionGrowingProcessor->SetUpperThreshold(p.m_UpperThreshold);
    m_UpperThreshold = p.m_UpperThreshold;
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::Update(GeneralSegmentorPipelineParams& params)
{
  try
  {
    // Note that from an implementation perspective, don't forget the following :-)
    // 1. List of seeds may be empty.
    // 2. DrawTool and PolyTool and any other future tool may have zero contours.

    // 1. Clear points
    m_AllSeeds->GetPoints()->Initialize();
    m_AllContours->GetPoints()->Initialize();

    // 2. Convert seeds and contours to ITK PointSets.
    ConvertMITKSeedsAndAppendToITKSeeds(params.m_Seeds, m_AllSeeds);
    ConvertMITKContoursFromAllToolsAndAppendToITKPoints(params, m_AllContours);

    // 3. Hook up the ITK PointSets. Images (Greyscale, Destination) should already be set in the main InvokeITKPipeline method.
    m_RegionGrowingProcessor->SetSeeds(m_AllSeeds);
    m_RegionGrowingProcessor->SetContours(m_AllContours);

    // 4. Go.
    m_RegionGrowingProcessor->Execute();
  }
  catch( itk::ExceptionObject & err )
  {
    MITK_ERROR << "GeneralSegmentorPipeline::Update Failed: " << err << std::endl;
  }
}

#endif

