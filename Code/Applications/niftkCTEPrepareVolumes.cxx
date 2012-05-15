/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-21 14:43:44 +0000 (Mon, 21 Nov 2011) $
 Revision          : $Revision: 7828 $
 Last modified by  : $Author: kkl $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <string>
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkSubtractImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkVotingBinaryIterativeHoleFillingImageFilter.h"
#include "itkCorrectGMUsingPVMapFilter.h"
#include "itkCorrectGMUsingNeighbourhoodFilter.h"
#include "itkCastImageFilter.h"

typedef float InputPixelType;

/*!
 * \file niftkCTEPrepareVolumes.cxx
 * \page niftkCTEPrepareVolumes
 * \section niftkCTEPrepareVolumesSummary Takes up to 5 volumes, each one being a probability map with values [0-1], and produces a single volume with 3 labels (GM, WM, CSF).
 */

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes up to 5 volumes, each one being a probability map with values [0-1]," << std::endl;
    std::cout << "  and produces a single volume with 3 labels (GM, WM, CSF)." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -g <filename> -w <filename> -c <filename> -o <filename> [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -g  <filename>          Input grey matter PV map" << std::endl;
    std::cout << "    -w  <filename>          Input white matter PV map" << std::endl;
    std::cout << "    -c  <filename>          Input CSF matter PV map" << std::endl;
    std::cout << "    -o  <filename>          Output combined label map" << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;   
    std::cout << "    -mask <filename>        Apply a mask to GM, WM and CSF images when classifying" << std::endl;
    std::cout << "    -dGM  <filename>        Add deep grey matter mask to WM" << std::endl;
    std::cout << "    -iCSF <filename>        Add internal CSF mask to WM" << std::endl;
    std::cout << "    -connected              Include connected component analysis" << std::endl;
    std::cout << "    -wl <int>      [1]      Label to use for white matter hard classification on output" << std::endl;
    std::cout << "    -gl <int>      [2]      Label to use for grey matter hard classification on output" << std::endl;    
    std::cout << "    -cl <int>      [3]      Label to use for CSF hard classification on output" << std::endl;
    std::cout << "    -zbg                    If set the label used for background is zero. By default background is treated as CSF." << std::endl;
    std::cout << "    -pg <float>    [0.0]    Minimum probability required for GM" << std::endl;
    std::cout << "    -pw <float>    [0.0]    Minimum probability required for WM" << std::endl;
    std::cout << "    -pc <float>    [0.0]    Minimum probability required for CSF" << std::endl;
    std::cout << "    -pm                     Compute mask from minimum probabilities (if GM < pg & WM < pw & CSF < pc -> mask = 0)" << std::endl;
    std::cout << "    -bm <filename>          Output brain mask (GM+WM) [0-1]" << std::endl;
    std::cout << "    -wm <filename>          Output white matter mask [0-1]" << std::endl;
    std::cout << "    -gm <filename>          Output grey mask [0-1]" << std::endl;
    std::cout << "    -cm <filename>          Output CSF mask [0-1]" << std::endl;
    std::cout << "    -clampGM <filename>     Output GM PV map, clamping values between [0-1]" << std::endl;
    std::cout << "    -AcostaCorrection       Correct the GM as per Acosta et. al. MIA 13 (2009) 730-743 doi:10.1016/j.media.2009.07.03, section 2.3" << std::endl;
    std::cout << "    -BourgeatCorrection     Correct the GM as per Bourgeat et. al. ISBI 2008, section 2.3.1." << std::endl;
  }

struct arguments
{
  // Define command line params
  std::string greyPVImage;
  std::string whitePVImage;
  std::string csfPVImage;
  std::string maskImageName;
  std::string deepGreyPVImage;
  std::string internalCSFPVImage;
  std::string outputLabelImage;
  std::string brainMaskImageName;
  std::string whiteMaskImageName;
  std::string greyMaskImageName;
  std::string csfMaskImageName;
  std::string clampGMPVImageName;
  
  int whiteLabel;
  int greyLabel;
  int csfLabel;
  int bgLabel;
  double minPG;
  double minPW;
  double minPC;
  bool connected;
  bool bourgeatCorrection;
  bool acostaCorrection;
  bool doMaskFromThresholds;
};

template <int Dimension, class OutputPixelType> 
unsigned long int 
GetLargestConnectedComponent(void* voidImage)
{
  typedef typename itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef typename itk::MinimumMaximumImageCalculator<OutputImageType> MinMaxCalculatorType;
  
  // Eeurgh.
  OutputImageType* image = static_cast<OutputImageType*>(voidImage);
  
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();
  minMaxCalculator->SetImage(image);
  minMaxCalculator->Compute();
  
  OutputPixelType min = minMaxCalculator->GetMinimum();
  OutputPixelType max = minMaxCalculator->GetMaximum();
  

  std::cout << "Min=" << niftk::ConvertToString(min) << ", max=" + niftk::ConvertToString(max) << std::endl;

  unsigned long int labels = max-min+1;
  
  std::cout << "Labels=" << niftk::ConvertToString((int)labels) << std::endl;
  
  unsigned long int* counts = new unsigned long int[labels];
  for (unsigned long int i = 0; i < labels; i++) counts[i] = 0;
  
  itk::ImageRegionIterator<OutputImageType> connectedIterator(image, image->GetLargestPossibleRegion());
  for (connectedIterator.GoToBegin(); !connectedIterator.IsAtEnd(); ++connectedIterator)
    {
      counts[connectedIterator.Get() - min]++;
    }
  
  unsigned long int largestIndex = 0;
  unsigned long int largestCounted = 0;
  for (unsigned long int i = 1; i < labels; i++) // start at 1 so we ignore background
    {
      if (counts[i] > largestCounted)
        {
          largestIndex = i;
          largestCounted = counts[i];
        }
    }
  
  std::cout << "Largest index=" << niftk::ConvertToString((int)largestIndex) << ", so most frequent value=" << niftk::ConvertToString((int)(largestIndex + min)) << ", which had " << niftk::ConvertToString((int)(largestCounted)) << std::endl;

  delete [] counts;
  
  return largestIndex + min;
}

template <int Dimension, class OutputPixelType> 
int DoMain(arguments args)
{
  typedef itk::Image< InputPixelType, Dimension >  InputImageType;  
  typedef itk::ImageFileReader< InputImageType >   InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType >   InputImageWriterType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::ImageFileWriter< OutputImageType >  OutputImageWriterType;
  
  typename InputImageReaderType::Pointer gmReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer wmReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer csfReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer maskReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer deepGreyReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer internalCSFReader = InputImageReaderType::New();
  

  try
  {
    std::cout << "Loading GM PV image:" << args.greyPVImage << std::endl;
    gmReader->SetFileName(args.greyPVImage);
    gmReader->Update();
    std::cout << "Done" << std::endl;

    std::cout << "Loading WM PV image:" << args.whitePVImage << std::endl;
    wmReader->SetFileName(args.whitePVImage);
    wmReader->Update();
    std::cout << "Done" << std::endl;

    std::cout << "Loading CSF PV image:" << args.csfPVImage << std::endl;
    csfReader->SetFileName(args.csfPVImage);
    csfReader->Update();
    std::cout << "Done" << std::endl;
    
    if (args.maskImageName.length() > 0)
      {
        std::cout << "Loading mask image:" << args.maskImageName << std::endl;
        maskReader->SetFileName(args.maskImageName);
        maskReader->Update();
        std::cout << "Done" << std::endl;
      }
    
    if (args.deepGreyPVImage.length() > 0)
      {
        std::cout << "Loading deep GM PV image:" << args.deepGreyPVImage << std::endl;
        deepGreyReader->SetFileName(args.deepGreyPVImage);
        deepGreyReader->Update();
        std::cout << "Done" << std::endl;
      }
    
    if (args.internalCSFPVImage.length() > 0)
      {
        std::cout << "Loading internal CSF PV image:" << args.internalCSFPVImage << std::endl;
        internalCSFReader->SetFileName(args.internalCSFPVImage);
        internalCSFReader->Update();
        std::cout << "Done" << std::endl;
      }
    
  }
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }    

  // Create a mask image copy, and then copy mask.
  typename InputImageType::Pointer maskImageCopy = InputImageType::New();
  maskImageCopy->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
  maskImageCopy->SetSpacing(gmReader->GetOutput()->GetSpacing());
  maskImageCopy->SetOrigin(gmReader->GetOutput()->GetOrigin());
  maskImageCopy->SetDirection(gmReader->GetOutput()->GetDirection());
  maskImageCopy->Allocate();
  maskImageCopy->FillBuffer(0);

  itk::ImageRegionConstIterator<InputImageType> maskImageIterator(maskReader->GetOutput(), maskReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<InputImageType> maskImageCopyIterator(maskImageCopy, maskImageCopy->GetLargestPossibleRegion());
  
  if (args.maskImageName.length() > 0)
    {
      for (maskImageIterator.GoToBegin(), maskImageCopyIterator.GoToBegin();
           !maskImageIterator.IsAtEnd();
           ++maskImageIterator, ++maskImageCopyIterator)
        {
          if (maskImageIterator.Get() > 0)
            {
              maskImageCopyIterator.Set(1);
            }
          else
            {
              maskImageCopyIterator.Set(0);
            }
        }
    }
  else
    {
      maskImageCopy->FillBuffer(1);
    }

  // Copy white matter as well.
  typename InputImageType::Pointer whiteImageCopy = InputImageType::New();
  whiteImageCopy->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
  whiteImageCopy->SetSpacing(gmReader->GetOutput()->GetSpacing());
  whiteImageCopy->SetOrigin(gmReader->GetOutput()->GetOrigin());
  whiteImageCopy->SetDirection(gmReader->GetOutput()->GetDirection());
  whiteImageCopy->Allocate();
  whiteImageCopy->FillBuffer(0);
  
  itk::ImageRegionConstIterator<InputImageType> whiteIterator(wmReader->GetOutput(), wmReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<InputImageType> whiteImageCopyIterator(whiteImageCopy, whiteImageCopy->GetLargestPossibleRegion());
  
  for (whiteIterator.GoToBegin(), whiteImageCopyIterator.GoToBegin();
       !whiteIterator.IsAtEnd();
       ++whiteIterator, ++whiteImageCopyIterator)
    {
      whiteImageCopyIterator.Set(whiteIterator.Get());
    }

  if (args.deepGreyPVImage.length() > 0)
    {
      std::cout << "Adding deep GM to WM" << std::endl;
      
      itk::ImageRegionConstIterator<InputImageType> deepGreyIterator(deepGreyReader->GetOutput(), deepGreyReader->GetOutput()->GetLargestPossibleRegion());
      
      for (whiteImageCopyIterator.GoToBegin(), deepGreyIterator.GoToBegin();
           !whiteImageCopyIterator.IsAtEnd();
           ++whiteImageCopyIterator, ++deepGreyIterator)
        {
          InputPixelType added = whiteImageCopyIterator.Get() + deepGreyIterator.Get();
          if (added > 1)
            {
              added = 1;
            }
          
          whiteImageCopyIterator.Set(added);
        }
      
      std::cout << "Done" << std::endl;
    }

  if (args.internalCSFPVImage.length() > 0)
    {
      
      std::cout << "Adding internal CSF to WM" << std::endl;
      
      itk::ImageRegionConstIterator<InputImageType> internalCSFIterator(internalCSFReader->GetOutput(), internalCSFReader->GetOutput()->GetLargestPossibleRegion());
      
      for (whiteImageCopyIterator.GoToBegin(), internalCSFIterator.GoToBegin();
           !whiteImageCopyIterator.IsAtEnd();
           ++whiteImageCopyIterator, ++internalCSFIterator)
        {
          InputPixelType added = whiteImageCopyIterator.Get() + internalCSFIterator.Get();
          if (added > 1)
            {
              added = 1;
            }
          
          whiteImageCopyIterator.Set(added);
        }
      
      std::cout << "Done" << std::endl;
    }

  // Iterate through GM, WM, CSF, picking the highest value for label.
  typename OutputImageType::Pointer hardClassificationImage = OutputImageType::New();
  hardClassificationImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
  hardClassificationImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
  hardClassificationImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
  hardClassificationImage->SetDirection(gmReader->GetOutput()->GetDirection());
  hardClassificationImage->Allocate();
  hardClassificationImage->FillBuffer(0);
  
  std::cout << "Calculating label image" << std::endl;
  
  itk::ImageRegionConstIterator<InputImageType> greyIterator(gmReader->GetOutput(), gmReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionConstIterator<InputImageType> csfIterator(csfReader->GetOutput(), csfReader->GetOutput()->GetLargestPossibleRegion());
  itk::ImageRegionIterator<OutputImageType> hardClassificationIterator(hardClassificationImage, hardClassificationImage->GetLargestPossibleRegion());

  OutputPixelType output = 0;
  InputPixelType g, w, c;
  
  for (maskImageCopyIterator.GoToBegin(),
       whiteImageCopyIterator.GoToBegin(),      
       greyIterator.GoToBegin(),
       csfIterator.GoToBegin(),
       hardClassificationIterator.GoToBegin();
       !greyIterator.IsAtEnd();
       ++maskImageCopyIterator,
       ++whiteImageCopyIterator,       
       ++greyIterator,
       ++csfIterator,
       ++hardClassificationIterator)
    {
    
      if (maskImageCopyIterator.Get() > 0)
        {
          w = whiteImageCopyIterator.Get();
          g = greyIterator.Get();
          c = csfIterator.Get();

          // Apply thresholds to each GM, WM, CSF value.
          if (g < args.minPG) 
            {
              g = 0;
            }
          
          if (w < args.minPW)
            {
              w = 0;
            }
          
          if (c < args.minPC)
            {
              c = 0;
            }

          output = args.csfLabel;

          // The order of these if statements is important.
          // It means in the case where two are equal, grey takes precedence.
          
          if (args.doMaskFromThresholds && w == 0 && g == 0 && c == 0) {
            hardClassificationIterator.Set(args.bgLabel);
          } else {
            if (w > 0 && w >= g && w >= c)
              {
                output = args.whiteLabel;
              }

            if (g > 0 && g >= w && g >= c)
              {
                output = args.greyLabel;
              }
            hardClassificationIterator.Set(output);
          }          
        }
      else
        {
          hardClassificationIterator.Set(args.bgLabel);
        }
    }

  std::cout << "Done" << std::endl;
  
  if (args.connected)
    {
      std::cout << "Doing connected components analysis" << std::endl;
      
      typename OutputImageType::Pointer greyImage = OutputImageType::New();
      greyImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      greyImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      greyImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      greyImage->SetDirection(gmReader->GetOutput()->GetDirection());
      greyImage->Allocate();
      greyImage->FillBuffer(0);

      typename OutputImageType::Pointer whiteImage = OutputImageType::New();
      whiteImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      whiteImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      whiteImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      whiteImage->SetDirection(gmReader->GetOutput()->GetDirection());
      whiteImage->Allocate();
      whiteImage->FillBuffer(0);

      itk::ImageRegionIterator<OutputImageType> greyIterator(greyImage, greyImage->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> whiteIterator(whiteImage, whiteImage->GetLargestPossibleRegion());

      for (hardClassificationIterator.GoToBegin(),
           whiteIterator.GoToBegin(),
           greyIterator.GoToBegin();
           !hardClassificationIterator.IsAtEnd();
           ++hardClassificationIterator,
           ++whiteIterator,
           ++greyIterator)
        {
          if(hardClassificationIterator.Get() == args.greyLabel)
            {
              greyIterator.Set(1);
            }
          else
            {
              greyIterator.Set(0);
            }
          
          if (hardClassificationIterator.Get() == args.whiteLabel)
            {
              whiteIterator.Set(1);
            }
          else
            {
              whiteIterator.Set(0);  
            }
        }
      typedef itk::ConnectedComponentImageFilter<OutputImageType, OutputImageType> ConnectedFilterType;
      typedef itk::BinaryThresholdImageFilter<OutputImageType, OutputImageType> ThresholdFilterType;
      
      typename ConnectedFilterType::Pointer connectedGreyFilter = ConnectedFilterType::New();
      connectedGreyFilter->SetInput(greyImage);
      connectedGreyFilter->UpdateLargestPossibleRegion();

      OutputPixelType largestGrey = GetLargestConnectedComponent<Dimension, OutputPixelType>(connectedGreyFilter->GetOutput());
      std::cout << "Largest grey region is " + niftk::ConvertToString(largestGrey) << std::endl;
      
      typename ThresholdFilterType::Pointer greyConnectedThresholdedFilter = ThresholdFilterType::New();
      greyConnectedThresholdedFilter->SetInput(connectedGreyFilter->GetOutput());
      greyConnectedThresholdedFilter->SetInsideValue(1);
      greyConnectedThresholdedFilter->SetOutsideValue(0);
      greyConnectedThresholdedFilter->SetLowerThreshold(largestGrey);
      greyConnectedThresholdedFilter->SetUpperThreshold(largestGrey);
      greyConnectedThresholdedFilter->UpdateLargestPossibleRegion();
      
      typename ConnectedFilterType::Pointer connectedWhiteFilter = ConnectedFilterType::New();
      connectedWhiteFilter->SetInput(whiteImage);
      connectedWhiteFilter->UpdateLargestPossibleRegion();

      OutputPixelType largestWhite = GetLargestConnectedComponent<Dimension, OutputPixelType>(connectedWhiteFilter->GetOutput());
      std::cout << "Largest white region is " + niftk::ConvertToString(largestWhite) << std::endl;
      
      typename ThresholdFilterType::Pointer whiteConnectedThresholdedFilter = ThresholdFilterType::New();
      whiteConnectedThresholdedFilter->SetInput(connectedWhiteFilter->GetOutput());
      whiteConnectedThresholdedFilter->SetInsideValue(1);
      whiteConnectedThresholdedFilter->SetOutsideValue(0);
      whiteConnectedThresholdedFilter->SetLowerThreshold(largestWhite);
      whiteConnectedThresholdedFilter->SetUpperThreshold(largestWhite);
      whiteConnectedThresholdedFilter->UpdateLargestPossibleRegion();
      
      // Also fill holes in WM...just for good measure.
      typename OutputImageType::SizeType radius;
      radius.Fill(1);
      
      std::cout << "Filling holes" << std::endl;
      
      typedef itk::VotingBinaryIterativeHoleFillingImageFilter<OutputImageType> HoleFillingType;
      typename HoleFillingType::Pointer holeFillingFilter = HoleFillingType::New();
      holeFillingFilter->SetInput(whiteConnectedThresholdedFilter->GetOutput());
      holeFillingFilter->SetBackgroundValue(0);
      holeFillingFilter->SetForegroundValue(1);
      holeFillingFilter->SetMajorityThreshold(2);
      holeFillingFilter->SetMaximumNumberOfIterations(10);
      holeFillingFilter->SetRadius(radius);
      holeFillingFilter->UpdateLargestPossibleRegion();

      // Now iterate through both and set the hard classified image
      itk::ImageRegionIterator<OutputImageType> greyThresholdIterator(greyConnectedThresholdedFilter->GetOutput(), greyConnectedThresholdedFilter->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> whiteThresholdIterator(holeFillingFilter->GetOutput(), whiteConnectedThresholdedFilter->GetOutput()->GetLargestPossibleRegion());

      for(greyThresholdIterator.GoToBegin(),
          whiteThresholdIterator.GoToBegin(),
          hardClassificationIterator.GoToBegin();
          !greyThresholdIterator.IsAtEnd();
          ++greyThresholdIterator,
          ++whiteThresholdIterator,
          ++hardClassificationIterator)
        {
          if (greyThresholdIterator.Get() == 1)
            {
              hardClassificationIterator.Set(args.greyLabel);
            }
          else if (whiteThresholdIterator.Get() == 1)
            {
              hardClassificationIterator.Set(args.whiteLabel);
            }
          else
            {
              hardClassificationIterator.Set(args.csfLabel); 
            }
        }
      std::cout << "Done" << std::endl;
    }
  
  // FIXME:
  typedef typename itk::CastImageFilter<OutputImageType, InputImageType> CastOutputToInputFilterType;
  typedef typename itk::CastImageFilter<InputImageType, OutputImageType> CastInputToOutputFilterType;
  typename CastOutputToInputFilterType::Pointer castOutputToInputFilter = CastOutputToInputFilterType::New();
  typename CastInputToOutputFilterType::Pointer castInputToOutputFilter = CastInputToOutputFilterType::New();
  
  // We now have a "hardClassification" image, which is the image of 3 labels.
  typedef typename itk::CorrectGMUsingPVMapFilter<InputImageType> CorrectUsingBourgeatFilterType;
  typename CorrectUsingBourgeatFilterType::Pointer bourgeatFilter = CorrectUsingBourgeatFilterType::New();
  
  typedef typename itk::CorrectGMUsingNeighbourhoodFilter<InputImageType> CorrectGMUsingAcostaFilterType;
  typename CorrectGMUsingAcostaFilterType::Pointer acostaFilter = CorrectGMUsingAcostaFilterType::New();

  typename OutputImageWriterType::Pointer writer = OutputImageWriterType::New();
  writer->SetFileName(args.outputLabelImage);
  writer->SetInput(hardClassificationImage);
  
  if (args.acostaCorrection)
    {
      castOutputToInputFilter->SetInput(hardClassificationImage);
      castOutputToInputFilter->Update();
      
      acostaFilter->SetLabelThresholds(args.greyLabel, args.whiteLabel, args.csfLabel); 
      acostaFilter->SetSegmentedImage(castOutputToInputFilter->GetOutput());
      acostaFilter->SetUseFullNeighbourHood(true);
      acostaFilter->Update();
      
      castInputToOutputFilter->SetInput(acostaFilter->GetOutput());
      writer->SetInput(castInputToOutputFilter->GetOutput());
    }
  
  if (args.bourgeatCorrection)
    {
      castOutputToInputFilter->SetInput(hardClassificationImage);
      castOutputToInputFilter->Update();

      bourgeatFilter->SetLabelThresholds(args.greyLabel, args.whiteLabel, args.csfLabel); 
      bourgeatFilter->SetSegmentedImage(castOutputToInputFilter->GetOutput());
      bourgeatFilter->SetGMPVMap(gmReader->GetOutput());
      bourgeatFilter->SetGreyMatterThreshold(0.999);
      bourgeatFilter->SetDoCSFCheck(true);
      bourgeatFilter->SetDoGreyMatterCheck(true);
      bourgeatFilter->Update();
      
      castInputToOutputFilter->SetInput(bourgeatFilter->GetOutput());
      writer->SetInput(castInputToOutputFilter->GetOutput());
      
    }
  
  std::cout << "Writing label image " + args.outputLabelImage << std::endl;
  writer->Update();
  std::cout << "Done" << std::endl;
  
  if (args.brainMaskImageName.length() > 0)
    {

      std::cout << "Calculating brain mask image:" + args.brainMaskImageName << std::endl;
      
      typename OutputImageType::Pointer brainMaskImage = OutputImageType::New();
      brainMaskImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      brainMaskImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      brainMaskImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      brainMaskImage->SetDirection(gmReader->GetOutput()->GetDirection());
      brainMaskImage->Allocate();
      brainMaskImage->FillBuffer(0);
      
      itk::ImageRegionIterator<OutputImageType> brainMaskIterator(brainMaskImage, brainMaskImage->GetLargestPossibleRegion());
      
      for(brainMaskIterator.GoToBegin(),
          hardClassificationIterator.GoToBegin();
          !brainMaskIterator.IsAtEnd();
          ++brainMaskIterator,
          ++hardClassificationIterator)
        {
          if (hardClassificationIterator.Get() == args.greyLabel || hardClassificationIterator.Get() == args.whiteLabel)
            {
              brainMaskIterator.Set(1);  
            }
          else
            {
              brainMaskIterator.Set(0);    
            }
        }
      
      writer->SetFileName(args.brainMaskImageName);
      writer->SetInput(brainMaskImage);
      writer->Update();
      std::cout << "Done" << std::endl;
      
    }
  
  if (args.whiteMaskImageName.length() > 0)
    {

      std::cout << "Calculating white matter mask image: " + args.whiteMaskImageName << std::endl;
      
      typename OutputImageType::Pointer whiteMaskImage = OutputImageType::New();
      whiteMaskImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      whiteMaskImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      whiteMaskImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      whiteMaskImage->SetDirection(gmReader->GetOutput()->GetDirection());
      whiteMaskImage->Allocate();
      whiteMaskImage->FillBuffer(0);
      
      itk::ImageRegionIterator<OutputImageType> whiteMaskIterator(whiteMaskImage, whiteMaskImage->GetLargestPossibleRegion());
      
      for(whiteMaskIterator.GoToBegin(),
          hardClassificationIterator.GoToBegin();
          !whiteMaskIterator.IsAtEnd();
          ++whiteMaskIterator,
          ++hardClassificationIterator)
        {
          if (hardClassificationIterator.Get() == args.whiteLabel)
            {
              whiteMaskIterator.Set(1);  
            }
          else
            {
              whiteMaskIterator.Set(0);    
            }
        }
      
      writer->SetFileName(args.whiteMaskImageName);
      writer->SetInput(whiteMaskImage);
      writer->Update();
      std::cout << "Done" << std::endl;
      
    }

  if (args.greyMaskImageName.length() > 0)
    {

      std::cout << "Calculating grey matter mask image: " << args.greyMaskImageName << std::endl;
      
      typename OutputImageType::Pointer greyMaskImage = OutputImageType::New();
      greyMaskImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      greyMaskImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      greyMaskImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      greyMaskImage->SetDirection(gmReader->GetOutput()->GetDirection());
      greyMaskImage->Allocate();
      greyMaskImage->FillBuffer(0);
      
      itk::ImageRegionIterator<OutputImageType> greyMaskIterator(greyMaskImage, greyMaskImage->GetLargestPossibleRegion());
      
      for(greyMaskIterator.GoToBegin(),
          hardClassificationIterator.GoToBegin();
          !greyMaskIterator.IsAtEnd();
          ++greyMaskIterator,
          ++hardClassificationIterator)
        {
          if (hardClassificationIterator.Get() == args.greyLabel)
            {
              greyMaskIterator.Set(1);  
            }
          else
            {
              greyMaskIterator.Set(0);    
            }
        }
      
      writer->SetFileName(args.greyMaskImageName);
      writer->SetInput(greyMaskImage);
      writer->Update();
      std::cout << "Done" << std::endl;
      
    }

  if (args.csfMaskImageName.length() > 0)
    {

      std::cout << "Calculating csf matter mask image: " << args.csfMaskImageName << std::endl;
      
      typename OutputImageType::Pointer csfMaskImage = OutputImageType::New();
      csfMaskImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      csfMaskImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      csfMaskImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      csfMaskImage->SetDirection(gmReader->GetOutput()->GetDirection());
      csfMaskImage->Allocate();
      csfMaskImage->FillBuffer(0);
      
      itk::ImageRegionIterator<OutputImageType> csfMaskIterator(csfMaskImage, csfMaskImage->GetLargestPossibleRegion());
      
      for(csfMaskIterator.GoToBegin(),
          hardClassificationIterator.GoToBegin();
          !csfMaskIterator.IsAtEnd();
          ++csfMaskIterator,
          ++hardClassificationIterator)
        {
          if (hardClassificationIterator.Get() == args.csfLabel)
            {
              csfMaskIterator.Set(1);  
            }
          else
            {
              csfMaskIterator.Set(0);    
            }
        }
      
      writer->SetFileName(args.csfMaskImageName);
      writer->SetInput(csfMaskImage);
      writer->Update();
      std::cout << "Done" << std::endl;
      
    }

  if (args.clampGMPVImageName.length() > 0)
    {
      std::cout << "Clamping GM PV between 0 and 1: " << args.clampGMPVImageName << std::endl;

      typename InputImageType::Pointer clampedGMImage = InputImageType::New();
      clampedGMImage->SetRegions(gmReader->GetOutput()->GetLargestPossibleRegion());
      clampedGMImage->SetSpacing(gmReader->GetOutput()->GetSpacing());
      clampedGMImage->SetOrigin(gmReader->GetOutput()->GetOrigin());
      clampedGMImage->SetDirection(gmReader->GetOutput()->GetDirection());
      clampedGMImage->Allocate();
      clampedGMImage->FillBuffer(0);

      InputPixelType pixel;
      itk::ImageRegionConstIterator<InputImageType> originalGreyIterator(gmReader->GetOutput(), gmReader->GetOutput()->GetLargestPossibleRegion());
      itk::ImageRegionIterator<InputImageType> clampedGreyIterator(clampedGMImage, clampedGMImage->GetLargestPossibleRegion());
      for (originalGreyIterator.GoToBegin(),
           clampedGreyIterator.GoToBegin();
           !originalGreyIterator.IsAtEnd();
           ++originalGreyIterator,
           ++clampedGreyIterator)
        {
          pixel = originalGreyIterator.Get();
          if (pixel < 0)
            {
              pixel = 0;
            }
          else if (pixel > 1)
            {
              pixel = 1;
            }
          clampedGreyIterator.Set(pixel);
        }
      
      typename InputImageWriterType::Pointer floatWriter = InputImageWriterType::New();
      floatWriter->SetFileName(args.clampGMPVImageName);
      floatWriter->SetInput(clampedGMImage);
      floatWriter->Update();
      std::cout << "Done" << std::endl;
      
    }
  return EXIT_SUCCESS;
}

/**
 * \brief Takes GM, WM, CSF segmentations with PV, and creates a labelled (3 values) segmentation.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.whiteLabel = 1;
  args.greyLabel = 2;
  args.csfLabel = 3;
  args.minPG = 0.0;
  args.minPW = 0.0;
  args.minPC = 0.0;
  args.connected = false;
  args.acostaCorrection = false;
  args.bourgeatCorrection = false;
  args.bgLabel = args.csfLabel;
  
  
  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-g") == 0){
      args.greyPVImage=argv[++i];
      std::cout << "Set -g=" << args.greyPVImage << std::endl;
    }
    else if(strcmp(argv[i], "-w") == 0){
      args.whitePVImage=argv[++i];
      std::cout << "Set -w=" << args.whitePVImage << std::endl;
    }
    else if(strcmp(argv[i], "-c") == 0){
      args.csfPVImage=argv[++i];
      std::cout << "Set -c=" << args.csfPVImage << std::endl;
    }        
    else if(strcmp(argv[i], "-o") == 0){
      args.outputLabelImage=argv[++i];
      std::cout << "Set -o=" << args.outputLabelImage << std::endl;
    }
    else if(strcmp(argv[i], "-bm") == 0){
      args.brainMaskImageName=argv[++i];
      std::cout << "Set -bm=" << args.brainMaskImageName << std::endl;
    }
    else if(strcmp(argv[i], "-mask") == 0){
      args.maskImageName=argv[++i];
      std::cout << "Set -mask=" << args.maskImageName << std::endl;
    }
    else if(strcmp(argv[i], "-dGM") == 0){
      args.deepGreyPVImage=argv[++i];
      std::cout << "Set -dGM=" << args.deepGreyPVImage << std::endl;
    }    
    else if(strcmp(argv[i], "-iCSF") == 0){
      args.internalCSFPVImage=argv[++i];
      std::cout << "Set -iCSF=" << args.internalCSFPVImage << std::endl;
    }        
    else if(strcmp(argv[i], "-wm") == 0){
      args.whiteMaskImageName=argv[++i];
      std::cout << "Set -wm=" << args.whiteMaskImageName << std::endl;
    }
    else if(strcmp(argv[i], "-gm") == 0){
      args.greyMaskImageName=argv[++i];
      std::cout << "Set -gm=" << args.greyMaskImageName << std::endl;
    }
    else if(strcmp(argv[i], "-cm") == 0){
      args.csfMaskImageName=argv[++i];
      std::cout << "Set -cm=" << args.csfMaskImageName << std::endl;
    }                
    else if(strcmp(argv[i], "-gl") == 0){
      args.greyLabel=atoi(argv[++i]);
      std::cout << "Set -gl=" << niftk::ConvertToString(args.greyLabel) << std::endl;
    }
    else if(strcmp(argv[i], "-cl") == 0){
      args.csfLabel=atoi(argv[++i]);
      std::cout << "Set -cl=" << niftk::ConvertToString(args.csfLabel) << std::endl;
    }
    else if(strcmp(argv[i], "-wl") == 0){
      args.whiteLabel=atoi(argv[++i]);
      std::cout << "Set -wl=" << niftk::ConvertToString(args.whiteLabel) << std::endl;
    }
    else if(strcmp(argv[i], "-pg") == 0){
      args.minPG=atof(argv[++i]);
      std::cout << "Set -pg=" << niftk::ConvertToString(args.minPG) << std::endl;
    }
    else if(strcmp(argv[i], "-pw") == 0){
      args.minPW=atof(argv[++i]);
      std::cout << "Set -pw=" << niftk::ConvertToString(args.minPW) << std::endl;
    }            
    else if(strcmp(argv[i], "-pc") == 0){
      args.minPC=atof(argv[++i]);
      std::cout << "Set -pc=" << niftk::ConvertToString(args.minPC) << std::endl;
    }
    else if(strcmp(argv[i], "-connected") == 0){
      args.connected=true;
      std::cout << "Set -connected=" << niftk::ConvertToString(args.connected) << std::endl;
    }   
    else if(strcmp(argv[i], "-clampGM") == 0){
      args.clampGMPVImageName=argv[++i];
      std::cout << "Set -clampGM=" << args.clampGMPVImageName << std::endl;
    }
    else if(strcmp(argv[i], "-AcostaCorrection") == 0){
      args.acostaCorrection=true;
      std::cout << "Set -AcostaCorrection=" << niftk::ConvertToString(args.acostaCorrection) << std::endl;
    }
    else if(strcmp(argv[i], "-BourgeatCorrection") == 0){
      args.bourgeatCorrection=true;
      std::cout << "Set -BourgeatCorrection=" << niftk::ConvertToString(args.bourgeatCorrection) << std::endl;
    }   
    else if (std::string(argv[i]) == "-pm") {
      args.doMaskFromThresholds = true;
      std::cout << "Set -pm=" << niftk::ConvertToString(args.doMaskFromThresholds) << std::endl;
    }
    else if (std::string(argv[i]) == "-zbg") {
      args.bgLabel = 0;
      std::cout << "Set -zbg=" << niftk::ConvertToString(true) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return EXIT_FAILURE;
    }            
  }

  args.bgLabel = args.bgLabel == 0? 0 : args.csfLabel;

  // Validate command line args
  if (args.greyPVImage.length() == 0 || args.whitePVImage.length() == 0 || args.csfPVImage.length() == 0 || args.outputLabelImage.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
  
  if (args.acostaCorrection && args.bourgeatCorrection)
    {
      std::cerr << "-AcostaCorrection and -BourgeatCorrection are mutually exclusive" << std::endl;
      return EXIT_FAILURE;
    }
  
  int dims = itk::PeekAtImageDimension(args.greyPVImage);
  int result;
  
  switch ( dims )
    {
      case 2:
        result = DoMain<2, short int>(args);
        break;
      case 3:
        result = DoMain<3, short int>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
