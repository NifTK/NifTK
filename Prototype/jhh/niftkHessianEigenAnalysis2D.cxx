/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMultiScaleHessianBasedMeasureImageFilterTest.cxx,v $
  Language:  C++
  Date:      $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
  Version:   $Revision: 7333 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkSymmetricSecondRankTensor.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImage.h"

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "byValue", NULL, "Order the eigenvalues by value [by magnitude]."},

  {OPT_DOUBLE, "sc", "sigma", "The scales to compute [1.0]."},

  {OPT_STRING, "l1", "filename", "The image file to save the first eigen value to."},
  {OPT_STRING, "l2", "filename", "The image file to save the second eigen value to."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to perform an eigen analysis of the Hessian matrix for a 2D image.\n"
  }
};


enum {
  O_BY_VALUE,

  O_SCALE,

  O_EIGEN_VALUE_OUTPUT_1,
  O_EIGEN_VALUE_OUTPUT_2,

  O_INPUT_IMAGE
};


int main( int argc, char *argv[] )
{
  bool flgByValue = false;
  
  double scale = 1;

  std::string fileEigenValueOutput1;
  std::string fileEigenValueOutput2;

  std::string fileInputImage;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_BY_VALUE, flgByValue);

  CommandLineOptions.GetArgument(O_SCALE, scale);

  CommandLineOptions.GetArgument(O_EIGEN_VALUE_OUTPUT_1, fileEigenValueOutput1);
  CommandLineOptions.GetArgument(O_EIGEN_VALUE_OUTPUT_2, fileEigenValueOutput2);

  CommandLineOptions.GetArgument(O_INPUT_IMAGE, fileInputImage);


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  const unsigned int ImageDimension = 2;

  typedef double InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension>  InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    std::cout << "Reading the input image";
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Set up the Hessian computation filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::NumericTraits< InputPixelType >::RealType RealPixelType;

  typedef itk::SymmetricSecondRankTensor< RealPixelType, ImageDimension > HessianPixelType;
  typedef itk::Image< HessianPixelType, ImageDimension >                  HessianImageType;

  typedef itk::HessianRecursiveGaussianImageFilter< InputImageType, HessianImageType> HessianFilterType;
  
  HessianFilterType::Pointer hessianFilter;

  hessianFilter = HessianFilterType::New();

  hessianFilter->SetInput( imageReader->GetOutput() );

  hessianFilter->SetNormalizeAcrossScale( true );
  hessianFilter->SetSigma( scale );

  try
  {
    std::cout << "Computing the Hessian matrix";
    hessianFilter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  // Set up the eigen analysis filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef double EigenValueType;
  typedef itk::FixedArray< EigenValueType, ImageDimension > EigenValueArrayType;
  typedef itk::Image< EigenValueArrayType, ImageDimension > EigenValueImageType;

  typedef itk::SymmetricEigenAnalysisImageFilter< HessianImageType, EigenValueImageType > EigenAnalysisFilterType;

  EigenAnalysisFilterType::Pointer symmetricEigenValueFilter;

  symmetricEigenValueFilter = EigenAnalysisFilterType::New();
  symmetricEigenValueFilter->SetDimension( ImageDimension );

  /** Typdedefs to order eigen values. 
   * OrderByValue:      lambda_1 < lambda_2 < ....
   * OrderByMagnitude:  |lambda_1| < |lambda_2| < .....
   * DoNotOrder:        Default order of eigen values obtained after QL method
   */

  if (flgByValue)
    symmetricEigenValueFilter->OrderEigenValuesBy(EigenAnalysisFilterType::FunctorType::OrderByValue);
  else
    symmetricEigenValueFilter->OrderEigenValuesBy(EigenAnalysisFilterType::FunctorType::OrderByMagnitude);

  symmetricEigenValueFilter->SetInput( hessianFilter->GetOutput() );
  
  try
  {
    std::cout << "Computing eigen values and vectors";
    symmetricEigenValueFilter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }
  

  // Write the enhanced image
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::Image< EigenValueType, ImageDimension > OutputEigenValueImageType;
  typedef itk::ImageFileWriter< OutputEigenValueImageType > EigenFileWriterType;

  EigenValueImageType::ConstPointer eigenImage = symmetricEigenValueFilter->GetOutput();
  
  itk::ImageRegionConstIterator<EigenValueImageType> it;
  it = itk::ImageRegionConstIterator<EigenValueImageType>(eigenImage, eigenImage->GetRequestedRegion());

  // Eigen value image 1 iterator

  OutputEigenValueImageType::Pointer eigenValueImage1 = OutputEigenValueImageType::New();
  eigenValueImage1->SetSpacing( eigenImage->GetSpacing() );
  eigenValueImage1->SetBufferedRegion( eigenImage->GetRequestedRegion() );
  eigenValueImage1->Allocate();

  itk::ImageRegionIterator< OutputEigenValueImageType > oit1;
  oit1 = itk::ImageRegionIterator< OutputEigenValueImageType >(eigenValueImage1, eigenImage->GetRequestedRegion());
  oit1.GoToBegin();

  // Eigen value image 2 iterator

  OutputEigenValueImageType::Pointer eigenValueImage2 = OutputEigenValueImageType::New();
  eigenValueImage2->SetSpacing( eigenImage->GetSpacing() );
  eigenValueImage2->SetBufferedRegion( eigenImage->GetRequestedRegion() );
  eigenValueImage2->Allocate();

  itk::ImageRegionIterator< OutputEigenValueImageType > oit2;
  oit2 = itk::ImageRegionIterator< OutputEigenValueImageType >(eigenValueImage2, eigenImage->GetRequestedRegion());
  oit2.GoToBegin();

  it.GoToBegin();

  EigenValueArrayType eigenValues;

  while (!it.IsAtEnd()) {

    // Get the eigenvalues
    eigenValues = it.Get();

    oit1.Set( static_cast< EigenValueType >( eigenValues[0] ));
    oit2.Set( static_cast< EigenValueType >( eigenValues[1] ));
    
    ++it;
    ++oit1;
    ++oit2;
  }
 
  EigenFileWriterType::Pointer eigenWriter = EigenFileWriterType::New();

  eigenWriter->SetFileName( fileEigenValueOutput1 );
  eigenWriter->SetInput( eigenValueImage1 );

  try
  {
    std::cout << "Writing the first eigen values.";
    eigenWriter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }

  eigenWriter->SetFileName( fileEigenValueOutput2 );
  eigenWriter->SetInput( eigenValueImage2 );

  try
  {
    std::cout << "Writing the second eigen values.";
    eigenWriter->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }

}

