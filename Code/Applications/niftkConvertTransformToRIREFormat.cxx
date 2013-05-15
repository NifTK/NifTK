/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegistrationFactory.h>
#include <itkPoint.h>
#include <itkEulerAffineTransform.h>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fstream>

/*!
 * \file niftkConvertTransformToRIREFormat.cxx
 * \page niftkConvertTransformToRIREFormat
 * \section niftkConvertTransformToRIREFormatSummary Converts an ITK transformation file to that required by Vanderbilt's Retrospective Image Registration Evaluation project.

 * \section niftkConvertTransformToRIREFormatCaveat Caveats
 */

void Usage(char *exec)
  {
	niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Converts an ITK transformation file to that required by Vanderbilt's Retrospective Image Registration Evaluation project." << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputImage -t inputTransformationFile -o outputFileName [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i    <filename>       Input Source/Moving image " << std::endl;
    std::cout << "    -t    <filename>       ITK transformation file  " << std::endl;
    std::cout << "    -o    <filename>       Output Vanderbilt transformation file " << std::endl << std::endl;      
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -p    <string> [001]   Patient identifier (eg. 001) " << std::endl;
    std::cout << "    -f    <string> [mr_PD] Fixed image identifier " << std::endl;
    std::cout << "    -m    <string> [CT]    Moving image identifier " << std::endl;
  }

std::string padString(std::string input, int desiredLength)
  {
    std::string result = input;
    while (result.length() < 11)
      {
        result = std::string(" ") + result;
      }
    return result;
  }

int main(int argc, char** argv)
{
  const   unsigned int Dimension = 3;
  typedef short        PixelType;

  // Define command line params
  std::string imageFile;
  std::string itkTransform;
  std::string vanderbiltTransform;
  std::string patientIdentifier = "001";
  std::string fixedImageIdentifier = "mr_PD";
  std::string movingImageIdentifier = "CT";
  
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      imageFile=argv[++i];
      std::cout << "Set -i=" << imageFile<< std::endl;
    }
    else if(strcmp(argv[i], "-t") == 0){
      itkTransform=argv[++i];
      std::cout << "Set -t=" << itkTransform<< std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      vanderbiltTransform=argv[++i];
      std::cout << "Set -o=" << vanderbiltTransform<< std::endl;
    }
    else if(strcmp(argv[i], "-p") == 0){
      patientIdentifier=argv[++i];
      std::cout << "Set -p=" << patientIdentifier<< std::endl;
    }
    else if(strcmp(argv[i], "-f") == 0){
      fixedImageIdentifier=argv[++i];
      std::cout << "Set -f=" << fixedImageIdentifier<< std::endl;
    }
    else if(strcmp(argv[i], "-m") == 0){
      movingImageIdentifier=argv[++i];
      std::cout << "Set -m=" << movingImageIdentifier<< std::endl;
    }            
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (imageFile.length() == 0 || itkTransform.length() == 0 || vanderbiltTransform.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  if (patientIdentifier.length() == 0 || fixedImageIdentifier.length() == 0 || movingImageIdentifier.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;      
    }
  
  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::ImageFileReader< ImageType >   ImageReaderType;
  ImageReaderType::Pointer imageReader  = ImageReaderType::New();
  imageReader->SetFileName(  imageFile );

  // Load image.
  try 
  { 
    std::cout << "Reading image:" <<  imageFile<< std::endl;
    imageReader->Update();
    std::cout << "Done"<< std::endl;
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed to load input images: " << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
  // Setup objects to load transformation
  typedef itk::ImageRegistrationFactory<ImageType, Dimension, double> FactoryType;
  FactoryType::Pointer factory = FactoryType::New();
  FactoryType::TransformType::Pointer globalTransform; 

  try
  {
    std::cout << "Loading transformation from:" << itkTransform<< std::endl;
    globalTransform = factory->CreateTransform(itkTransform);
    
    itk::EulerAffineTransform<double, Dimension, Dimension>* affineTransform = dynamic_cast<itk::EulerAffineTransform<double, Dimension, Dimension>*>(globalTransform.GetPointer());
    affineTransform->InvertTransformationMatrix();
    
    std::cout << "Done"<< std::endl;
  }  
  catch (itk::ExceptionObject& exceptionObject)
  {
    std::cerr << "Failed to load ITK tranform:" << exceptionObject << std::endl;
    return EXIT_FAILURE; 
  }
  
  using namespace boost::gregorian;
  date today = day_clock::local_day();

  std::ofstream myfile;
  myfile.open (vanderbiltTransform.c_str());
  
  myfile << "-------------------------------------------------------------------------" << std::endl;
  myfile << "Transformation Parameters" << std::endl;
  myfile << std::endl;
  myfile << "Investigator(s): M. J. Clarkson, K. K. Leung and S. Ourselin" << std::endl;
  myfile << std::endl;
  myfile << "Dementia Research Centre, Institute Of Neurology, University College London, London, UK" << std::endl;
  myfile << std::endl;
  myfile << "Method: NifTK" << std::endl;
  myfile << "Date: " << to_iso_extended_string(today) << std::endl;
  myfile << "Patient number: " << patientIdentifier << std::endl;
  myfile << "From: " << movingImageIdentifier << std::endl;
  myfile << "To: " << fixedImageIdentifier << std::endl;
  myfile << std::endl;
  myfile << "Point      x          y          z        new_x       new_y       new_z" << std::endl;
  myfile << std::endl;
  
  typedef itk::Point<double, 3> PointType;
  ImageType::Pointer image = imageReader->GetOutput();
  ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
  ImageType::SpacingType spacing = image->GetSpacing();
  ImageType::IndexType index;
  PointType point;
  PointType transformedPoint;
  
  unsigned int counter = 1;
  for (unsigned int z = 0; z <= size[2]; z+= (size[2]-1))
    {
      for (unsigned int y = 0; y <= size[1]; y+= (size[1]-1))
        {
          for (unsigned int x = 0; x <= size[0]; x+= (size[0]-1))
            {
              index[0] = x;
              index[1] = y;
              index[2] = z;
              image->TransformIndexToPhysicalPoint( index, point );
              transformedPoint = globalTransform->TransformPoint(point);
              
              double mx = point[0];
              double my = point[1];
              double mz = point[2];

              double fx = transformedPoint[0];
              double fy = transformedPoint[1];
              double fz = transformedPoint[2];
              
              myfile << boost::format("  %s   %10.4f %10.4f %10.4f%10.4f  %10.4f  %10.4f ") % counter % mx % my %mz % fx %fy %fz << std::endl;  
              
              counter++;
            }
        }
    }
  myfile << std::endl;
  myfile << "(All distances are in millimeters.)" << std::endl;
  myfile << "-------------------------------------------------------------------------" << std::endl;
  
  myfile.close();
  return 0;
  
}

