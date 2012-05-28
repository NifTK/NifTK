/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-08 16:23:32 +0100 (Thu, 08 Sep 2011) $
 Revision          : $Revision: 7267 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkConfig.h"
#include "mitkNifTKItkImageFileReader.h"
#include "mitkImageCast.h"
#include "itkImageFileReader.h"
#include "itksys/SystemTools.hxx"
#include "itksys/Directory.hxx"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageIOFactory.h"
#include "itkImageIORegion.h"
#include "itkImageIOBase.h"
#include "itkDRCAnalyzeImageIO3160.h"
#include "itkNiftiImageIO3201.h"

// TODO: Marc - make this load our Nifti reader called

mitk::NifTKItkImageFileReader::NifTKItkImageFileReader()
{
}

mitk::NifTKItkImageFileReader::~NifTKItkImageFileReader()
{
}

void mitk::NifTKItkImageFileReader::GenerateData()
{

  // Basic plan here:
  // If its Analyze,
  //   call NifTK specific functionality
  // Else
  //   call base class functionality

  itk::DRCAnalyzeImageIO3160::Pointer drcAnalyzeIO = itk::DRCAnalyzeImageIO3160::New();
  itk::NiftiImageIO3201::Pointer niftiIO = itk::NiftiImageIO3201::New();
  if (drcAnalyzeIO->CanReadFile(this->m_FileName.c_str()) || niftiIO->CanReadFile(this->m_FileName.c_str()))
  {
    mitk::Image::Pointer image = this->GetOutput();

    // More detailed plan:
    //   We don't know data type.
    //   So take a peek at the header,
    //   Then instantiate the right type itkFileReader
    //   Override the default (ITK) IO object, with the NifTK specific one
    //   Read image
    //   Use mitk::Image::InitializeFromItk to create the output.
    try
    {
      itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(this->m_FileName.c_str(), itk::ImageIOFactory::ReadMode);
      bool result = false;

      if (imageIO.IsNotNull())
      {
        imageIO->SetFileName(this->m_FileName.c_str());
        imageIO->ReadImageInformation();

        itk::ImageIOBase::IOComponentType componentType = imageIO->GetComponentType();
        unsigned int numberOfDimensions = imageIO->GetNumberOfDimensions();

        MITK_INFO << "NifTKItkImageFileReader::GenerateData(), Reading data using NifTK specific code, numberOfDimensions=" << numberOfDimensions << ", componentType=" << imageIO->GetComponentTypeAsString(componentType) << std::endl;

        switch(componentType)
        {
        case itk::ImageIOBase::UCHAR:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, unsigned char>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, unsigned char>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::CHAR:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, char>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, char>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::USHORT:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, unsigned short>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, unsigned short>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::SHORT:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, short>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, short>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::UINT:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, unsigned int>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, unsigned int>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::INT:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, int>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, int>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::ULONG:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, unsigned long>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, unsigned long>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::LONG:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, long>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, long>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::FLOAT:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, float>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, float>(image, this->m_FileName);
            }
          break;
        case itk::ImageIOBase::DOUBLE:
          if (numberOfDimensions == 2)
            {
              result = LoadImageUsingItk<2, double>(image, this->m_FileName);
            }
          else
            {
              result = LoadImageUsingItk<3, double>(image, this->m_FileName);
            }
          break;
        default:
          std::cerr << "non standard pixel format" << std::endl;
        }
      }

      if (!result)
      {
        MITK_ERROR << "Reading file " << this->m_FileName << " failed" << std::endl;
        // Default to normal to avoid crashing.
        ItkImageFileReader::GenerateData();
      }
    }
    catch( itk::ExceptionObject& err )
    {
      std::string msg = std::string("Failed to Load image:") \
        + this->m_FileName \
        + std::string("\nEXCEPTION:") \
        + std::string(err.GetDescription()) \
        + std::string("\n\tat location:") \
        + std::string(err.GetLocation()) \
        + std::string("\n\tat file:") \
        + std::string(err.GetFile()) \
      ;
      MITK_ERROR << msg;

      // Default to normal to avoid crashing.
      ItkImageFileReader::GenerateData();
    }
  }
  else
  {
    MITK_INFO << "NifTKItkImageFileReader::GenerateData(), reverting to ItkImageFileReader::GenerateData()" << std::endl;
    ItkImageFileReader::GenerateData();
  }
}

template<unsigned int VImageDimension, typename TPixel>
bool
mitk::NifTKItkImageFileReader
::LoadImageUsingItk(mitk::Image::Pointer mitkImage, std::string fileName)
{
  MITK_INFO << "NifTKItkImageFileReader::LoadImageUsingItk loading filename=" << fileName << std::endl;

  bool result = false;

  typename itk::DRCAnalyzeImageIO3160::Pointer drcAnalyzeIO = itk::DRCAnalyzeImageIO3160::New();
  typename itk::NiftiImageIO3201::Pointer niftiIO = itk::NiftiImageIO3201::New();

  if (drcAnalyzeIO->CanReadFile(this->m_FileName.c_str())){
      try
      {
        typedef itk::Image<TPixel, VImageDimension> ImageType;

        typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();

        reader->SetImageIO(drcAnalyzeIO);
        reader->SetFileName(fileName);
        reader->Update();

        mitk::CastToMitkImage<ImageType>(reader->GetOutput(), mitkImage);

        MITK_INFO << "NifTKItkImageFileReader::LoadImageUsingItk finished" << std::endl;
        result = true;
      }
      catch( itk::ExceptionObject& err )
      {
        std::string msg = std::string("Failed to Load image:") \
          + this->m_FileName \
          + std::string("\nEXCEPTION:") \
          + std::string(err.GetDescription()) \
          + std::string("\n\tat location:") \
          + std::string(err.GetLocation()) \
          + std::string("\n\tat file:") \
          + std::string(err.GetFile()) \
        ;
        MITK_ERROR << msg;
      }
  }
  else if(niftiIO->CanReadFile(this->m_FileName.c_str())){
      try
      {
        typedef itk::Image<TPixel, VImageDimension> ImageType;

        typename itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();

        reader->SetImageIO(niftiIO);
        reader->SetFileName(fileName);
        reader->Update();

        mitk::CastToMitkImage<ImageType>(reader->GetOutput(), mitkImage);

        MITK_INFO << "NifTKItkImageFileReader::LoadImageUsingItk finished" << std::endl;
        result = true;
      }
      catch( itk::ExceptionObject& err )
      {
        std::string msg = std::string("Failed to Load image:") \
          + this->m_FileName \
          + std::string("\nEXCEPTION:") \
          + std::string(err.GetDescription()) \
          + std::string("\n\tat location:") \
          + std::string(err.GetLocation()) \
          + std::string("\n\tat file:") \
          + std::string(err.GetFile()) \
        ;
        MITK_ERROR << msg;
      }

  }

  return result;
}
