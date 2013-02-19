/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCommandLineHelper_cxx
#define __itkCommandLineHelper_cxx

#include "itkCommandLineHelper.h"
#include "ConversionUtils.h"
#include "itkImageIOBase.h"
#include "itkImageIOFactory.h"
#include "itkExceptionObject.h"

namespace itk
{

std::string GetExceptionString(itk::ExceptionObject& err)
  {
    return std::string("EXCEPTION:") +
           std::string(err.GetDescription()) + 
           std::string("\n\tat location:") +
           std::string(err.GetLocation());
           std::string("\n\tat file:") + 
           std::string(err.GetFile()) + 
           std::string("\n\tline:") +
           std::string(niftk::ConvertToString((int)err.GetLine()));
  }

void InitialiseImageIO(std::string filename, ImageIOBase::Pointer& imageIO)
{
    try
    {
       imageIO = itk::ImageIOFactory::CreateImageIO(filename.c_str(), itk::ImageIOFactory::ReadMode);

       if ( imageIO )
       {
          imageIO->SetFileName(filename.c_str());
          imageIO->ReadImageInformation();
       }
       else
       {
         std::string msg("Failed to instantiate the imageIO object");
         //niftkitkErrorMacro(<< msg);
         ::itk::ExceptionObject e_(__FILE__, __LINE__, msg.c_str(), ITK_LOCATION);
         throw e_;
       }
    }
    catch( itk::ExceptionObject& err )
    {
      std::string msg("Failed to determine image dimension due to:\n" + GetExceptionString(err));
      //niftkitkErrorMacro(<< msg);
      throw err;
    }

}

int PeekAtImageDimension(std::string filename)
  {
    itk::ImageIOBase::Pointer imageIO;
    InitialiseImageIO(filename, imageIO);

    int numberOfDimensions = imageIO->GetNumberOfDimensions();
    //niftkitkDebugMacro(<< "Image in file " + filename + " has " + niftk::ConvertToString((int)numberOfDimensions) + " dimensions");

    return numberOfDimensions; 
  }

ImageIOBase::IOComponentType PeekAtComponentType(std::string filename)
  {
    itk::ImageIOBase::Pointer imageIO;
    InitialiseImageIO(filename, imageIO);

    ImageIOBase::IOComponentType componentType = imageIO->GetComponentType();
    //niftkitkDebugMacro(<<"Image in file " + filename + " has component type:" + imageIO->GetComponentTypeAsString(componentType));

    return componentType;
  }

ImageIOBase::IOPixelType PeekAtPixelType(std::string filename)
  {
    itk::ImageIOBase::Pointer imageIO;
    InitialiseImageIO(filename, imageIO);

    ImageIOBase::IOPixelType pixelType = imageIO->GetPixelType();
    //niftkitkDebugMacro(<<"Image in file " + filename + " has pixel type:" + imageIO->GetPixelTypeAsString(pixelType));

    return pixelType;
  }

int PeekAtImageDimensionFromSizeInVoxels(std::string filename)
  {
    int numberOfDimensions = 0;

    itk::ImageIOBase::Pointer imageIO;

    InitialiseImageIO(filename, imageIO);

    numberOfDimensions = imageIO->GetNumberOfDimensions();

    switch (numberOfDimensions) 
    {
    case 1: {
      return 1;
      break;
    }
    case 2: {
      unsigned int nVoxelsInX = imageIO->GetDimensions(0);
      unsigned int nVoxelsInY = imageIO->GetDimensions(1);
      if ((nVoxelsInX > 1) && (nVoxelsInY > 1))
        return 2;
      else
        return 1;
      break;
    }
    case 3: {
      unsigned int nVoxelsInX = imageIO->GetDimensions(0);
      unsigned int nVoxelsInY = imageIO->GetDimensions(1);
      unsigned int nVoxelsInZ = imageIO->GetDimensions(2);
      if ((nVoxelsInX > 1) && (nVoxelsInY > 1) && (nVoxelsInZ > 1))
        return 3;
      else if ((nVoxelsInX > 1) && (nVoxelsInY > 1))
        return 2;
      else
        return 1;
      break;      
    }
    case 4: {
      unsigned int nVoxels1 = imageIO->GetDimensions(0);
      unsigned int nVoxels2 = imageIO->GetDimensions(1);
      unsigned int nVoxels3 = imageIO->GetDimensions(2);
      unsigned int nVoxels4 = imageIO->GetDimensions(3);
      if ((nVoxels1 > 1) && (nVoxels2 > 1) && (nVoxels3 > 1) && (nVoxels4 > 1))
        return 4;
      else if ((nVoxels1 > 1) && (nVoxels2 > 1) && (nVoxels3 > 1))
        return 3;
      else if ((nVoxels1 > 1) && (nVoxels2 > 1))
        return 2;
      else
        return 1;
      break;      
    }
    }

  return 0;
  }
} // end namespace

#endif
