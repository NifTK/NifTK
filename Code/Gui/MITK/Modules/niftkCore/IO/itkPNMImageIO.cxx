/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkPNMImageIO.cxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkPNMImageIO.h"
#include <itkRGBPixel.h>
#include <itkRGBAPixel.h>
#include <itksys/SystemTools.hxx>
#include <vtkPNMReader.h>
#include <vtkPNMWriter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkImageImport.h>
#include <vtkUnsignedCharArray.h>
#include <vtkSmartPointer.h>

namespace itk
{

bool PNMImageIO::CanReadFile(const char* file)
{
  // First check the extension
  std::string filename = file;
  if (filename == "")
  {
    itkDebugMacro(<<"No filename specified.");
    return false;
  }

  this->AddSupportedReadExtension(".PBM");
  this->AddSupportedReadExtension(".pbm");
  this->AddSupportedReadExtension(".PGM");
  this->AddSupportedReadExtension(".pgm");
  this->AddSupportedReadExtension(".PPM");
  this->AddSupportedReadExtension(".ppm");

  vtkPNMReader * pnmReader = vtkPNMReader::New();

  int status = pnmReader->CanReadFile(file);
  pnmReader->Delete();
  pnmReader = 0;

  if (status <= 0)
    return false;

  return true;
}


void PNMImageIO::ReadVolume(void*)
{

}


void PNMImageIO::Read(void* buffer)
{
  itkDebugMacro("Read: file dimensions = " << this->GetNumberOfDimensions() );

  if (!this->CanReadFile(this->GetFileName()))
  {
    itkExceptionMacro("PNMImageIO could not open file: "
                      << this->GetFileName() << " for reading."
                      << std::endl
                      << "Reason: "
                      << itksys::SystemTools::GetLastSystemError());
    return;
  }

  vtkPNMReader * pnmReader = vtkPNMReader::New();
  pnmReader->SetFileName(this->GetFileName());
  pnmReader->Update();
  vtkImageData * imageData = pnmReader->GetOutput();
  imageData->GetScalarSize();
  imageData->GetScalarType();
  imageData->GetScalarTypeAsString();
  imageData->GetScalarPointer();

  vtkUnsignedCharArray *arr = vtkUnsignedCharArray::SafeDownCast(imageData->GetPointData()->GetScalars());
  unsigned char *tempImage = static_cast<unsigned char*>(buffer);

  memcpy(tempImage, arr->GetPointer(0), arr->GetSize());
  pnmReader->Delete();
}


PNMImageIO::PNMImageIO()
{
  this->SetNumberOfDimensions(2);
  m_PixelType = SCALAR;
  m_ComponentType = UCHAR;
  m_UseCompression = false;
  m_CompressionLevel = 0; // Range 0-9; 0 = no file compression, 9 = maximum file compression
  m_Spacing[0] = 1.0;
  m_Spacing[1] = 1.0;

  m_Origin[0] = 0.0;
  m_Origin[1] = 0.0;
}

PNMImageIO::~PNMImageIO()
{
}

void PNMImageIO::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Compression Level : " << m_CompressionLevel << "\n";
}



void PNMImageIO::ReadImageInformation()
{
  m_Spacing[0] = 1.0;  // We'll look for PNM pixel size information later,
  m_Spacing[1] = 1.0;  // but set the defaults now

  m_Origin[0] = 0.0;
  m_Origin[1] = 0.0;

  // use this class so return will call close
  if (!this->CanReadFile(this->GetFileName()))
  {
    itkExceptionMacro("PNMImageIO could not open file: "
                      << this->GetFileName() << " for reading."
                      << std::endl
                      << "Reason: "
                      << itksys::SystemTools::GetLastSystemError());
    return;
  }

  vtkPNMReader * pnmReader = vtkPNMReader::New();
  pnmReader->SetFileName(this->GetFileName());
  pnmReader->Update();
  vtkImageData * imageData = pnmReader->GetOutput();

  // update the info now that we have defined the filters
  this->SetNumberOfDimensions(2);
  m_Dimensions[0] = imageData->GetDimensions()[0];
  m_Dimensions[1] = imageData->GetDimensions()[1];

  int scalarType = imageData->GetScalarType();
  int pixelType  = imageData->GetNumberOfScalarComponents();

  switch (pixelType)
  {
    case 1:
      m_PixelType = SCALAR;
      break;
    case 3:
      m_PixelType = RGB;
      break;
    case 4:
      m_PixelType = RGBA;
      break;
    default:
      m_PixelType = UNKNOWNPIXELTYPE;
  }
  this->SetNumberOfComponents(pixelType);

  switch (scalarType)
  {
    case VTK_VOID:
      m_ComponentType = UNKNOWNCOMPONENTTYPE;
      break;
    case VTK_FLOAT:
      m_ComponentType = FLOAT;
      break;
    case VTK_INT:
      m_ComponentType = INT;
      break;
    case VTK_SHORT:
      m_ComponentType = SHORT;
      break;
    case VTK_UNSIGNED_CHAR:
      m_ComponentType = UCHAR;
      break;
    case VTK_UNSIGNED_SHORT:
      m_ComponentType = USHORT;
      break;
    default:
      m_ComponentType = UNKNOWNCOMPONENTTYPE;
  }

  // PNM files are usually big endian
  this->SetByteOrderToBigEndian();

  // clean up
  pnmReader->Delete();

  return;
}

bool PNMImageIO::CanWriteFile( const char * name )
{
  this->AddSupportedWriteExtension(".PBM");
  this->AddSupportedWriteExtension(".pbm");
  this->AddSupportedWriteExtension(".PGM");
  this->AddSupportedWriteExtension(".pgm");
  this->AddSupportedWriteExtension(".PPM");
  this->AddSupportedWriteExtension(".ppm");

  std::string filename = name;

  if (filename == "")
  {
    return false;
  }

  std::string::size_type PBMPos = filename.rfind(".PBM");
  if ( (PBMPos != std::string::npos) && (PBMPos == filename.length() - 4) )
    return true;

  PBMPos = filename.rfind(".pbm");
  if ( (PBMPos != std::string::npos) && (PBMPos == filename.length() - 4) )
    return true;

  std::string::size_type PGMPos = filename.rfind(".PGM");
  if ( (PGMPos != std::string::npos) && (PGMPos == filename.length() - 4) )
    return true;

  PGMPos = filename.rfind(".pgm");
  if ( (PGMPos != std::string::npos) && (PGMPos == filename.length() - 4) )
    return true;

  std::string::size_type PPMPos = filename.rfind(".PPM");
  if ( (PPMPos != std::string::npos) && (PPMPos == filename.length() - 4) )
    return true;

  PPMPos = filename.rfind(".ppm");
  if ( (PPMPos != std::string::npos) && (PPMPos == filename.length() - 4) )
    return true;

  return false;
}


void PNMImageIO::WriteImageInformation(void)
{
}

void PNMImageIO::Write(const void* buffer)
{
  this->WriteSlice(m_FileName, buffer);
}

void PNMImageIO::WriteSlice(const std::string & fileName, const void * buffer)
{
  // use this class so return will call close
  if (!this->CanWriteFile(fileName.c_str()) )
  {
    itkExceptionMacro("PNMImageIO could not open file: "
                        << this->GetFileName() << " for reading."
                        << std::endl
                        << "Reason: "
                        << itksys::SystemTools::GetLastSystemError());
    return;
  }

  vtkImageImport* imageImport = vtkImageImport::New();

  imageImport->SetWholeExtent(0, m_Dimensions[0]-1, 0, m_Dimensions[1]-1, 0, 0);
  imageImport->SetDataExtentToWholeExtent();
  imageImport->SetImportVoidPointer(const_cast<void *>(buffer));
  imageImport->SetNumberOfScalarComponents(m_NumberOfComponents);
  imageImport->SetDataSpacing(m_Spacing[0], m_Spacing[1], 1);
  imageImport->SetDataOrigin(m_Origin[0], m_Origin[1], 0);
  imageImport->SetDataScalarTypeToUnsignedChar();

  imageImport->Update();

  vtkPNMWriter * pnmWriter = vtkPNMWriter::New();
  pnmWriter->SetFileName(fileName.c_str());
  pnmWriter->SetInput(imageImport->GetOutput());
  pnmWriter->Update();

  pnmWriter->Delete();
  imageImport->Delete();
}

} // end namespace itk
