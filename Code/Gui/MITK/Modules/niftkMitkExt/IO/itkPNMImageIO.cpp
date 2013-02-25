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
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>

namespace itk
{

//extern "C"
//{
//  #include <setjmp.h>
//  /* The PNM library does not expect the error function to return.
//     Therefore we must use this ugly longjmp call.  */
//  void itkPNMWriteErrorFunction(PNM_structp PNM_ptr,
//                                PNM_const_charp itkNotUsed(error_msg))
//    {
//    longjmp(PNM_ptr->jmpbuf, 1);
//    }
//}
//
//
//extern "C"
//{
//  void itkPNMWriteWarningFunction(PNM_structp itkNotUsed(PNM_ptr),
//                                  PNM_const_charp itkNotUsed(warning_msg))
//    {
//    }
//}


// simple class to call fopen on construct and
// fclose on destruct
class PNMFileWrapper
{
public:
  PNMFileWrapper(const char * const fname, const char * const openMode):m_FilePointer(NULL)
    {
    m_FilePointer = fopen(fname, openMode);
    }
  virtual ~PNMFileWrapper()
    {
    if(m_FilePointer)
      {
      fclose(m_FilePointer);
      }
    }
  FILE* m_FilePointer;
};

bool PNMImageIO::CanReadFile(const char* file) 
{ 
  // First check the extension
  std::string filename = file;
  if(  filename == "" )
    {
    itkDebugMacro(<<"No filename specified.");
    return false;
    }

  this->AddSupportedWriteExtension(".PBM");
  this->AddSupportedWriteExtension(".pbm");
  this->AddSupportedWriteExtension(".PGM");
  this->AddSupportedWriteExtension(".pgm");
  this->AddSupportedWriteExtension(".PPM");
  this->AddSupportedWriteExtension(".ppm");

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

//  unsigned char header[8];
//  fread(header, 1, 8, fp);
//  bool is_PNM = !PNM_sig_cmp(header, 0, 8);
//  if(!is_PNM)
//    {
//    itkExceptionMacro("File is not PNM type: " << this->GetFileName());
//    return;
//    }
//  PNM_structp PNM_ptr = PNM_create_read_struct
//    (PNM_LIBPNM_VER_STRING, (PNM_voidp)NULL,
//     NULL, NULL);
//  if (!PNM_ptr)
//    {
//    itkExceptionMacro("File is not PNM type" << this->GetFileName());
//    return;
//    }
//  
//  PNM_infop info_ptr = PNM_create_info_struct(PNM_ptr);
//  if (!info_ptr)
//    {
//    PNM_destroy_read_struct(&PNM_ptr,
//                            (PNM_infopp)NULL, (PNM_infopp)NULL);
//    itkExceptionMacro("File is not PNM type " << this->GetFileName());
//    return;
//    }
//
//  PNM_infop end_info = PNM_create_info_struct(PNM_ptr);
//  if (!end_info)
//    {
//    PNM_destroy_read_struct(&PNM_ptr, &info_ptr,
//                            (PNM_infopp)NULL);
//    itkExceptionMacro("File is not PNM type " << this->GetFileName());
//    return;
//    }
//  
//  //  VS 7.1 has problems with setjmp/longjmp in C++ code
//#if !defined(MSC_VER) || _MSC_VER != 1310
//  if( setjmp( PNM_jmpbuf( PNM_ptr ) ) )
//    {
//    PNM_destroy_read_struct( &PNM_ptr, &info_ptr, &end_info );
//    itkExceptionMacro("File is not PNM type " << this->GetFileName());
//    return;
//    }
//#endif
//
//  PNM_init_io(PNM_ptr, fp);
//  PNM_set_sig_bytes(PNM_ptr, 8);
//
//  PNM_read_info(PNM_ptr, info_ptr);
//
//  PNM_uint_32 width, height;
//  int bitDepth, colorType, interlaceType;
//  int compression_type, filter_method;
//  PNM_get_IHDR(PNM_ptr, info_ptr, 
//               &width, &height,
//               &bitDepth, &colorType, &interlaceType,
//               &compression_type, &filter_method);
//
//  // convert palettes to RGB
//  if (colorType == PNM_COLOR_TYPE_PALETTE)
//    {
//    PNM_set_palette_to_rgb(PNM_ptr);
//    }
//
//  // minimum of a byte per pixel
//  if (colorType == PNM_COLOR_TYPE_GRAY && bitDepth < 8) 
//    {
//    PNM_set_gray_1_2_4_to_8(PNM_ptr);
//    }
//
//  // add alpha if any alpha found
//  if (PNM_get_valid(PNM_ptr, info_ptr, PNM_INFO_tRNS)) 
//    {
//    PNM_set_tRNS_to_alpha(PNM_ptr);
//    }
//
//  if (bitDepth > 8)
//    {
//#ifndef ITK_WORDS_BIGENDIAN
//    PNM_set_swap(PNM_ptr);
//#endif
//    }
//
//  if (info_ptr->valid & PNM_INFO_sBIT)
//    {
//    PNM_set_shift(PNM_ptr, &(info_ptr->sig_bit));
//    }
//  // have libPNM handle interlacing
//  //int number_of_passes = PNM_set_interlace_handling(PNM_ptr);
//  // update the info now that we have defined the filters
//  PNM_read_update_info(PNM_ptr, info_ptr);
//
//  unsigned long rowbytes = PNM_get_rowbytes(PNM_ptr, info_ptr);
//  unsigned char *tempImage = static_cast<unsigned char*>(buffer);
//  PNM_bytep *row_pointers = new PNM_bytep [height];
//  for (unsigned int ui = 0; ui < height; ++ui)
//    {
//    row_pointers[ui] = tempImage + rowbytes*ui;
//    }
//  PNM_read_image(PNM_ptr, row_pointers);
//  delete [] row_pointers;
//  // close the file
//  PNM_read_end(PNM_ptr, NULL);
//  PNM_destroy_read_struct(&PNM_ptr, &info_ptr, &end_info);

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
  std::string filename = name;

  if (filename == "")
    {
    return false;
    }
  
  std::string::size_type PNMPos = filename.rfind(".PNM");
  if ( (PNMPos != std::string::npos)
       && (PNMPos == filename.length() - 4) )
    {
    return true;
    }

  PNMPos = filename.rfind(".PNM");
  if ( (PNMPos != std::string::npos)
       && (PNMPos == filename.length() - 4) )
    {
    return true;
    }


  return false;
}


void PNMImageIO::WriteImageInformation(void)
{
}

void PNMImageIO::Write(const void* buffer)
{
  this->WriteSlice(m_FileName, buffer);
}

void PNMImageIO::WriteSlice(const std::string& fileName, const void* buffer)
{
//  volatile const unsigned char *outPtr = ( (const unsigned char *) buffer);
//
//  // use this class so return will call close
//  PNMFileWrapper PNMfp(fileName.c_str(),"wb");
//  FILE* fp = PNMfp.m_FilePointer;
//  if(!fp)
//    {
//    // IMPORTANT: The itkExceptionMacro() cannot be used here due to a bug in Visual
//    //            Studio 7.1 in release mode. That compiler will corrupt the RTTI type
//    //            of the Exception and prevent the catch() from recognizing it.
//    //            For details, see Bug # 1872 in the bugtracker.
//
//    ::itk::ExceptionObject excp(__FILE__, __LINE__, "Problem while opening the file.", ITK_LOCATION); 
//    throw excp; 
//    }
//
//  volatile int bitDepth;
//  switch (this->GetComponentType())
//    {
//    case UCHAR:
//      bitDepth = 8;
//      break;
//
//    case USHORT:
//      bitDepth = 16;
//      break;
//
//    default:
//      {
//      // IMPORTANT: The itkExceptionMacro() cannot be used here due to a bug in Visual
//      //            Studio 7.1 in release mode. That compiler will corrupt the RTTI type
//      //            of the Exception and prevent the catch() from recognizing it.
//      //            For details, see Bug # 1872 in the bugtracker.
//      ::itk::ExceptionObject excp(__FILE__, __LINE__, "PNM supports unsigned char and unsigned short", ITK_LOCATION);
//      throw excp; 
//      }
//    }
//  
//  PNM_structp PNM_ptr = PNM_create_write_struct
//    (PNM_LIBPNM_VER_STRING, (PNM_voidp)NULL, NULL, NULL);
//  if (!PNM_ptr)
//    {
//    itkExceptionMacro(<<"Unable to write PNM file! PNM_create_write_struct failed.");
//    }
//  
//  PNM_infop info_ptr = PNM_create_info_struct(PNM_ptr);
//  if (!info_ptr)
//    {
//    PNM_destroy_write_struct(&PNM_ptr,
//                             (PNM_infopp)NULL);
//    itkExceptionMacro(<<"Unable to write PNM file!. PNM_create_info_struct failed.");
//    }
//
//  PNM_init_io(PNM_ptr, fp);
//
////  VS 7.1 has problems with setjmp/longjmp in C++ code
//#if !defined(_MSC_VER) || _MSC_VER != 1310
//  PNM_set_error_fn(PNM_ptr, PNM_ptr,
//                   itkPNMWriteErrorFunction, itkPNMWriteWarningFunction);
//  if (setjmp(PNM_ptr->jmpbuf))
//    {
//    fclose(fp);
//    itkExceptionMacro("Error while writing Slice to file: "
//                      <<this->GetFileName()
//                      << std::endl
//                      << "Reason: "
//                      << itksys::SystemTools::GetLastSystemError());
//    return;
//    } 
//#endif
//
//  int colorType;
//  unsigned int numComp = this->GetNumberOfComponents();
//  switch ( numComp )
//    {
//    case 1: colorType = PNM_COLOR_TYPE_GRAY;
//      break;
//    case 2: colorType = PNM_COLOR_TYPE_GRAY_ALPHA;
//      break;
//    case 3: colorType = PNM_COLOR_TYPE_RGB;
//      break;
//    default: colorType = PNM_COLOR_TYPE_RGB_ALPHA;
//      break;
//    }
//  
//  PNM_uint_32 width, height;
//  double rowSpacing, colSpacing;
//  width = this->GetDimensions(0);
//  colSpacing = m_Spacing[0];
//
//  if( m_NumberOfDimensions > 1 )
//    {
//    height = this->GetDimensions(1);
//    rowSpacing = m_Spacing[1];
//    }
//  else
//    {
//    height = 1;
//    rowSpacing = 1;
//    }
//  
//  PNM_set_IHDR(PNM_ptr, info_ptr, width, height,
//               bitDepth, colorType, PNM_INTERLACE_NONE,
//               PNM_COMPRESSION_TYPE_DEFAULT, 
//               PNM_FILTER_TYPE_DEFAULT);
//  // interlaceType - PNM_INTERLACE_NONE or
//  //                 PNM_INTERLACE_ADAM7
//    
//  if(m_UseCompression)
//    {
//    // Set the image compression level.
//    PNM_set_compression_level(PNM_ptr, m_CompressionLevel); 
//    }
//
//  // write out the spacing information:
//  //      set the unit_type to unknown.  if we add units to ITK, we should
//  //          convert pixel size to meters and store units as meters (PNM
//  //          has three set of units: meters, radians, and unknown).
//  PNM_set_sCAL(PNM_ptr, info_ptr, PNM_SCALE_UNKNOWN, colSpacing,
//               rowSpacing);
//
//  //std::cout << "PNM_INFO_sBIT: " << PNM_INFO_sBIT << std::endl;
//
//  PNM_write_info(PNM_ptr, info_ptr);
//  // default is big endian
//  if (bitDepth > 8)
//    {
//#ifndef ITK_WORDS_BIGENDIAN
//    PNM_set_swap(PNM_ptr);
//#endif
//    }
//  PNM_byte **row_pointers = new PNM_byte *[height];
//  int rowInc = width*numComp*bitDepth/8;
//  for (unsigned int ui = 0; ui < height; ui++)
//    {
//    row_pointers[ui] = const_cast<PNM_byte *>(outPtr);
//    outPtr = const_cast<unsigned char *>(outPtr) + rowInc;
//    }
//  PNM_write_image(PNM_ptr, row_pointers);
//  PNM_write_end(PNM_ptr, info_ptr);
//
//  delete [] row_pointers;
//  PNM_destroy_write_struct(&PNM_ptr, &info_ptr);
}


} // end namespace itk
