/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkINRImageIO.h"
#include <fstream>
#include <algorithm>
#include <vector>
#include <itkPixelTraits.h>
#include <itkByteSwapper.h>
#include <itkPixelTraits.h>
#include <itkzlib/zlib.h>
#include <itkUCLMacro.h>

namespace itk
{

// simple class to call fopen on construct and
// fclose on destruct
struct INRFileWrapper
{
  INRFileWrapper(const char* fname, const char *openMode)
  {
    m_FilePointer = ::gzopen( fname, openMode );
  }
  gzFile m_FilePointer;
  ~INRFileWrapper()
  {
    if(m_FilePointer)
      {
        gzclose(m_FilePointer);
      }
  }
};

// uncompressed INR File Wrapper
// only used for writing
struct uINRFileWrapper
{
  uINRFileWrapper(const char* fname, const char *openMode)
  {
    m_FilePointer = fopen(fname, openMode);
  }
  FILE* m_FilePointer;
  ~uINRFileWrapper()
  {
    if(m_FilePointer)
      {
        fclose(m_FilePointer);
      }
  }
};

bool INRImageIO::ReadHeader()
{
  INRFileWrapper inrfp(m_FileName.c_str(),"rb");
  gzFile fp = inrfp.m_FilePointer;
  if(fp == NULL)
    {
      niftkitkErrorMacro (<< "File does not exist: Aborting Read of " << m_FileName.c_str());
      return false;
    }
  char header[257];

  if(::gzread( fp, header, 256) != 256)
    return false;
  header[256] = '\0';

  char * ptr_header = header;

  int dims[4];
  float spacing[3], origin[3];

  // Read in image dimensions
  this->GetParamFromHeader( ptr_header, "XDIM", "XDIM=%d", &dims[0] );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "YDIM", "YDIM=%d", &dims[1] );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "ZDIM", "ZDIM=%d", &dims[2] );
  ptr_header = header;
  // Need to change to add support for vector INR Images
  this->GetParamFromHeader( ptr_header, "VDIM", "VDIM=%d", &dims[3] );
  ptr_header = header;

  if(dims[3] > 1)
    this->SetPixelType(ImageIOBase::VECTOR);
  else
    this->SetPixelType(ImageIOBase::SCALAR);
  this->SetNumberOfComponents(dims[3]);

  for(int i = 0; i < 3; i++)
    this->SetDimensions(i, dims[i]);

  if(dims[2] > 1)
    this->SetNumberOfDimensions(3);
  else
    this->SetNumberOfDimensions(2);

  char type[256];
  this->GetParamFromHeader( ptr_header, "TYPE", "TYPE=%s", type );

  int bitType = 0;
  if (strstr(type, "float") != NULL)
    {
      bitType = 2;
    }
  else if( (strstr(type, "fixed") != NULL) && (strstr(type, "unsigned") != NULL) )
    {
      bitType = 0;
    }
  else if(strstr(type, "unsigned") != NULL)
    {
      bitType = 0;
    }
  else if( (strstr(type, "fixed") != NULL) && (strstr(type, "signed") != NULL) )
    {
      bitType = 1;
    }
  else if( (strstr(type, "fixed") != NULL) )
    {
      bitType = 1;
    }
  else if(strstr(type, "signed") != NULL)
    {
      bitType = 1;
    }
  else
    {
      niftkitkDebugMacro ("Unsuported byte type?: " << m_FileName.c_str());
      return false;
    }

  // Find the pixel size thing
  this->GetParamFromHeader( ptr_header, "PIXSIZE", "PIXSIZE=%d", &m_pixelSize, 8 );
  ptr_header = header;

  // Find the anisotropy
  this->GetParamFromHeader( ptr_header, "VX", "VX=%f", &spacing[0], 1 );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "VY", "VY=%f", &spacing[1], 1 );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "VZ", "VZ=%f", &spacing[2], 1 );
  ptr_header = header;
  for(unsigned int i = 0; i < GetNumberOfDimensions(); i++)
    {
      SetSpacing(i, spacing[i]);
    }

  // Find the origin
  this->GetParamFromHeader( ptr_header, "OX", "OX=%f", &origin[0], 0 );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "OY", "OY=%f", &origin[1], 0 );
  ptr_header = header;
  this->GetParamFromHeader( ptr_header, "OZ", "OZ=%f", &origin[2], 0 );
  ptr_header = header;
  for(unsigned int i = 0; i < GetNumberOfDimensions(); i++)
    {
      this->SetOrigin(i, origin[i]);
    }

  // Need to add check on fixed/float type etc
  if (bitType == 2)
    {
      if (m_pixelSize <= 32)
        {
          this->SetComponentType(FLOAT);
        }
      else if (m_pixelSize <= 64)
        {
          this->SetComponentType(DOUBLE);
        }
      else
        {
          niftkitkErrorMacro (<< "Pixel Size wrong for float/double: " << m_pixelSize);
          return false;
        }
    }

  if (bitType == 1)
    {
      if (m_pixelSize <= 8)
        {
          this->SetComponentType(CHAR);
        }
      else if (m_pixelSize <= 16)
        {
          this->SetComponentType(SHORT);
        }
      else if (m_pixelSize <= 32)
        {
          this->SetComponentType(INT);
        }
      else if (m_pixelSize <= 64)
        {
          this->SetComponentType(LONG);
        }
      else
        {
          niftkitkErrorMacro (<< "File does not exist?: " << m_FileName.c_str());
          return false;
        }
    }

  if (bitType == 0)
    {
      if (m_pixelSize <= 8)
        {
          this->SetComponentType(UCHAR);
        }
      else if (m_pixelSize <= 16)
        {
          this->SetComponentType(USHORT);
          //std::cout << "Unsigned Short" << std::endl;
        }
      else if (m_pixelSize <= 32)
        {
          this->SetComponentType(UINT);
        }
      else if (m_pixelSize <= 64)
        {
          this->SetComponentType(ULONG);
        }
      else
        {
          niftkitkErrorMacro (<< "Component Type Not valid: " << m_pixelSize);
          return false;
        }
    }

  // Check Little or Big Endian
  char endianType[256];
  this->GetParamFromHeader( ptr_header, "CPU", "CPU=%s", endianType );
  m_LittleEndian = true;
  m_ByteOrder = LittleEndian;
  if (strstr(endianType, "sun") != NULL)
    {
      m_LittleEndian = false;
      m_ByteOrder = BigEndian;
      std::cout << "Sun computer created data" << std::endl;
    }
  else if (strstr(endianType, "sgi") != NULL)
    {
      m_LittleEndian = false;
      m_ByteOrder = BigEndian;
      std::cout << "SGI computer created data" << std::endl;
    }

  return true;
}


bool INRImageIO::CanReadFile(const char* file)
{
  std::string filename(file);
  // First check the extension
  std::string::size_type inrPos = filename.rfind(".inr");
  if ((inrPos != std::string::npos)
      && (inrPos == filename.length() - 4))
    {
      return true;
    }
  inrPos = filename.rfind(".INR");
  if ((inrPos != std::string::npos)
      && (inrPos == filename.length() - 4))
    {
      return true;
    }

  // Header file given?
  std::string::size_type inrgzPos = filename.rfind(".inr.gz");
  if ((inrgzPos != std::string::npos)
      && (inrgzPos == filename.length() - 7))
    {
      return true;
    }
  inrgzPos = filename.rfind(".INR.gz");
  if ((inrgzPos != std::string::npos)
      && (inrgzPos == filename.length() - 7))
    {
      return true;
    }
  return false;
}

unsigned int INRImageIO::GetComponentSize() const
{
  switch(m_ComponentType)
    {
    case UCHAR:
      return sizeof(unsigned char);
    case USHORT:
      return sizeof(unsigned short);
    case CHAR:
      return sizeof(char);
    case SHORT:
      return sizeof(short);
    case UINT:
      return sizeof(unsigned int);
    case INT:
      return sizeof(int);
    case ULONG:
      return sizeof(unsigned long);
    case LONG:
      return sizeof(long);
    case FLOAT:
      return sizeof(float);
    case DOUBLE:
      return sizeof(double);
    case UNKNOWNCOMPONENTTYPE:
    default:
    {
      niftkitkErrorMacro (<< "Invalid type: " << m_PixelType
                       << ", only signed and unsigned char, short, int, long and float/double are allowed.");
    return 0;
    }
    }
  return 1;
}

void INRImageIO::ReadVolume(void*)
{

}

void INRImageIO::Read(void* buffer)
{
  //std::cout << "Read: file dimensions = " << this->GetNumberOfDimensions() << std::endl;
  //std::ifstream file;
  INRFileWrapper inrfp(m_FileName.c_str(),"rb");
  gzFile file = inrfp.m_FilePointer;
/*
  if ( file.is_open() )
    {
      file.close();
    }
  file.open(m_FileName.c_str(), std::ios::in | std::ios::binary);

  if ( file.fail() )
    {
      niftkitkErrorMacro(<< "Could not open file: " << m_FileName);
    }
*/

  // Skip header
  //file.seekg((long)256, std::ios::beg);
  ::gzseek (file, 256, 0);
  /*
  if ( file.fail() )
    {
      niftkitkErrorMacro(<<"File seek failed");
    }
  */

  const unsigned long numberOfBytesToBeRead =
    static_cast< unsigned long>( this->GetImageSizeInBytes() );

  niftkitkDebugMacro("Reading " << numberOfBytesToBeRead << " bytes");

  if(m_LittleEndian == false)
    this->SetByteOrderToBigEndian();
  else
    this->SetByteOrderToLittleEndian();

  if ( m_FileType == Binary )
    {
      ::gzread(file, buffer, numberOfBytesToBeRead);
    /*
    if( !this->ReadBufferAsBinary( file, buffer, numberOfBytesToBeRead ) )
      {
        niftkitkErrorMacro(<<"Read failed: Wanted "
                        << numberOfBytesToBeRead
                        << " bytes, but read "
                        << file.gcount() << " bytes.");
      }
      */
    }
  else
    {
    //this->ReadBufferAsASCII(file, buffer, this->GetComponentType(), this->GetImageSizeInComponents());
    }

  this->SwapBytesIfNecessary(buffer, this->GetImageSizeInComponents());
  //std::cout << "Reading Done" << std::endl;
  niftkitkDebugMacro("Reading Done");
}


INRImageIO::INRImageIO()
{
  this->SetNumberOfDimensions(3);
  m_PixelType = SCALAR;
  m_ComponentType = UCHAR;
  m_Spacing[0] = 1.0;
  m_Spacing[1] = 1.0;
  m_Spacing[2] = 1.0;

  m_Origin[0] = 0.0;
  m_Origin[1] = 0.0;
  m_Origin[2] = 0.0;

  // Intel PC's are Little endian
  m_ByteOrder = ImageIOBase::LittleEndian;
  m_LittleEndian = true;
  m_FileType = Binary;
}

INRImageIO::~INRImageIO()
{
}

/***************************
 *
 * The following functions look for paremeters in the header. Functions exit if no information on
 * a field is found.
 *
***************************/

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, int *target )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "Failed to read '" << patternToFind << "' - bailing out!" << std::endl;
  return false;
  }

  free(bufferPtr);
  return true;
}

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, char *target )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "Failed to read '" << patternToFind << "' - bailing out!" << std::endl;
  return false;
  }

  free(bufferPtr);
  return true;
}

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, float *target )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "Failed to read '" << patternToFind << "' - bailing out!" << std::endl;
  return false;
  }

  free(bufferPtr);
  return true;
}

/***************************
 *
 * The following functions look for paremeters in the header and are supplied a default value if no information
 * is found.
 *
***************************/

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, int *target, int defaultValue )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "No information on '" << variableToFind << "' - Defaulting to" <<  defaultValue << "!" << std::endl;
    *target = defaultValue;
  }

  free(bufferPtr);
  return true;
}

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, unsigned int *target, unsigned int defaultValue )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "No information on '" << variableToFind << "' - Defaulting to" <<  defaultValue << "!" << std::endl;
    *target = defaultValue;
  }

  free(bufferPtr);
  return true;
}

bool INRImageIO::GetParamFromHeader( char *headerPtr, const char *variableToFind, const char *patternToFind, float *target, float defaultValue )
{
  char *bufferPtr;
  bufferPtr = (char*)malloc(256);

  while( sscanf( headerPtr, "%s", bufferPtr ) > 0 )
  {
    if( strstr(bufferPtr, variableToFind ) != NULL )
    break;
    headerPtr += strlen(bufferPtr);
  }

  //the pointer to the header information should be at the right location now :)
  if( sscanf( bufferPtr, patternToFind, target ) == 0 )
  {
    //std::cerr << "No information on '" << variableToFind << "' - Defaulting to" <<  defaultValue << "!" << std::endl;
    *target = defaultValue;
  }

  free(bufferPtr);
  return true;
}

void INRImageIO::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  //os << indent << "PixelType " << m_PixelType << "\n";
  //os << indent << "Dimensions (" << this->GetDimensions(0) << "," << this->GetDimensions(1) << "," << this->GetDimensions(2) << ")\n";
}

void INRImageIO::ReadImageInformation()
{
  // Now check the file header
  bool check = this->ReadHeader();
  if(check == false)
    std::cout << "Failed to read header" << std::endl;
  return;
}

bool INRImageIO::CanWriteFile( const char * name )
{
  std::string filename = name;
  m_compression = false;
  if (filename == "")
    {
    return false;
    }

  std::string::size_type inrPos = filename.rfind(".inr");
  if ( (inrPos != std::string::npos)
       && (inrPos == filename.length() - 4) )
    {
      m_compression = false;
      return true;
    }

  inrPos = filename.rfind(".inr.gz");
  if ( (inrPos != std::string::npos)
       && (inrPos == filename.length() - 7) )
    {
      m_compression = true;
      return true;
    }

  inrPos = filename.rfind(".INR");
  if ( (inrPos != std::string::npos)
       && (inrPos == filename.length() - 4) )
    {
      m_compression = false;
      return true;
    }

  inrPos = filename.rfind(".INR.GZ");
  if ( (inrPos != std::string::npos)
       && (inrPos == filename.length() - 7) )
    {
      m_compression = true;
      return true;
    }

  return false;
}


void INRImageIO::WriteImageInformation(void)
{
}

void INRImageIO::Write(const void* buffer)
{
  this->CanWriteFile(m_FileName.c_str());

  this->WriteSlice(m_FileName, buffer);
}

void INRImageIO::WriteSlice(std::string& fileName, const void* buffer)
{
  char pixelType[256];
  if ( (this->GetComponentTypeInfo() == typeid(unsigned char)) || (this->GetComponentTypeInfo() == typeid(unsigned short)) || (this->GetComponentTypeInfo() == typeid(unsigned int)) || (this->GetComponentTypeInfo() == typeid(unsigned long)))
    strcpy(pixelType, "unsigned fixed");
  else if((this->GetComponentTypeInfo() == typeid(char)) || (this->GetComponentTypeInfo() == typeid(short)) || (this->GetComponentTypeInfo() == typeid(int)) || (this->GetComponentTypeInfo() == typeid(long)))
    strcpy(pixelType, "signed fixed");
 else if ((this->GetComponentTypeInfo() == typeid(float)) || (this->GetComponentTypeInfo() ==typeid(double)))
    strcpy(pixelType, "float");
 else
   {
     // Bail if not valid -> Need to send abort ;)
     std::cout << "Not a valid pixel type" << std::endl;
     niftkitkDebugMacro("Not a valid INR pixel type found to write");
     return;
   }

  char header[260];

  if ((int)this->GetNumberOfDimensions() == 2)
  {
  sprintf(header, "#INRIMAGE-4#{\nXDIM=%u\nYDIM=%u\nZDIM=1\nVDIM=%u\nTYPE=%s\nPIXSIZE=%u bits\nCPU=decm\nVX=%g\nVY=%g\nVZ=%g\n", (int)this->GetDimensions(0), (int)this->GetDimensions(1), (int)this->GetNumberOfComponents(), pixelType,(int)(8*this->GetPixelSize()), this->GetSpacing(0), this->GetSpacing(1), this->GetSpacing(2));
  }
  else
  {
  sprintf(header, "#INRIMAGE-4#{\nXDIM=%u\nYDIM=%u\nZDIM=%u\nVDIM=%u\nTYPE=%s\nPIXSIZE=%u bits\nCPU=decm\nVX=%g\nVY=%g\nVZ=%g\n", (int)this->GetDimensions(0), (int)this->GetDimensions(1), (int)this->GetDimensions(2), (int)this->GetNumberOfComponents(), pixelType,(int)(8*this->GetPixelSize()), this->GetSpacing(0), this->GetSpacing(1), this->GetSpacing(2));
  }
  unsigned int h_length = strlen(header);
  unsigned long i=0;
  for (i = 0; i < 252-h_length; i++)
    strcat(header, "\n");
  strcat(header, "##}\n");

  const unsigned long numberOfBytes = this->GetImageSizeInBytes();
  niftkitkDebugMacro("Writing " << numberOfBytes << "to inr file");

  if(m_compression == false)
    {
      std::ofstream fout(fileName.c_str(), std::ios::binary | std::ios::trunc);

      fout.write((char *)(header), 256);
      fout.write(static_cast<const char*>(buffer), numberOfBytes);

      if (fout.fail()) {
    	  niftkitkErrorMacro(<< "Writing to file " << fileName << " failed.");
      }

      fout.close();
    }
  else
    {
      INRFileWrapper inrfp(fileName.c_str(),"wb");
      gzFile file = inrfp.m_FilePointer;
      std::vector<char> tempBuffer(numberOfBytes);

      std::copy(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + numberOfBytes, tempBuffer.begin());
      ::gzsetparams (file, 9, 0);
      ::gzwrite(file, (char *)(header), 256);
      //uint bytesWritten = 
      ::gzwrite(file, &tempBuffer.front(), numberOfBytes);
    }
}

/**
 * Warning this code is stole from DicomImageIO.cxx
 *  - Been extended to support all the types inr supports
 */
void
INRImageIO
::SwapBytesIfNecessary( void* buffer, unsigned long numberOfPixels )
{
  switch(m_ComponentType)
    {
    case CHAR:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<char>::SwapRangeFromSystemToLittleEndian(
        (char*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<char>::SwapRangeFromSystemToBigEndian(
        (char *)buffer, numberOfPixels );
      }
    break;
    }
    case UCHAR:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<unsigned char>::SwapRangeFromSystemToLittleEndian(
        (unsigned char*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<unsigned char>::SwapRangeFromSystemToBigEndian(
        (unsigned char *)buffer, numberOfPixels );
      }
    break;
    }
    case SHORT:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<short>::SwapRangeFromSystemToLittleEndian(
        (short*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<short>::SwapRangeFromSystemToBigEndian(
        (short *)buffer, numberOfPixels );
      }
    break;
    }
    case USHORT:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<unsigned short>::SwapRangeFromSystemToLittleEndian(
        (unsigned short*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<unsigned short>::SwapRangeFromSystemToBigEndian(
        (unsigned short *)buffer, numberOfPixels );
      }
    break;
    }
    case INT:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<int>::SwapRangeFromSystemToLittleEndian(
        (int*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<int>::SwapRangeFromSystemToBigEndian(
        (int *)buffer, numberOfPixels );
      }
    break;
    }
    case UINT:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<unsigned int>::SwapRangeFromSystemToLittleEndian(
        (unsigned int*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<unsigned int>::SwapRangeFromSystemToBigEndian(
        (unsigned int *)buffer, numberOfPixels );
      }
    break;
    }
    case LONG:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<long int>::SwapRangeFromSystemToLittleEndian(
        (long int*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<long int>::SwapRangeFromSystemToBigEndian(
        (long int *)buffer, numberOfPixels );
      }
    break;
    }
    case ULONG:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<unsigned long int>::SwapRangeFromSystemToLittleEndian(
        (unsigned long int*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<unsigned long int>::SwapRangeFromSystemToBigEndian(
        (unsigned long int *)buffer, numberOfPixels );
      }
    break;
    }
    case FLOAT:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<float>::SwapRangeFromSystemToLittleEndian(
        (float*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<float>::SwapRangeFromSystemToBigEndian(
        (float *)buffer, numberOfPixels );
      }
    break;
    }
    case DOUBLE:
    {
    if ( m_ByteOrder == LittleEndian )
      {
      ByteSwapper<double>::SwapRangeFromSystemToLittleEndian(
        (double*)buffer, numberOfPixels );
      }
    else if ( m_ByteOrder == BigEndian )
      {
      ByteSwapper<double>::SwapRangeFromSystemToBigEndian(
        (double *)buffer, numberOfPixels );
      }
    break;
    }
    default:
      ExceptionObject exception(__FILE__, __LINE__);
      exception.SetDescription("Pixel Type Unknown");
      throw exception;
    }
}



} // end namespace itk

