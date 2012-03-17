/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFreeSurferBinarySurfaceReader.txx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFreeSurferBinarySurfaceReader_txx
#define __itkFreeSurferBinarySurfaceReader_txx

#include "itkFreeSurferBinarySurfaceReader.h"
#include "itkByteSwapper.h"

#include "itksys/SystemTools.hxx"

namespace itk
{

// 
// Constructor
// 
template<class TOutputMesh>
FreeSurferBinarySurfaceReader<TOutputMesh>
::FreeSurferBinarySurfaceReader()
{
  //
  // Create the output
  //
  typename TOutputMesh::Pointer output = TOutputMesh::New();
  this->ProcessObject::SetNumberOfRequiredOutputs(1);
  this->ProcessObject::SetNthOutput(0, output.GetPointer());
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::GenerateData()
{
  this->OpenGeometryFile();
  this->ReadGeometryHeader();
  this->PrepareOutputMesh();
  this->ReadSurface();
  this->CloseGeometryFile();

  if( this->DataFileIsAvailable() )
    {
    this->OpenDataFile();
    this->ReadDataHeader();
    this->ReadPointData();
    this->CloseDataFile();
    }
}


template<class TOutputMesh>
bool
FreeSurferBinarySurfaceReader<TOutputMesh>
::DataFileIsAvailable() const
{
  if( this->m_DataFileName.empty() )
    {
    return false;
    }

  if( ! itksys::SystemTools::FileExists( m_DataFileName.c_str() ) )
    {
    itkWarningMacro("File " << this->m_DataFileName << " does not exist");
    return false;
    }

  return true;
}

template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::OpenGeometryFile()
{
  if( this->m_FileName.empty() )
    {
    itkExceptionMacro("No input FileName");
    }

  if( ! itksys::SystemTools::FileExists( m_FileName.c_str() ) )
    {
    itkExceptionMacro("File " << this->m_FileName << " does not exist");
    }

  //
  // Open file 
  //
  this->m_InputGeometryFile.open( this->m_FileName.c_str() );

  if( !this->m_InputGeometryFile.is_open() )
    {
    itkExceptionMacro("Unable to open file inputFilename= " << this->m_FileName );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::CloseGeometryFile()
{
  this->m_InputGeometryFile.close();
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::OpenDataFile()
{
  //
  // Open file 
  //
  this->m_InputDataFile.open( this->m_DataFileName.c_str() );

  if( !this->m_InputDataFile.is_open() )
    {
    itkExceptionMacro("Unable to open file inputFilename= " << this->m_DataFileName );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::CloseDataFile()
{
  this->m_InputDataFile.close();
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadGeometryHeader()
{
  this->ReadFileTypeFromGeometryFile();
  this->ReadCommentFromGeometryFile();
  this->ReadNumberOfPointsFromGeometryFile();
  this->ReadNumberOfCellsFromGeometryFile();
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadFileTypeFromGeometryFile()
{
  const unsigned int fileTypeIdLength = 3;
  char fileTypeId[fileTypeIdLength];

  this->m_InputGeometryFile.read( fileTypeId, fileTypeIdLength );

  this->m_FileType = 0;
  this->m_FileType <<= 8;

  this->m_FileType |= fileTypeId[0];
  this->m_FileType <<= 8;

  this->m_FileType |= fileTypeId[1];
  this->m_FileType <<= 8;

  this->m_FileType |= fileTypeId[2];
  this->m_FileType <<= 8;

  itk::ByteSwapper<ITK_UINT32>::SwapFromSystemToBigEndian( &(this->m_FileType) );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadCommentFromGeometryFile()
{
  char byte;

  //
  //  Extract Comment, and ignore it.
  //
  byte = this->m_InputGeometryFile.get();

  this->m_Comment = "";

  while( byte != '\n' )
    {
    this->m_Comment += byte;
    byte = this->m_InputGeometryFile.get();
    }


  //
  // Try to get the second '\n', but if the '\n' is not there, we put the byte
  // back.
  //
  byte = this->m_InputGeometryFile.get();
  if( byte != '\n' )
    {
    this->m_InputGeometryFile.unget();
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadNumberOfPointsFromGeometryFile()
{
  this->ReadInteger32( this->m_InputGeometryFile, this->m_NumberOfPoints );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadNumberOfCellsFromGeometryFile()
{
  this->ReadInteger32( this->m_InputGeometryFile, this->m_NumberOfCells );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadDataHeader()
{
  this->ReadFileTypeFromDataFile();
  this->ReadNumberOfPointsFromDataFile();
  this->ReadNumberOfCellsFromDataFile();
  this->ReadNumberOfValuesPerPointFromDataFile();
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadFileTypeFromDataFile()
{
  const unsigned int fileTypeIdLength = 3;
  char fileTypeId[fileTypeIdLength];

  this->m_InputDataFile.read( fileTypeId, fileTypeIdLength );

  this->m_DataFileType = 0;
  this->m_DataFileType <<= 8;

  this->m_DataFileType |= fileTypeId[0];
  this->m_DataFileType <<= 8;

  this->m_DataFileType |= fileTypeId[1];
  this->m_DataFileType <<= 8;

  this->m_DataFileType |= fileTypeId[2];
  this->m_DataFileType <<= 8;

  itk::ByteSwapper<ITK_UINT32>::SwapFromSystemToBigEndian( &(this->m_DataFileType) );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadNumberOfPointsFromDataFile()
{
  ITK_UINT32 numberOfPointsInDataFile;
  this->ReadInteger32( this->m_InputDataFile, numberOfPointsInDataFile );

  if( numberOfPointsInDataFile != this->m_NumberOfPoints )
    {
    itkExceptionMacro("Data file has " << numberOfPointsInDataFile 
      << " points while Geometry file has " << this->m_NumberOfPoints );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadNumberOfCellsFromDataFile()
{
  ITK_UINT32 numberOfCellsInDataFile;
  this->ReadInteger32( this->m_InputDataFile, numberOfCellsInDataFile );

  if( numberOfCellsInDataFile != this->m_NumberOfCells )
    {
    itkExceptionMacro("Data file has " << numberOfCellsInDataFile 
      << " points while Geometry file has " << this->m_NumberOfCells );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadNumberOfValuesPerPointFromDataFile()
{
  ITK_UINT32 numberOfValuesPerPoint;
  this->ReadInteger32( this->m_InputDataFile, numberOfValuesPerPoint );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::PrepareOutputMesh()
{
  typename OutputMeshType::Pointer outputMesh = this->GetOutput();

  outputMesh->SetCellsAllocationMethod(
      OutputMeshType::CellsAllocatedDynamicallyCellByCell );

  outputMesh->GetPoints()->Reserve( this->m_NumberOfPoints );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadSurface()
{
  this->ReadPoints();
  this->ReadCells();
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadPoints()
{
  typename OutputMeshType::Pointer outputMesh = this->GetOutput();

  PointType point;
  for( unsigned int ip=0; ip < this->m_NumberOfPoints; ip++ )
    {
    this->ReadPoint( point );
    outputMesh->SetPoint( ip, point );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadCells()
{
  typename OutputMeshType::Pointer outputMesh = this->GetOutput();

  for( unsigned int ic=0; ic < this->m_NumberOfCells; ic++ )
    {
    TriangleCellType * triangleCell = new TriangleCellType;

    this->ReadCell( *triangleCell );

    CellAutoPointer cell;
    cell.TakeOwnership( triangleCell );
    outputMesh->SetCell( ic, cell );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadPointData()
{
  typename PointDataContainer::Pointer pointData = PointDataContainer::New();
  pointData->Reserve( this->m_NumberOfPoints );

  PointDataIterator  pointDataIterator = pointData->Begin();  

  PointDataIterator endPoint = pointData->End();

  float pixelValue;

  while( pointDataIterator != endPoint ) 
    {
    this->m_InputDataFile.read( (char *)(&pixelValue), sizeof( pixelValue ) );
    itk::ByteSwapper<float>::SwapFromSystemToBigEndian( &pixelValue );
    pointDataIterator.Value() = static_cast<PixelType>( pixelValue );
    ++pointDataIterator;
    }

  typename OutputMeshType::Pointer outputMesh = this->GetOutput();
  outputMesh->SetPointData( pointData );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadCell( TriangleCellType & triangleCell )
{
  const unsigned int numberOfCellPoints = 3; // Triangles

  ITK_UINT32 pointId;

  for( PointIdentifier k = 0; k < numberOfCellPoints; k++ )
    {
    this->ReadInteger32( this->m_InputGeometryFile, pointId );
    triangleCell.SetPointId( k, pointId );
    }
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadInteger32( std::ifstream & inputStream, ITK_UINT32 & valueToRead )
{
  inputStream.read( (char *)(&valueToRead), sizeof(valueToRead) );

  itk::ByteSwapper<ITK_UINT32>::SwapFromSystemToBigEndian( &valueToRead );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadFloat( std::ifstream & inputStream, float & valueToRead  )
{
  inputStream.read( (char *)(&valueToRead), sizeof(valueToRead) );

  itk::ByteSwapper<float>::SwapFromSystemToBigEndian( &valueToRead );
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::ReadPoint( PointType & point )
{
  float x;
  float y;
  float z;
  this->ReadFloat( this->m_InputGeometryFile, x );
  this->ReadFloat( this->m_InputGeometryFile, y );
  this->ReadFloat( this->m_InputGeometryFile, z );
  
  point[0] = x;
  point[1] = y;
  point[2] = z;
}


template<class TOutputMesh>
void
FreeSurferBinarySurfaceReader<TOutputMesh>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "FileName: " << this->m_FileName << std::endl;
  os << indent << "FileType: " << this->m_FileType << std::endl;
  os << indent << "Comment : " << this->m_Comment << std::endl;
  os << indent << "Number of Points : " << this->m_NumberOfPoints << std::endl;
  os << indent << "Number of Cells  : " << this->m_NumberOfCells << std::endl;

  os << indent << "DataFileName: " << this->m_DataFileName << std::endl;
  os << indent << "DataFileType: " << this->m_DataFileType << std::endl;
}

} //end of namespace itk


#endif
