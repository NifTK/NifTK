/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFreeSurferBinarySurfaceReader.h,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFreeSurferBinarySurfaceReader_h
#define __itkFreeSurferBinarySurfaceReader_h

#include "itkMesh.h"
#include "itkMeshSource.h"
#include "itkTriangleCell.h"
#include "itkMapContainer.h"
#include "itkIntTypes.h"

#include <fstream>

namespace itk
{

/** \class FreeSurferBinarySurfaceReader
 * \brief
 * Reads a surface from the FreeSurface binary file format.
 *
 * The description of the FreeSurface file format can be found at
 * http://wideman-one.com/gw/brain/fs/surfacefileformats.htm
 *
 */
template <class TOutputMesh>
class FreeSurferBinarySurfaceReader : public MeshSource<TOutputMesh>
{
public:
  /** Standard "Self" typedef. */
  typedef FreeSurferBinarySurfaceReader         Self;
  typedef MeshSource<TOutputMesh>               Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FreeSurferBinarySurfaceReader, MeshSource);

  /** Hold on to the type information specified by the template parameters. */
  typedef TOutputMesh                          OutputMeshType;
  typedef typename OutputMeshType::MeshTraits  MeshTraits;
  typedef typename OutputMeshType::PointType   PointType;
  typedef typename MeshTraits::PixelType       PixelType;

  /** Some convenient typedefs. */
  typedef typename OutputMeshType::Pointer         OutputMeshPointer;
  typedef typename OutputMeshType::CellTraits      CellTraits;
  typedef typename OutputMeshType::CellIdentifier  CellIdentifier;
  typedef typename OutputMeshType::CellType        CellType;
  typedef typename OutputMeshType::CellAutoPointer CellAutoPointer;
  typedef typename OutputMeshType::PointIdentifier PointIdentifier;
  typedef typename CellTraits::PointIdIterator     PointIdIterator;

  typedef typename OutputMeshType::PointsContainerPointer PointsContainerPointer;
  
  typedef typename OutputMeshType::PointsContainer PointsContainer;

  /** Types related to point data */
  typedef typename OutputMeshType::PointDataContainer  PointDataContainer;
  typedef typename PointDataContainer::Iterator        PointDataIterator;

  /** Define the triangular cell types which form the surface  */
  typedef TriangleCell<CellType>      TriangleCellType;

  typedef typename TriangleCellType::SelfAutoPointer TriangleCellAutoPointer;

  typedef std::pair<unsigned long,unsigned long>     IndexPairType;
  typedef MapContainer<IndexPairType, unsigned long> PointMapType;
  typedef typename PointType::VectorType             VectorType;

  /** Set/Get the name of the file to be read. */
  itkSetStringMacro(FileName);
  itkGetStringMacro(FileName);

  /** Set/Get the name of the file containing point data */
  itkSetStringMacro(DataFileName);
  itkGetStringMacro(DataFileName);

protected:
  FreeSurferBinarySurfaceReader();
  ~FreeSurferBinarySurfaceReader() {}
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Reads the file */
  void GenerateData();

  /** Filename to read */
  std::string      m_FileName;
  std::string      m_DataFileName;

private:
  FreeSurferBinarySurfaceReader(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  void OpenGeometryFile();
  void CloseGeometryFile();
  void PrepareOutputMesh();
  void ReadGeometryHeader();
  void ReadSurface();
  void ReadFileTypeFromGeometryFile();
  void ReadCommentFromGeometryFile();
  void ReadNumberOfPointsFromGeometryFile();
  void ReadNumberOfCellsFromGeometryFile();
  void ReadPoints();
  void ReadCells();

  bool DataFileIsAvailable() const;
  void OpenDataFile();
  void CloseDataFile();
  void ReadDataHeader();
  void ReadFileTypeFromDataFile();
  void ReadPointData();
  void ReadNumberOfPointsFromDataFile();
  void ReadNumberOfCellsFromDataFile();
  void ReadNumberOfValuesPerPointFromDataFile();

  void ReadInteger32( std::ifstream & inputStream, ITK_UINT32 & valueToRead );
  void ReadFloat( std::ifstream & inputStream, float & valueToRead );
  void ReadPoint( PointType & point );
  void ReadCell( TriangleCellType & triangleCell );

  std::ifstream  m_InputGeometryFile;
  std::ifstream  m_InputDataFile;

  ITK_UINT32     m_FileType;
  ITK_UINT32     m_DataFileType;

  ITK_UINT32     m_NumberOfPoints;
  ITK_UINT32     m_NumberOfCells;
  std::string    m_Comment;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFreeSurferBinarySurfaceReader.txx"
#endif

#endif //_itkFreeSurferBinarySurfaceReader_h
