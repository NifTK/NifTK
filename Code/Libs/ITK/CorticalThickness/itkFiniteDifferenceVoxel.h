/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkFiniteDifferenceVoxel_h
#define itkFiniteDifferenceVoxel_h
#include <itkContinuousIndex.h>
#include <itkPoint.h>

namespace itk
{
/**
 * \class FiniteDifferenceVoxel
 * \brief Simple data type to hold a voxel value, and indexes that can
 * be used to refer to other voxels.
 * 
 * These FiniteDifferenceVoxels should be stored in a list, so the index
 * is assumed to be pointing to it's neighbour in 2D/3D space.
 * 
 */
template <int Dimension, int VectorSize, typename PixelType, typename PrecisionType>
class ITK_EXPORT FiniteDifferenceVoxel {

  public:
    
    typedef ContinuousIndex<PrecisionType, Dimension> ContinuousIndexType;
    typedef Point<PrecisionType, Dimension> PointType;
    
    ~FiniteDifferenceVoxel() {
      delete [] plusIndex;
      delete [] minusIndex;
      delete [] values;
      delete [] needsSolving;
    }
    
    FiniteDifferenceVoxel() {
      values = new PixelType[VectorSize];
      needsSolving = new bool[VectorSize];
      for (unsigned int i = 0; i < VectorSize; i++)
        {
          values[i] = 0;
          needsSolving[i] = true;
        }
      voxelArrayIndex = 0;
      voxelIndex.Fill(0);
      voxelPointInMillimetres.Fill(0);
      isBoundary = false;
      isNextToCSF = false;
      isNextToWM = false;
      plusIndex = new long int[Dimension];
      minusIndex = new long int[Dimension];
      for (unsigned int i = 0; i < Dimension; i++)
        {
          plusIndex[i] = 0;
          minusIndex[i] = 0;
        }
      //std::cout << "FiniteDifferenceVoxel():Default constructor, this=" << this << std::endl;
    }

    FiniteDifferenceVoxel(const FiniteDifferenceVoxel& another) {
      values = new PixelType[VectorSize];
      needsSolving = new bool[VectorSize];
      for (unsigned int i = 0; i < VectorSize; i++)
        {
          values[i] = another.GetValue(i);
          needsSolving[i] = another.GetNeedsSolving(i);
        }      
      voxelArrayIndex = another.GetVoxelArrayIndex();
      voxelIndex = another.GetVoxelIndex();      
      voxelPointInMillimetres = another.GetVoxelPointInMillimetres();
      isBoundary = another.GetBoundary();
      isNextToCSF = another.GetIsNextToCSF();
      isNextToWM = another.GetIsNextToWM();      
      plusIndex = new long int[Dimension];
      minusIndex = new long int[Dimension];
      for (unsigned int i = 0; i < Dimension; i++)
        {
          plusIndex[i] = another.GetPlus(i);
          minusIndex[i] = another.GetMinus(i);
        }
      //std::cout << "FiniteDifferenceVoxel():Copy constructor, this=" << this << ", another=" << &another << std::endl;
    }

    void operator=(const FiniteDifferenceVoxel& another) {
      for (unsigned int i = 0; i < VectorSize; i++)
        {
          values[i] = another.GetValue(i);
          needsSolving[i] = another.GetNeedsSolving(i);
        }            
      voxelArrayIndex = another.GetVoxelArrayIndex();
      voxelIndex = another.GetVoxelIndex();
      voxelPointInMillimetres = another.GetVoxelPointInMillimetres();
      isBoundary = another.GetBoundary();
      isNextToCSF = another.GetIsNextToCSF();
      isNextToWM = another.GetIsNextToWM();
      for (unsigned int i = 0; i < Dimension; i++)
        {
          plusIndex[i] = another.GetPlus(i);
          minusIndex[i] = another.GetMinus(i);
        }
      //std::cout << "FiniteDifferenceVoxel():Operator=, this=" << this << ", another=" << &another << std::endl;
    }
    
    void SetPlus(const int& dim, const long int index) {
      plusIndex[dim] = index;
    }
    
    long int GetPlus (const int& dim) const {
      return plusIndex[dim];
    }

    void SetMinus(const int& dim, const long int index) {
      minusIndex[dim] = index;
    }
    
    long int GetMinus (const int& dim) const {
      return minusIndex[dim];
    }

    void SetValue(int i, PixelType input) {
      values[i] = input;
    }
    
    PixelType GetValue (int i) const {
      return values[i];
    }

    void SetNeedsSolving(int i, bool b) {
      needsSolving[i] = b;
    }
    
    bool GetNeedsSolving(int i) const {
      return needsSolving[i];
    }
    
    void SetBoundary(bool input) {
      isBoundary = input;
    }
    
    bool GetBoundary () const {
      return isBoundary;
    }

    void SetIsNextToCSF(bool input) {
      isNextToCSF = input;
    }
    
    bool GetIsNextToCSF () const {
      return isNextToCSF;
    }

    void SetIsNextToWM(bool input) {
      isNextToWM = input;
    }
    
    bool GetIsNextToWM () const {
      return isNextToWM;
    }

    void SetVoxelIndex(const ContinuousIndexType& index) {
      voxelIndex = index;
    }
    
    ContinuousIndexType GetVoxelIndex () const {
      return voxelIndex;
    }

    void SetVoxelArrayIndex(unsigned long int i) {
      voxelArrayIndex = i;
    }
    
    unsigned long int GetVoxelArrayIndex() const {
      return voxelArrayIndex;
    }
    
    void SetVoxelPointInMillimetres(PointType& p) { 
      voxelPointInMillimetres = p;
    }
    
    PointType GetVoxelPointInMillimetres() const {
      return voxelPointInMillimetres;
    }
    
    int GetSizeofObject() {
      return sizeof(bool)*VectorSize + sizeof(bool*) + sizeof(PixelType)*VectorSize + sizeof(PixelType*) \
      + sizeof(unsigned long int) + sizeof(ContinuousIndexType) + sizeof(ContinuousIndexType) \
      + sizeof(bool)*3 + sizeof(long int *)*2 + sizeof(long int)*2*Dimension \
      ;
    }
    
  private:
    bool *needsSolving;
    PixelType* values;
    unsigned long int voxelArrayIndex;
    ContinuousIndexType voxelIndex;
    PointType voxelPointInMillimetres;
    bool isBoundary;
    bool isNextToCSF;
    bool isNextToWM;
    long int *plusIndex;
    long int *minusIndex;
};

} // end namespace
#endif
