/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkLagrangianInitializedRelaxStreamlinesFilter_h
#define __itkLagrangianInitializedRelaxStreamlinesFilter_h

#include "itkRelaxStreamlinesFilter.h"
#include "itkContinuousIndex.h"

namespace itk {
/** 
 * \class LagrangianInitializedRelaxStreamlinesFilter
 * \brief Implements section 2.3.2 in Bourgeat et. al. ISBI 2008.
 * 
 * This class is implemented as a sub-class of RelaxStreamlinesFilter
 * to make it clear that apart from initializing the boundaries, there
 * is no difference in terms of PDE solving.
 * 
 * Usage: See the implementation of niftkCTEBourgeat2008.cxx
 * 
 * You can additionally set the interpolator, but it defaults to Linear
 * which should be fine. The StepSize threshold is the minimum step size
 * in the dichotomy search, which defaults to 0.001 as mentioned in the
 * paper. This class assumes that at the boundary you are modelling two
 * tissue types, so that when the paper says the stopping point for
 * Lagrangian Initialization (i.e. ray casting) is "where the PV maps
 * are equal", then in this implementation, we simply search for the 
 * threshold 0.5. So GreyMatterPercentage defaults to 0.5, but you 
 * can tweak it if you want to.
 * 
 * \sa RelaxStreamlinesFilter
 */
template < class TImageType, typename TScalarType, unsigned int NDimensions> 
class ITK_EXPORT LagrangianInitializedRelaxStreamlinesFilter :
  public RelaxStreamlinesFilter< TImageType, TScalarType, NDimensions>
{
public:

  /** Standard "Self" typedef. */
  typedef LagrangianInitializedRelaxStreamlinesFilter                   Self;
  typedef RelaxStreamlinesFilter<TImageType, TScalarType, NDimensions>  Superclass;
  typedef SmartPointer<Self>                                            Pointer;
  typedef SmartPointer<const Self>                                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LagrangianInitializedRelaxStreamlinesFilter, RelaxStreamlinesFilter);

  /** Standard typedefs. */
  typedef typename Superclass::InputVectorImagePixelType      InputVectorImagePixelType;
  typedef typename Superclass::InputVectorImageType           InputVectorImageType;
  typedef typename Superclass::InputVectorImagePointer        InputVectorImagePointer;
  typedef typename Superclass::InputVectorImageConstPointer   InputVectorImageConstPointer;
  typedef typename Superclass::InputVectorImageIndexType      InputVectorImageIndexType; 
  typedef typename Superclass::InputScalarImagePixelType      InputScalarImagePixelType;
  typedef typename Superclass::InputScalarImageType           InputScalarImageType;
  typedef typename Superclass::InputScalarImageSpacingType    InputScalarImageSpacingType;
  typedef typename Superclass::InputScalarImagePointType      InputScalarImagePointType;
  typedef typename Superclass::InputScalarImagePointer        InputScalarImagePointer;
  typedef typename Superclass::InputScalarImageIndexType      InputScalarImageIndexType;
  typedef typename Superclass::InputScalarImageConstPointer   InputScalarImageConstPointer;
  typedef typename Superclass::InputScalarImageRegionType     InputScalarImageRegionType;
  typedef typename Superclass::OutputImageType                OutputImageType;
  typedef typename Superclass::OutputImagePixelType           OutputImagePixelType;
  typedef typename Superclass::OutputImagePointer             OutputImagePointer;
  typedef typename Superclass::OutputImageConstPointer        OutputImageConstPointer;
  typedef typename Superclass::OutputImageIndexType           OutputImageIndexType;
  typedef typename Superclass::OutputImageSpacingType         OutputImageSpacingType;
  typedef typename Superclass::OutputImageRegionType          OutputImageRegionType;
  typedef typename Superclass::OutputImageSizeType            OutputImageSizeType;
  typedef typename Superclass::VectorInterpolatorType         VectorInterpolatorType;
  typedef typename Superclass::VectorInterpolatorPointer      VectorInterpolatorPointer;
  typedef typename Superclass::VectorInterpolatorPointType    VectorInterpolatorPointType;
  typedef typename Superclass::ScalarInterpolatorType         ScalarInterpolatorType;
  typedef typename Superclass::ScalarInterpolatorPointer      ScalarInterpolatorPointer;
  typedef typename Superclass::ScalarInterpolatorPointType    ScalarInterpolatorPointType;
  typedef ContinuousIndex<TScalarType, NDimensions>           ContinuousIndexType;
  
  /** Sets the segmented/label image, at input 2. */
  void SetSegmentedImage(const InputScalarImageType *image) {this->SetNthInput(2, const_cast<InputScalarImageType *>(image)); }

  /** Sets the pv map, at input 3. */
  void SetGMPVMap(const InputScalarImageType *image) {this->SetNthInput(3, const_cast<InputScalarImageType *>(image)); }

  /** Set/Get the interpolator for the GM pv map. Defaults to Linear. */
  itkSetObjectMacro(GreyMatterPVInterpolator, ScalarInterpolatorType);
  itkGetObjectMacro(GreyMatterPVInterpolator, ScalarInterpolatorType);
  
  /** 
   * Set lower threshold of dichotomy search. 
   * Default 0.001 as per paper. 
   */
  itkSetMacro(StepSizeThreshold, double);
  itkGetMacro(StepSizeThreshold, double);

  /** 
   * Grey matter percentage to search for. 
   * Should always be 50%, so default is 0.5, 
   * but you can set it if you feel like it. 
   */
  itkSetMacro(GreyMatterPercentage, double);
  itkGetMacro(GreyMatterPercentage, double);

  /** 
   * Maximum distance to search for Lagrangian initialization. 
   * In a perfect world, your PV boundary would be 1 voxel thick,
   * and this would be somewhat unnecessary. In the real world,
   * you may be less trustworthy of how wide your PV boundary
   * should be. So, we provide a maximum threshold. If the 
   * distance found is greater than this value, the distance
   * value is capped at this value. Default 2mm.
   */
  itkGetMacro(MaximumSearchDistance, double);
  void SetMaximumSearchDistance(double d)
    {
      m_MaximumSearchDistance = d;
      m_DefaultMaximumSearchDistance = false;
    }

  /** Set/Get the flag to determine if we let this filter establish its own max search distance. Defaults to true. */
  itkSetMacro(DefaultMaximumSearchDistance, bool);
  itkGetMacro(DefaultMaximumSearchDistance, bool);
  
protected:
  LagrangianInitializedRelaxStreamlinesFilter();
  ~LagrangianInitializedRelaxStreamlinesFilter() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /**
   * We set the whole of L0 and L1 image to zero,
   * and extract the set of GM voxels.
   */ 
  virtual void InitializeBoundaries(
    std::vector<InputScalarImageIndexType>& completeListOfGreyMatterPixels,
    InputScalarImageType* scalarImage,
    InputVectorImageType* vectorImage,
    OutputImageType* L0Image,
    OutputImageType* L1Image,
    std::vector<InputScalarImageIndexType>& L0greyList,
    std::vector<InputScalarImageIndexType>& L1greyList
    );

  /** Actually does the search for a single voxel. */
  OutputImagePixelType LagrangianInitialisation(
    ContinuousIndexType& index, 
    double& direction,
    double& defaultStepSize, 
    double& minStepSize,
    InputVectorImageType* vectorImage,
    InputScalarImageType* greyMatterPVMap);

  /** 
   * In section 2.3.2 of Bourgeat's paper, they describe a section, where
   * the PV value is redistributed, thereby editing the GM volume.
   */
  void UpdateGMPVMap(
      std::vector<InputScalarImageIndexType>& listOfGreyMatterPixels,
      InputScalarImageType* segmentedImage,
      InputVectorImageType* vectorImage,
      InputScalarImageType* gmPVMapImage, 
      InputScalarImageType* editedPVMap);
  
  /** Calculates a simple estimate for the max step size of dichotomy search. */
  double GetMaxStepSize(InputScalarImageType* image);
  
private:
  
  /**
   * Prohibited copy and assingment. 
   */
  LagrangianInitializedRelaxStreamlinesFilter(const Self&); 
  void operator=(const Self&); 

  /** The minimum step size in dichotomy search, as a percentage of voxel size. */
  double m_StepSizeThreshold;
  
  /** Grey matter percentage that we search to. */
  double m_GreyMatterPercentage;
  
  /** Maximum distance to search. */
  double m_MaximumSearchDistance;
  
  /** Let the filter select the maximum search distance, which will be 1/2 voxel diagonal length. */
  bool m_DefaultMaximumSearchDistance;
  
  /** Interpolator for PV map. */
  ScalarInterpolatorPointer m_GreyMatterPVInterpolator;
  
  /** Interpolator for normals. */
  VectorInterpolatorPointer m_NormalsInterpolator;
};

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLagrangianInitializedRelaxStreamlinesFilter.txx"
#endif

#endif
