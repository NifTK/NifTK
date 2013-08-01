/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkJonesThicknessFilter_h
#define itkJonesThicknessFilter_h
#include <itkImageToImageFilter.h>
#include <itkVector.h>
#include "itkCheckForThreeLevelsFilter.h"
#include "itkLaplacianSolverImageFilter.h"
#include "itkScalarImageToNormalizedGradientVectorImageFilter.h"
#include "itkIntegrateStreamlinesFilter.h"

namespace itk
{

/**
 * \class JonesThicknessFilter
 * \brief Composite filter to implement Jones et. al. Human Brain Mapping 2005 11:12-32(2000).
 * 
 * \sa CheckForThreeLevelsFilter
 * \sa LaplacianSolverImageFilter
 * \sa LaplacianSolverImageFilter
 * \sa GaussianSmoothVectorFieldFilter
 * \sa IntegrateStreamlinesFilter
 */
template <typename TImageType, typename TScalarType, unsigned int NDimensions>
class ITK_EXPORT JonesThicknessFilter : public ImageToImageFilter<TImageType, TImageType>
{
  public:
    /** Standard class typedefs. */
    typedef JonesThicknessFilter                                Self;
    typedef ImageToImageFilter<TImageType, TImageType>          Superclass;
    typedef SmartPointer<Self>                                  Pointer;
    typedef SmartPointer<const Self>                            ConstPointer;
    typedef CheckForThreeLevelsFilter<TImageType>                                     CheckFilterType;
    typedef typename CheckFilterType::Pointer                                         CheckFilterPointer;
    typedef LaplacianSolverImageFilter<TImageType, TScalarType>                       LaplaceFilterType;
    typedef typename LaplaceFilterType::Pointer                                       LaplaceFilterPointer;
    typedef ScalarImageToNormalizedGradientVectorImageFilter<TImageType, TScalarType> NormalsFilterType;
    typedef typename NormalsFilterType::Pointer                                       NormalsFilterPointer;
    typedef IntegrateStreamlinesFilter<TImageType, TScalarType, NDimensions>          IntegrateFilterType;
    typedef typename IntegrateFilterType::Pointer                                     IntegrateFilterPointer;
    typedef typename NormalsFilterType::OutputImageType                               VectorNormalImageType;
    typedef typename VectorNormalImageType::Pointer                                   VectorNormalImagePointer;
    typedef typename LaplaceFilterType::OutputImageType                               LaplacianImageType;
    
    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    
    /** Runtime information support. */
    itkTypeMacro(JonesThicknessFilter, ImageToImageFilter);

    /** Print internal ivars */
    void PrintSelf(std::ostream& os, Indent indent) const;

    /** Set/Get the low voltage, or low potential mentioned in paper. Defaults to 0. */
    itkSetMacro(LowVoltage, TScalarType);
    itkGetMacro(LowVoltage, TScalarType);

    /** Set/Get the high voltage, or high potential mentioned in paper. Defaults to 10000. */
    itkSetMacro(HighVoltage, TScalarType);
    itkGetMacro(HighVoltage, TScalarType);

    /** Set/Get the epsilon convergence tolerance. Defaults to 0.00001. */
    itkSetMacro(LaplaceEpsionRatio, TScalarType);
    itkGetMacro(LaplaceEpsionRatio, TScalarType);

    /** Set/Get the maximum number of iterations. Defaults to 200. */
    itkSetMacro(LaplaceMaxIterations, unsigned long int);
    itkGetMacro(LaplaceMaxIterations, unsigned long int);

    /** Set/Get the white matter label. Defaults to 1. */
    itkSetMacro(WhiteMatterLabel, short int);
    itkGetMacro(WhiteMatterLabel, short int);

    /** Set/Get the grey matter label. Defaults to 2. */
    itkSetMacro(GreyMatterLabel, short int);
    itkGetMacro(GreyMatterLabel, short int);

    /** Set/Get the CSF matter label. Defaults to 2. */
    itkSetMacro(CSFMatterLabel, short int);
    itkGetMacro(CSFMatterLabel, short int);

    /** Set/Get the minimum step size. Defaults to 0.1. */
    itkSetMacro(MinimumStepSize, TScalarType);
    itkGetMacro(MinimumStepSize, TScalarType);

    /** Set/Get the maximum length for ray casting. Stops the iterations going on for ever. Defaults to 10 */
    itkSetMacro(MaximumLength, TScalarType);
    itkGetMacro(MaximumLength, TScalarType);

    /** Set/Get the sigma for Gaussian smoothing of vector normals. Defaults to 0. ie. OFF */
    itkSetMacro(Sigma, TScalarType);
    itkGetMacro(Sigma, TScalarType);

    /** 
     * Set/Get UseLabels flag. If true, we iterate towards the average of the segmentation labels.
     * If false, we iterate towards the potential boundaries (0,10000). Defaults to false as per Jones paper.
     */
    itkSetMacro(UseLabels, bool);
    itkGetMacro(UseLabels, bool);

    /**
     * Set/Get the flag to determine if we are smoothing vector normals.
     */
    itkSetMacro(UseSmoothing, bool);
    itkGetMacro(UseSmoothing, bool);

    /** Set a pointer to an image of vector normals. If set, we use this instead. */
    void SetVectorNormalsOverrideImage(VectorNormalImageType* v);
    VectorNormalImageType* GetVectorNormalsOverrideImage();
    VectorNormalImageType* GetVectorNormalsFilterImage();
    LaplacianImageType* GetLaplacianFilterImage();
    
  protected:
    
    JonesThicknessFilter();
    virtual ~JonesThicknessFilter() {};

    // The main filter method. Note, single threaded.
    virtual void GenerateData();
    
  private:
    JonesThicknessFilter(const Self&); // purposely not implemented
    void operator=(const Self&); // purposely not implemented

    TScalarType m_LowVoltage;
    TScalarType m_HighVoltage;
    TScalarType m_LaplaceEpsionRatio;
    unsigned long int m_LaplaceMaxIterations;
    short int m_WhiteMatterLabel;
    short int m_GreyMatterLabel;
    short int m_CSFMatterLabel;
    TScalarType m_MinimumStepSize;
    TScalarType m_MaximumLength;
    TScalarType m_Sigma;
    bool m_DontUseGaussSiedel;
    bool m_UseLabels;
    bool m_UseSmoothing;
    
    CheckFilterPointer m_CheckFilter;
    LaplaceFilterPointer m_LaplaceFilter;
    NormalsFilterPointer m_NormalsFilter;
    IntegrateFilterPointer m_IntegrateFilter;
    VectorNormalImagePointer m_NormalsOverrideImage;
    
};

} // end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJonesThicknessFilter.txx"
#endif
  
#endif


