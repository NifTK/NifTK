#ifndef __itkNCCRegistrationFunction2_txx
#define __itkNCCRegistrationFunction2_txx

#include "itkNCCRegistrationFunction2.h"
#include "itkExceptionObject.h"
#include "vnl/vnl_math.h"

namespace itk {

/**
 * Default constructor
 */
template <class TFixedImage, class TMovingImage, class TDeformationField>
NCCRegistrationFunction2<TFixedImage,TMovingImage,TDeformationField>
::NCCRegistrationFunction2()
{

  RadiusType r;
  unsigned int j;
  for( j = 0; j < ImageDimension; j++ )
    {
    r[j] = 1;
    }
  this->SetRadius(r);
  m_MetricTotal=0.0;

  m_TimeStep = 1.0;
  m_DenominatorThreshold = 1e-9;
  this->SetMovingImage(NULL);
  this->SetFixedImage(NULL);
  m_FixedImageSpacing.Fill( 1.0 );
  m_FixedImageGradientCalculator = GradientCalculatorType::New();

  typename DefaultInterpolatorType::Pointer interp =
    DefaultInterpolatorType::New();

  m_MovingImageInterpolator = static_cast<InterpolatorType*>(
    interp.GetPointer() );

  m_SubtractMean = false;
}


/*
 * Standard "PrintSelf" method.
 */
template <class TFixedImage, class TMovingImage, class TDeformationField>
void
NCCRegistrationFunction2<TFixedImage,TMovingImage,TDeformationField>
::PrintSelf(std::ostream& os, Indent indent) const
{

  Superclass::PrintSelf(os, indent);

  os << indent << "MovingImageIterpolator: ";
  os << m_MovingImageInterpolator.GetPointer() << std::endl;
  os << indent << "FixedImageGradientCalculator: ";
  os << m_FixedImageGradientCalculator.GetPointer() << std::endl;
  os << indent << "DenominatorThreshold: ";
  os << m_DenominatorThreshold << std::endl;
  os << indent << "SubtractMean: ";
  os << m_SubtractMean << std::endl;
}


/*
 * Set the function state values before each iteration
 */
template <class TFixedImage, class TMovingImage, class TDeformationField>
void
NCCRegistrationFunction2<TFixedImage,TMovingImage,TDeformationField>
::InitializeIteration()
{
  if( !this->m_MovingImage || !this->m_FixedImage || !m_MovingImageInterpolator )
    {
    itkExceptionMacro( << "MovingImage, FixedImage and/or Interpolator not set" );
    }

  // cache fixed image information
  m_FixedImageSpacing    = this->m_FixedImage->GetSpacing();

  // setup gradient calculator
  m_FixedImageGradientCalculator->SetInputImage( this->m_FixedImage );

  // setup moving image interpolator
  m_MovingImageInterpolator->SetInputImage( this->m_MovingImage );

  //std::cout << " total metric " << m_MetricTotal << " field size " <<
  //  this->GetDeformationField()->GetLargestPossibleRegion().GetSize()<< " image size " <<
  //  this->m_FixedImage->GetLargestPossibleRegion().GetSize() << std::endl;
  m_MetricTotal=0.0;
}


/*
 * Compute update at a non boundary neighbourhood
 */
template <class TFixedImage, class TMovingImage, class TDeformationField>
typename NCCRegistrationFunction2<TFixedImage,TMovingImage,TDeformationField>
::PixelType
NCCRegistrationFunction2<TFixedImage,TMovingImage,TDeformationField>
::ComputeUpdate(const NeighborhoodType &it, void * itkNotUsed(globalData),
                const FloatOffsetType& itkNotUsed(offset))
{
  const IndexType oindex = it.GetIndex();
  const typename FixedImageType::SizeType hradius=it.GetRadius();
  FixedImageType* img =const_cast<FixedImageType *>(this->m_FixedImage.GetPointer());
  const typename FixedImageType::SizeType imagesize=img->GetLargestPossibleRegion().GetSize();

  NeighborhoodIterator<FixedImageType>
    hoodIt( hradius , img, img->GetRequestedRegion());
  hoodIt.SetLocation(oindex);

  double sff=0.0;
  double smm=0.0;
  double sfm=0.0;
  double sf=0.0;
  double sm=0.0;

  double derivativeF[ImageDimension];
  double derivativeM[ImageDimension];
  for (unsigned int j=0; j<ImageDimension;j++)
    {
    derivativeF[j]=0;
    derivativeM[j]=0;
    }

  // We can store vectors of interesting values without worrying about
  // memory issues since the neighborhood is typically small
  const unsigned int hoodlen=hoodIt.Size();
  std::vector<bool> inimages(hoodlen,true);
  std::vector<CovariantVectorType> fixedGradients(hoodlen);
 
  unsigned int numberOfPixelsCounted = 0;
  for(unsigned int indct=0; indct<hoodlen-1; indct++)
    {
    const IndexType & index = hoodIt.GetIndex(indct);
    for (unsigned int dd=0; dd<ImageDimension; dd++)
      {
      if ( index[dd] < 0 ||
           index[dd] > static_cast<typename IndexType::IndexValueType>(imagesize[dd]-1) )
        {
        inimages[indct]=false;
        }
      }
    if (inimages[indct])
      {
      // Get moving image related information
      typedef typename TDeformationField::PixelType  DeformationPixelType;
      const DeformationPixelType & vec = this->GetDeformationField()->GetPixel(index);
      PointType mappedPoint;
      this->GetFixedImage()->TransformIndexToPhysicalPoint(index, mappedPoint);
      for(unsigned int j = 0; j < ImageDimension; j++ )
        {
        mappedPoint[j] += vec[j];
        }
      double movingValue=0.0;
      if( m_MovingImageInterpolator->IsInsideBuffer( mappedPoint ) )
        {
        movingValue = m_MovingImageInterpolator->Evaluate( mappedPoint );
        ++numberOfPixelsCounted;
        }
      else
        {
        inimages[indct] = false;
        continue;
        }

      // Get fixed image related information
      // Note: no need to check the index is within
      // fixed image buffer. This is done by the external filter.
      const double fixedValue = (double) this->m_FixedImage->GetPixel( index );
      fixedGradients[indct] = m_FixedImageGradientCalculator->EvaluateAtIndex( index );
      
      
      // Compute sums
      sff += fixedValue*fixedValue;
      smm += movingValue*movingValue;
      sfm += fixedValue*movingValue;
      if ( this->m_SubtractMean )
        {
        sf += fixedValue;
        sm += movingValue;
        }

      for(unsigned int dim=0; dim<ImageDimension; dim++)
        {
        derivativeF[dim] += fixedValue  * fixedGradients[indct][dim];
        derivativeM[dim] += movingValue * fixedGradients[indct][dim];
        }
      }
    }

  

  if ( this->m_SubtractMean )
    {
    const double mf = sf / numberOfPixelsCounted;
    const double mm = sm / numberOfPixelsCounted;
    
    sff -= ( sf * mf );
    smm -= ( sm * mm );
    sfm -= ( sf * mm );

    // Update contributions to derivatives
    for(unsigned int indct=0; indct<hoodlen-1; indct++)
      {
      if (inimages[indct])
        {
        for(unsigned int dim=0; dim<ImageDimension; dim++)
          {
          if ( this->m_SubtractMean && numberOfPixelsCounted > 0 )
            {
            derivativeF[dim] -= fixedGradients[indct][dim] * mf;
            derivativeM[dim] -= fixedGradients[indct][dim] * mm;
            }
          }
        }
      }
    }
  
  PixelType update;
  update.Fill(0.0);
  double updatenorm=0.0;
  const double denom = vcl_sqrt(sff * smm );
  if( denom >= m_DenominatorThreshold )
    {
    const double factor = 1.0 / denom;
    for(unsigned int i=0; i<ImageDimension; i++)
      {
      update[i] = factor * ( derivativeF[i] - (sfm/smm)*derivativeM[i]);
      updatenorm += (update[i]*update[i]);
      }
    updatenorm=vcl_sqrt(updatenorm);
    m_MetricTotal += sfm*factor;
    this->m_Energy += sfm*factor;
    }
  else
    {
    update.Fill(0.0);
    updatenorm=1.0;
    }

  if (this->GetNormalizeGradient() && updatenorm != 0.0 )
    {
    update /= (updatenorm);
    }
  return update * this->m_GradientStep;
}

} // end namespace itk

#endif
