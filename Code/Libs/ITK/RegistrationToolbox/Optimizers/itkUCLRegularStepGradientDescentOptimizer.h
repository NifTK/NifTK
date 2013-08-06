/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkUCLRegularStepGradientDescentOptimizer_h
#define itkUCLRegularStepGradientDescentOptimizer_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkSingleValuedNonLinearOptimizer.h>

namespace itk
{
  
/** 
 * \class UCLRegularStepGradientDescentBaseOptimizer
 * \brief Implement a Regular Step Size Gradient Descent optimizer.
 *
 * Unfortunately its a cut and paste of the ITK original, with one main changes.
 * I wanted to keep track of the best position found so far, and at the end of registration 
 * return that regardless.  There were some cases where if you always followed
 * the gradient, while it was computing the gradient (say using finite differences)
 * it would find a position with a lower cost minima, but because
 * the gradient was uphill, it would ignore it.
 * 
 * (I had to cut and paste as the original class didnt have virtual 
 * methods in the right place)
 * 
 * \ingroup Numerics Optimizers
 */
class NIFTKITK_WINEXPORT ITK_EXPORT UCLRegularStepGradientDescentOptimizer :
  public SingleValuedNonLinearOptimizer
{
public:
  /** Standard "Self" typedef. */
  typedef UCLRegularStepGradientDescentOptimizer   Self;
  typedef SingleValuedNonLinearOptimizer           Superclass;
  typedef SmartPointer<Self>                       Pointer;
  typedef SmartPointer<const Self>                 ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( UCLRegularStepGradientDescentOptimizer, 
                SingleValuedNonLinearOptimizer );
  

  /** Codes of stopping conditions. */
  typedef enum {
    GradientMagnitudeTolerance = 1,
    StepTooSmall = 2,
    ImageNotAvailable = 3,
    CostFunctionError = 4,
    MaximumNumberOfIterations = 5,
    Unknown = 6
  } StopConditionType;

  /** Specify whether to minimize or maximize the cost function. */
  itkSetMacro( Maximize, bool );
  itkGetConstReferenceMacro( Maximize, bool );
  itkBooleanMacro( Maximize );
  bool GetMinimize( ) const
    { return !m_Maximize; }
  void SetMinimize(bool v)
    { this->SetMaximize(!v); }
  void    MinimizeOn(void) 
    { SetMaximize( false ); }
  void    MinimizeOff(void) 
    { SetMaximize( true ); }

  /** Start optimization. */
  void    StartOptimization( void );

  /** Resume previously stopped optimization with current parameters.
   * \sa StopOptimization */
  void    ResumeOptimization( void );

  /** Stop optimization.
   * \sa ResumeOptimization */
  void    StopOptimization( void );

  /** Set/Get parameters to control the optimization process. */
  itkSetMacro( MaximumStepLength, double );
  itkSetMacro( MinimumStepLength, double );
  itkSetMacro( RelaxationFactor, double );
  itkSetMacro( NumberOfIterations, unsigned long );
  itkSetMacro( GradientMagnitudeTolerance, double );
  itkGetConstReferenceMacro( CurrentStepLength, double);
  itkGetConstReferenceMacro( MaximumStepLength, double );
  itkGetConstReferenceMacro( MinimumStepLength, double );
  itkGetConstReferenceMacro( RelaxationFactor, double );
  itkGetConstReferenceMacro( NumberOfIterations, unsigned long );
  itkGetConstReferenceMacro( GradientMagnitudeTolerance, double );
  itkGetConstMacro( CurrentIteration, unsigned int );
  itkGetConstReferenceMacro( StopCondition, StopConditionType );
  itkGetConstReferenceMacro( Value, MeasureType );
  itkGetConstReferenceMacro( Gradient, DerivativeType );
  
protected:
  UCLRegularStepGradientDescentOptimizer();
  virtual ~UCLRegularStepGradientDescentOptimizer() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Advance one step following the gradient direction
   * This method verifies if a change in direction is required
   * and if a reduction in steplength is required. */
  virtual void AdvanceOneStep( void );

  /** Advance one step along the corrected gradient taking into
   * account the steplength represented by factor.
   * This method is invoked by AdvanceOneStep. It is expected
   * to be overrided by optimization methods in non-vector spaces
   * \sa AdvanceOneStep */
  virtual void StepAlongGradient(double, const DerivativeType&);

private:  
  UCLRegularStepGradientDescentOptimizer(const Self&); //purposely not implemented
  void operator=(const Self&);//purposely not implemented

protected:
  DerivativeType                m_Gradient; 
  DerivativeType                m_PreviousGradient; 

  bool                          m_Stop;
  bool                          m_Maximize;
  MeasureType                   m_Value;
  MeasureType                   m_BestSoFarValue;
  ParametersType                m_BestSoFarParameters;
  double                        m_GradientMagnitudeTolerance;
  double                        m_MaximumStepLength;
  double                        m_MinimumStepLength;
  double                        m_CurrentStepLength;
  double                        m_RelaxationFactor;
  StopConditionType             m_StopCondition;
  unsigned long                 m_NumberOfIterations;
  unsigned long                 m_CurrentIteration;


};

} // end namespace itk

#endif
