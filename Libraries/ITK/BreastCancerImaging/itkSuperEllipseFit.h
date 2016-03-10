/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#ifndef __itkSuperEllipseFit_h
#define __itkSuperEllipseFit_h

#include <itkMultipleValuedCostFunction.h>
#include <itkLevenbergMarquardtOptimizer.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_math.h>

#include <itkArray.h>
#include <itkArray2D.h>

#include <deque>

namespace itk {
  
typedef vnl_matrix<double> MatrixType;
typedef vnl_vector<double> VectorType;


#if 0

class ITK_EXPORT Index2D : public itk::Index< 2 >
{
  Index2D() {
    this->m_Index[0] = 0.;
    this->m_Index[1] = 0.;
  }

  Index2D operator=(const Index2D &newIndex) {
    
    for ( unsigned int i = 0; i < Dimension; i++ )
    { 
      m_Index[i] = newIndex.m_Index[i];
    }
    return *this;
  }
}

#endif


// --------------------------------------------------------------------------
// SuperEllipseFitMetric
// --------------------------------------------------------------------------

class ITK_EXPORT SuperEllipseFitMetric : public MultipleValuedCostFunction
{
public:

  typedef SuperEllipseFitMetric        Self;

  typedef MultipleValuedCostFunction   Superclass;
  typedef SmartPointer<Self>           Pointer;
  typedef SmartPointer<const Self>     ConstPointer;

  itkNewMacro( Self );

  enum { NumberOfParameters =  5 };

  typedef Superclass::ParametersType              ParametersType;
  typedef Superclass::DerivativeType              DerivativeType;
  typedef Superclass::MeasureType                 MeasureType;

  typedef itk::Index< 2 > Index2D;

  typedef std::deque< Index2D > DataType;

  SuperEllipseFitMetric( void )  
  {
  }

  void SetData( DataType &inputData )
  {
    m_NumberOfDataPoints = inputData.size();

    m_Measure.SetSize( m_NumberOfDataPoints );
    m_Derivative.SetSize( NumberOfParameters, m_NumberOfDataPoints );

    m_Data = inputData;
  }

  MeasureType GetValue( const ParametersType &parameters ) const
  {
    unsigned int i;

    double A = parameters[0];
    double B = parameters[1];
    double C = parameters[2];
    double D = parameters[3];
    double E = parameters[4];

    double x;
    double y;

    unsigned int iPoint = 0;

    std::deque< Index2D >::const_iterator itData = m_Data.begin();

    while ( itData != m_Data.end() ) 
    {
      x = (*itData)[0];
      y = (*itData)[1];

      m_Measure[ iPoint ] = A*x*x + B*y*y + C*x + D*y + E; 
      
      ++itData;
      iPoint++;
    }

    return m_Measure; 
 }

  void GetDerivative( const ParametersType &parameters,
                            DerivativeType &derivative ) const
  {
    unsigned int iPoint = 0;

    double x;
    double y;

    std::deque< Index2D >::const_iterator itData = m_Data.begin();

    while ( itData != m_Data.end() ) 
    {
      x = (*itData)[0];
      y = (*itData)[1];

      m_Derivative[0][iPoint] =  x*x;
      m_Derivative[1][iPoint] =  y*y;
      m_Derivative[2][iPoint] =  x;
      m_Derivative[3][iPoint] =  y;
      m_Derivative[4][iPoint] =  1.0;      
    }

    derivative = m_Derivative;

    ++itData;
    iPoint++;
  }

  unsigned int GetNumberOfParameters(void) const
  {
    return NumberOfParameters;
  }

  unsigned int GetNumberOfValues(void) const
  {
    return m_NumberOfDataPoints;
  }


protected:

  DataType m_Data;


private:

  unsigned int m_NumberOfDataPoints;

  mutable MeasureType       m_Measure;
  mutable DerivativeType    m_Derivative;

};



// --------------------------------------------------------------------------
// CommandIterationUpdateSuperEllipseFit
// --------------------------------------------------------------------------

class CommandIterationUpdateSuperEllipseFit : public Command 
{
  public:
  typedef  CommandIterationUpdateSuperEllipseFit   Self;
  typedef  Command                               Superclass;
  typedef SmartPointer<Self>                     Pointer;
  itkNewMacro( Self );
  protected:
  CommandIterationUpdateSuperEllipseFit() 
  {
    m_IterationNumber=0;
  }
  public:
  typedef LevenbergMarquardtOptimizer OptimizerType;
  typedef const OptimizerType *OptimizerPointer;

  void Execute(Object *caller, const EventObject & event)
  {
    Execute( (const Object *)caller, event);
  }

  void Execute(const Object * object, const EventObject & event)
  {
    std::cout << "Observer::Execute() " << std::endl;
    OptimizerPointer optimizer = 
      dynamic_cast< OptimizerPointer >( object );
    if( m_FunctionEvent.CheckEvent( &event ) )
    {
      std::cout << m_IterationNumber++ << "   ";
      std::cout << optimizer->GetCachedValue() << "   ";
      std::cout << optimizer->GetCachedCurrentPosition() << std::endl;
    }
    else if( m_GradientEvent.CheckEvent( &event ) )
    {
      std::cout << "Gradient " << optimizer->GetCachedDerivative() << "   ";
    }

  }
  private:
  unsigned long m_IterationNumber;

  FunctionEvaluationIterationEvent m_FunctionEvent;
  GradientEvaluationIterationEvent m_GradientEvent;
};


// --------------------------------------------------------------------------
// SuperEllipseFit()
// --------------------------------------------------------------------------

int SuperEllipseFit( SuperEllipseFitMetric::DataType &data,
                     bool useGradient = true, 
                     double fTolerance = 1e-2, 
                     double gTolerance = 1e-2, 
                     double xTolerance = 1e-5, 
                     double epsilonFunction = 1e-9, 
                     int maxIterations = 200 )
{
  std::cout << "Levenberg Marquardt optimisation \n \n"; 

  typedef  LevenbergMarquardtOptimizer  OptimizerType;

  typedef  OptimizerType::InternalOptimizerType  vnlOptimizerType;

  // Declaration of a itkOptimizer
  OptimizerType::Pointer  optimizer = OptimizerType::New();

  // Declaration of the CostFunction adaptor
  SuperEllipseFitMetric::Pointer costFunction = SuperEllipseFitMetric::New();

  costFunction->SetData( data );

  typedef SuperEllipseFitMetric::ParametersType ParametersType;
  ParametersType parameters( SuperEllipseFitMetric::NumberOfParameters );

  parameters.Fill( 0.0 );
  costFunction->GetValue(parameters);

  std::cout << "Number of Values = " << costFunction->GetNumberOfValues() << "\n";

  try 
  {
    optimizer->SetCostFunction( costFunction.GetPointer() );
  }
  catch( ExceptionObject & e )
  {
    std::cout << "Exception thrown ! " << std::endl;
    std::cout << "An error ocurred during Optimization" << std::endl;
    std::cout << e << std::endl;
    return EXIT_FAILURE;
  }

  // this following call is equivalent to invoke: costFunction->SetUseGradient( useGradient );
  optimizer->GetUseCostFunctionGradient();
  optimizer->UseCostFunctionGradientOn();
  optimizer->UseCostFunctionGradientOff();
  optimizer->SetUseCostFunctionGradient( useGradient );


  vnlOptimizerType * vnlOptimizer = optimizer->GetOptimizer();

  vnlOptimizer->set_f_tolerance( fTolerance );
  vnlOptimizer->set_g_tolerance( gTolerance );
  vnlOptimizer->set_x_tolerance( xTolerance ); 
  vnlOptimizer->set_epsilon_function( epsilonFunction );
  vnlOptimizer->set_max_function_evals( maxIterations );

  // We start not so far from the solution 
  typedef SuperEllipseFitMetric::ParametersType ParametersType;
  ParametersType  initialValue( SuperEllipseFitMetric::NumberOfParameters );

  initialValue[0] = 10;
  initialValue[1] = 20;
  initialValue[2] = 100;
  initialValue[3] = 200;
  initialValue[4] = 1;

  OptimizerType::ParametersType currentValue(SuperEllipseFitMetric::NumberOfParameters);

  currentValue = initialValue;

  optimizer->SetInitialPosition( currentValue );

  CommandIterationUpdateSuperEllipseFit::Pointer observer = 
    CommandIterationUpdateSuperEllipseFit::New();
  optimizer->AddObserver( IterationEvent(), observer );
  optimizer->AddObserver( FunctionEvaluationIterationEvent(), observer );

  try 
  {
    optimizer->StartOptimization();
  }
  catch( ExceptionObject & e )
  {
    std::cerr << "Exception thrown ! " << std::endl;
    std::cerr << "An error ocurred during Optimization" << std::endl;
    std::cerr << "Location    = " << e.GetLocation()    << std::endl;
    std::cerr << "Description = " << e.GetDescription() << std::endl;
    return EXIT_FAILURE;
  }


  // Error codes taken from vxl/vnl/vnl_nonlinear_minimizer.h
  std::cout << "End condition   = ";
  switch( vnlOptimizer->get_failure_code() )
  {
  case vnl_nonlinear_minimizer::ERROR_FAILURE: 
    std::cout << " Error Failure"; break;
  case vnl_nonlinear_minimizer::ERROR_DODGY_INPUT: 
    std::cout << " Error Dogy Input"; break;
  case  vnl_nonlinear_minimizer::CONVERGED_FTOL: 
    std::cout << " Converged F  Tolerance"; break;
  case  vnl_nonlinear_minimizer::CONVERGED_XTOL: 
    std::cout << " Converged X  Tolerance"; break;
  case  vnl_nonlinear_minimizer::CONVERGED_XFTOL:
    std::cout << " Converged XF Tolerance"; break;
  case  vnl_nonlinear_minimizer::CONVERGED_GTOL: 
    std::cout << " Converged G  Tolerance"; break;
  case  vnl_nonlinear_minimizer::FAILED_TOO_MANY_ITERATIONS:
    std::cout << " Too many iterations   "; break;
  case  vnl_nonlinear_minimizer::FAILED_FTOL_TOO_SMALL:
    std::cout << " Failed F Tolerance too small "; break;
  case  vnl_nonlinear_minimizer::FAILED_XTOL_TOO_SMALL:
    std::cout << " Failed X Tolerance too small "; break;
  case  vnl_nonlinear_minimizer::FAILED_GTOL_TOO_SMALL:
    std::cout << " Failed G Tolerance too small "; break;
  case  vnl_nonlinear_minimizer::FAILED_USER_REQUEST:
    std::cout << " Failed user request "; break;
  }
  std::cout << std::endl;
  std::cout << "Number of iters = " << vnlOptimizer->get_num_iterations() << std::endl;
  std::cout << "Number of evals = " << vnlOptimizer->get_num_evaluations() << std::endl;    
  std::cout << std::endl;


  OptimizerType::ParametersType finalPosition;
  finalPosition = optimizer->GetCurrentPosition();

  std::cout << "Solution        = (";
  std::cout << finalPosition[0] << "," ;
  std::cout << finalPosition[1] << "," ;
  std::cout << finalPosition[2] << "," ;
  std::cout << finalPosition[3] << "," ;
  std::cout << finalPosition[4] << ")" << std::endl;  

  return EXIT_SUCCESS;
};

} // end namespace itk

#endif


