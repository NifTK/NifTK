#include <iostream>
#include <cstdlib>

#include "vnl_sd_matrix_tools.h"

#include <vnl/vnl_random.h>

#if MATLAB_FOUND
#  include "engine.h"
#endif

int main(int, char* [] )
{
  bool test_passed = true;
  const double eps = 1e-8;

  // Generate a random affine homogeneous matrix
  unsigned int dim = 4;

  vnl_matrix<double> m(dim,dim,0);

  // Generate a random linear submatrix
  // not to far from Id
  vnl_random rng;
  for (unsigned int i=0; i<dim-1; ++i)
    {
    for (unsigned int j=0; j<dim-1; ++j)
      {
      m(i,j) = 0.1 * rng.normal();
      }
    m(i,i) += 1.0;
    }

  // Bottom right element is 1 for affine homogeneous
  // matrices
  m(dim-1,dim-1) = 1.0;

  // Generate a random translation
  for (unsigned int i=0; i<dim-1; ++i)
    {
    m(i,dim-1) = 10 * rng.normal();
    }

  std::cout<<"m:"<<std::endl<<m<<std::endl;

  const vnl_matrix<double> log_m = sdtools::GetLogarithm(m);
  std::cout<<"log_m:"<<std::endl<<log_m<<std::endl;
  const vnl_matrix<double> exp_log_m = sdtools::GetExponential(log_m);
  std::cout<<"exp_log_m:"<<std::endl<<exp_log_m<<std::endl;

  const double el_diff = (exp_log_m - m).frobenius_norm();
  std::cout<<"el_diff = "<<el_diff<<std::endl;
  if ( el_diff > eps )
    {
    std::cerr << "exp(log(m)) is too far from m" << std::endl;
    test_passed = false;
    }

  const vnl_matrix<double> exp_m = sdtools::GetExponential(m);
  std::cout<<"exp_m:"<<std::endl<<exp_m<<std::endl;
  const vnl_matrix<double> log_exp_m = sdtools::GetLogarithm(exp_m);
  std::cout<<"log_exp_m:"<<std::endl<<log_exp_m<<std::endl;

  const double le_diff = (exp_log_m - m).frobenius_norm();
  std::cout<<"le_diff = "<<le_diff<<std::endl;
  if ( el_diff > eps )
    {
    std::cerr << "log(exp(m)) is too far from m" << std::endl;
    test_passed = false;
    }


#if MATLAB_FOUND
  Engine *ep = engOpen("\0");
  if ( ep )
    {
    mxArray *M = mxCreateDoubleMatrix(dim, dim, mxREAL);

    // Copy information into M
    double * Mptr = mxGetPr(M);
    for (unsigned int i=0; i<dim; ++i)
      {
      for (unsigned int j=0; j<dim; ++j)
        {
        Mptr[i+j*dim] = m(i,j);
        }
      }
   
    // Place the variable M into the MATLAB workspace
    engPutVariable(ep, "M", M);
   
    // evaluate the matrix functions of M
    engEvalString(ep, "logM = logm(M); expM = expm(M);");
   
    // Get the variables back from the MATLAB workspace
    mxArray *logM = engGetVariable(ep,"logM");
    mxArray *expM = engGetVariable(ep,"expM");
   
    // Copy information from logM and expM
    const double * logMptr = mxGetPr(logM);
    const double * expMptr = mxGetPr(expM);
   
    vnl_matrix<double> log_m_matlab(dim,dim);
    vnl_matrix<double> exp_m_matlab(dim,dim);
   
    for (unsigned int i=0; i<dim; ++i)
      {
      for (unsigned int j=0; j<dim; ++j)
        {
        log_m_matlab(i,j) = logMptr[i+j*dim];
        exp_m_matlab(i,j) = expMptr[i+j*dim];
        }
      }
   
    std::cout<<"log_m_matlab:"<<std::endl<<log_m_matlab<<std::endl;
    std::cout<<"exp_m_matlab:"<<std::endl<<exp_m_matlab<<std::endl;
   
    const double llm_diff = (log_m - log_m_matlab).frobenius_norm();
    std::cout<<"llm_diff = "<<llm_diff<<std::endl;
    if ( llm_diff > eps )
      {
      std::cerr << "log(m) is too far from logm(m) from MATLAB" << std::endl;
      test_passed = false;
      }
   
    const double eem_diff = (exp_m - exp_m_matlab).frobenius_norm();
    std::cout<<"eem_diff = "<<eem_diff<<std::endl;
    if ( eem_diff > eps )
      {
      std::cerr << "exp(m) is too far from expm(m) from MATLAB" << std::endl;
      test_passed = false;
      }
    }
  else
    {
    std::cerr << std::endl;
    std::cerr << "WARNING: Can't start MATLAB engine." << std::endl;
    std::cerr << "This will however not report an error in ctest." << std::endl;
    std::cerr << "Please check that MATLAB's licence manager is running." << std::endl;
    std::cerr << std::endl;
    }

#endif // MATLAB_FOUND

  if ( test_passed )
    {
    std::cout << "Test passed." << std::endl;
    return EXIT_SUCCESS;
    }
  else
    {
    std::cout << "Test failed." << std::endl;
    return EXIT_FAILURE;
    }
}
