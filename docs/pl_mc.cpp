#include <Rcpp.h>
// [[Rcpp::depends(dqrng, BH, sitmo)]]
#include <pcg_random.hpp>
#include <dqrng_distribution.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <random>
#include <math.h>
#include <numeric>





//PRNG: pseudo-random number generator
std::random_device rd1;
pcg64 gen(rd1(),rd1());





// [[Rcpp::export]]
// Sample from Dirichlet distribution
Rcpp::NumericVector rdirichlet( Rcpp::NumericVector alpha ){
  
  Rcpp::NumericVector res(alpha.size()); 
  
  for ( int i=1; i<=alpha.size(); i++ ){
    std::gamma_distribution<double> gamma_dist(alpha[i-1],1.0);
    res(i-1) = gamma_dist(gen);
  }
  
  res = res / sum(res);
  
  return (res);
  
}






struct L_mca : public RcppParallel::Worker { 
  
  RcppParallel::RVector<double> alpha; 
  RcppParallel::RMatrix<double> Phi;      
  RcppParallel::RMatrix<double> LDA_data; 
  double IS;
  double S1;  
  uint64_t seed;
  
  // Accumulated value: likelihood
  double L; 
  
  // Standard constructor: 
  L_mca( Rcpp::NumericVector alpha, 
         Rcpp::NumericMatrix Phi, 
         Rcpp::NumericMatrix LDA_data,
         double IS,
         double S1, 
         uint64_t seed ) : alpha(alpha), Phi(Phi), LDA_data(LDA_data), IS(IS), S1(S1), seed(seed), L(0) {}
  
  // Splitting constructor: 
  L_mca( const L_mca& l_mca, RcppParallel::Split ):  
    alpha(l_mca.alpha), Phi(l_mca.Phi), LDA_data(l_mca.LDA_data), IS(l_mca.IS), S1(l_mca.S1), seed(l_mca.seed), L(0) {}
  
  // Operator to accumulate the value: 
  void operator() ( std::size_t begin, std::size_t end ) { 
    
    // Random seed for a worker: 
    pcg64 rng(seed, end);
    
    for ( std::size_t d=begin; d<end; d++ ) {     // loop through individuals
      
      double pL_d = 0;   

      for ( std::size_t is=0; is<IS; is++ ){    // loop through samples of topic weights (theta)
        
        double pL_is = 1;
        
        // Sample topic weights (theta) 
        std::vector<double> theta_sample(alpha.size());       

        for ( std::size_t k=0; k<alpha.size(); k++ ){
          std::gamma_distribution<double> dist_gamma( alpha[k],1.0 );
          theta_sample[k] = dist_gamma(rng);
        }

        double sum = std::accumulate(theta_sample.begin(), theta_sample.end(), 0.0);
        for ( std::size_t k=0; k<alpha.size(); k++ ){ theta_sample[k] = theta_sample[k]/sum; }
        

        for ( std::size_t l=0; l<LDA_data.ncol(); l++ ){       // loop through disease codes
          
          std::vector<double> L_word(alpha.size());
          
          for ( std::size_t k=0; k<alpha.size(); k++ ){ 
             if ( LDA_data(d,l) == 1 ) { 
               L_word[k] = Phi(k,l); 
             } else { 
               L_word[k] = 1-Phi(k,l); 
             } 
          }
          
          pL_is = pL_is * ( std::inner_product(L_word.begin(), L_word.end(), theta_sample.begin(), 0.0) ); 
          
        }   // end of loop through disease codes (for one sample of theta)
        
        pL_d = pL_d + pL_is; 
        
      }   // end of loop through all samples of theta
      
      pL_d = pL_d / IS;
      L = L + log10(pL_d); 
      
    }  // end of loop through individuals

  }  // end of operator 
  
  // Operator to do the work: 
  void join ( const L_mca& rhs ) { L += rhs.L; }
  
};



// [[Rcpp::export]]
double pred_L_mca ( Rcpp::NumericVector alpha, 
                    Rcpp::NumericMatrix Phi, 
                    Rcpp::NumericMatrix LDA_data,
                    double IS, double S1 ) { 
  
  std::uniform_real_distribution<double> unif_dist(0,999999999);
  uint64_t seed = unif_dist(gen); 
  
  // Create the worker:
  L_mca l_mca( alpha, Phi, LDA_data, IS, S1, seed );
  
  // Call the worker:
  RcppParallel::parallelReduce(0, LDA_data.nrow(), l_mca );

  return (l_mca.L);  
  
} 
