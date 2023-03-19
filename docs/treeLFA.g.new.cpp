#include <Rcpp.h>
// [[Rcpp::depends(dqrng, BH, sitmo)]]
#include <pcg_random.hpp>
#include <dqrng_distribution.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>

#include <random>
#include <math.h>

#include <boost/math/special_functions/digamma.hpp>

#include <iostream>
#include <fstream>





//PRNG: pseudo random number generator
std::random_device rd1;
pcg64 gen(rd1(),rd1());










// Sample from categorical distribution: 
// [[Rcpp::export]]
int rcat( Rcpp::NumericVector p ){
  
  int len = p.length();
  Rcpp::NumericVector pn( p.length() ); 
  int C=0; 
  double P;       
  
  double u = R::runif(0,1);      
  
  pn = p / sum(p); 
  
  P = pn[0];
  for ( int i=1; i<=len; i++ ){
    if ( u <= P ) { 
      C=i; 
      break;
    }
    else { P = P + pn[i]; }
  } 
  
  return (C);
  
}





// Sample from Dirichlet distribution
// [[Rcpp::export]]
Rcpp::NumericVector rdirichlet( Rcpp::NumericVector alpha ){
  
  int len = alpha.size();
  Rcpp::NumericVector res(len); 
  
  for ( int i=1; i<=len; i++ ){
    res(i-1) = R::rgamma( alpha[i-1], 1.0 );
  }
  
  res = res / sum(res);
  
  return (res);
  
}





// Summary of topic assignments for individuals:
struct Z_count_parallel : public RcppParallel::Worker { 
  
  RcppParallel::RMatrix<double> Z; 
  RcppParallel::RMatrix<double> Z_count;
  int K;
  int S1; 
  uint64_t seed;
  
  
  Z_count_parallel( Rcpp::NumericMatrix Z, 
                    Rcpp::NumericMatrix Z_count, 
                    int K, int S1, uint64_t seed ) : Z(Z), Z_count(Z_count), K(K), S1(S1), seed(seed) {}
  
  
  void operator() ( std::size_t begin, std::size_t end ) { 
    
    pcg64 rng(seed, end);
    
    for( std::size_t d=begin; d<end; d++ ) {           
      
      for ( int l=0; l<Z.ncol(); l++ ){ 
        Z_count( d, (Z(d,l)-1) ) = Z_count( d,(Z(d,l)-1) ) + 1; 
      }
      
    }   // end of one document in the operator 
    
  }    // end of operator 
  
};   // end of worker



Rcpp::NumericMatrix gibbs_Z_count_parallel( Rcpp::NumericMatrix Z, int K, int S1 ) { 
  
  Rcpp::NumericMatrix Z_count( Z.nrow(), K ); 
  
  std::uniform_real_distribution<double> unif_dist(0,999999999);
  uint64_t seed = unif_dist(gen);
  
  Z_count_parallel z_count_parallel( Z, Z_count, K, S1, seed );
  
  RcppParallel::parallelFor(0, Z.nrow(), z_count_parallel );
  
  return(Z_count);
  
}






// Upate topic assignments: Z
struct Z4_parallel : public RcppParallel::Worker { 
  
  int S1; 
  RcppParallel::RVector<double> alpha;
  RcppParallel::RMatrix<double> Phi; 
  RcppParallel::RMatrix<double> Phi2; 
  RcppParallel::RMatrix<double> LDA_data; 
  RcppParallel::RMatrix<double> Z; 
  RcppParallel::RMatrix<double> Z_sum_m;
  RcppParallel::RMatrix<double> term11;
  RcppParallel::RMatrix<double> term12;
  uint64_t seed;
  
  Z4_parallel( int S1, 
               Rcpp::NumericVector alpha, 
               Rcpp::NumericMatrix Phi,
               Rcpp::NumericMatrix Phi2,
               Rcpp::NumericMatrix LDA_data,
               Rcpp::NumericMatrix Z, 
               Rcpp::NumericMatrix Z_sum_m,
               Rcpp::NumericMatrix term11,
               Rcpp::NumericMatrix term12,
               uint64_t seed
  ) : S1(S1), alpha(alpha), Phi(Phi), Phi2(Phi2), LDA_data(LDA_data), Z(Z), 
  Z_sum_m(Z_sum_m), term11(term11), term12(term12), seed(seed) {}
  
  void operator() ( std::size_t begin, std::size_t end ) {
    
    pcg64 rng(seed, end);   // random generator for the worker
    
    for( std::size_t d=begin; d<end; d++ ) {     // loop through documents
      
      for ( int l=0; l<Z.ncol(); l++ ){      // loop through words
        
        Z_sum_m( d, ( Z(d,l) - 1 ) ) = Z_sum_m( d, ( Z(d,l) - 1 ) ) - 1;
        
        std::vector<double> p( alpha.size() );
        
        double U; 
        double all; 
        
        if ( LDA_data(d,l) == 1 ) {    // when data is 1 
          
          for ( int h=0; h<alpha.size(); h++ ) { 
            if ( Z_sum_m(d,h) != 0 ) {
              p[h] = term11(h,l) + Z_sum_m(d,h) * Phi(h,l);
            } else { 
              p[h] = term11(h,l);
            }
          }   
          
        } else {    // when data is 0
          
          for ( int h=0; h<alpha.size(); h++ ) { 
            if ( Z_sum_m(d,h) != 0 ) {
              p[h] = term12(h,l) + Z_sum_m(d,h) * Phi2(h,l);
            } else { 
              p[h] = term12(h,l); 
            }
          }   
          
        }  
        
        double p_sum=0.0; 
        p_sum = std::accumulate(p.begin(), p.end(), 0.0);
        
        // Sample from unif dist: 
        std::uniform_real_distribution<double> uniform_dist(0.0, p_sum);
        U = uniform_dist(rng);
        
        all=0.0; 
        for ( int h=0; h<alpha.size(); h++ ) {      // loop through topics 
          all = all + p[h];
          if ( U <= all ) { 
            Z(d,l) = (h+1);
            break;
          }
        }
        
        // Update topic assignment: 
        Z_sum_m( d,(Z(d,l)-1) ) = Z_sum_m( d,(Z(d,l)-1) ) + 1;
        
      }        // end of loop through words 
      
    }       // end of loop through documents 
    
  }      // end of the operator 
  
};  



Rcpp::List gibbs_Z_c_new( int S1,  
                          Rcpp::NumericVector alpha,
                          Rcpp::NumericMatrix Phi, 
                          Rcpp::NumericMatrix LDA_data, 
                          Rcpp::NumericMatrix Z,
                          Rcpp::NumericMatrix Z_sum_m ) { 
  
  Rcpp::List result_Z(2); 
  
  Rcpp::NumericMatrix term11(alpha.size(),Z.ncol()); 
  Rcpp::NumericMatrix term12(alpha.size(),Z.ncol()); 
  
  for ( int k=0; k<alpha.size(); k++ ){ 
    for ( int l=0; l<Z.ncol(); l++  ){
      term11(k,l) = alpha[k] * Phi(k,l);       // when the data is 1 
      term12(k,l) = alpha[k] - term11(k,l);     // when the data is 0 
    }
  }
  
  Rcpp::NumericMatrix Phi2(Phi.nrow(),Phi.ncol()); 
  
  for ( int k=0; k<alpha.size(); k++ ){ 
    for ( int l=0; l<Z.ncol(); l++  ){
      Phi2(k,l) = 1.0 - Phi(k,l);
    }
  }
  
  std::uniform_real_distribution<double> unif_dist(0,999999999);
  uint64_t seed = unif_dist(gen);
  
  Z4_parallel z4_parallel( S1, alpha, Phi, Phi2, LDA_data, Z, Z_sum_m,
                           term11, term12, seed );
  
  RcppParallel::parallelFor(0, Z.nrow(), z4_parallel );
  
  result_Z[0] = Z; 
  result_Z[1] = Z_sum_m; 
  
  return (result_Z);
  
}    






// Update Phi: 
struct Phi_parallel : public RcppParallel::Worker { 
  
  int S1; 
  double a00; 
  double a01; 
  double a10; 
  double a11; 
  RcppParallel::RVector<double> alpha;
  RcppParallel::RMatrix<double> Phi; 
  RcppParallel::RMatrix<double> Z; 
  RcppParallel::RMatrix<double> I; 
  RcppParallel::RMatrix<double> LDA_data;
  uint64_t seed;
  
  Phi_parallel( int S1, 
                double a00, 
                double a01, 
                double a10, 
                double a11,
                Rcpp::NumericVector alpha,
                Rcpp::NumericMatrix Phi,
                Rcpp::NumericMatrix Z,
                Rcpp::NumericMatrix I,
                Rcpp::NumericMatrix LDA_data,
                uint64_t seed ) : S1(S1), a00(a00), a01(a01), a10(a10), a11(a11),
                alpha(alpha), Phi(Phi), Z(Z), I(I), LDA_data(LDA_data), 
                seed(seed) {}
  
  void operator() ( std::size_t begin, std::size_t end ) {
    
    pcg64 rng(seed, seed*end);   
    
    for( int k=begin; k<end; k++ ) {  
      
      double n0;      // number of words in the corpus which has value 0 and are assigned with topic k
      double n1;
      double X; 
      double Y; 
      
      for ( int l=0; l<Z.ncol(); l++ ){      // loop through words
        
        n0 = 0.0;
        n1 = 0.0; 
        
        for ( int d=0; d<Z.nrow(); d++ ){
          if ( Z(d,l)==(k+1) ){
            if ( LDA_data(d,l)==1 ) { n1=n1+1.0; }
            else { n0=n0+1.0; }
          } 
        }
        
        if ( I(k,l+S1)==0 ) {                  
          // Phi(k-1,l-1) = R::rbeta(a00+n1,a01+n0);
          std::gamma_distribution<double> gamma_dist1( a00+n1, 1.0 );
          std::gamma_distribution<double> gamma_dist2( a01+n0, 1.0 );
          X = gamma_dist1(gen);
          Y = gamma_dist2(gen);
          Phi(k,l) = (X/(X+Y));
        } else { 
          // Phi(k-1,l-1) = R::rbeta(a10+n1,a11+n0); 
          std::gamma_distribution<double> gamma_dist1( a10+n1, 1.0 );
          std::gamma_distribution<double> gamma_dist2( a11+n0, 1.0 );
          X = gamma_dist1(gen);
          Y = gamma_dist2(gen);
          Phi(k,l) = (X/(X+Y));
        }
        
      }      // end of loop through words 
      
    }     // end of loop through topics
    
  }      // end of the operator 
  
};  



Rcpp::NumericMatrix gibbs_Phi_new( int K, int S, int S1, int D, 
                                   double a00, double a01, double a10, double a11, 
                                   double rho0, double rho1, 
                                   Rcpp::NumericVector alpha, 
                                   Rcpp::NumericMatrix Phi, 
                                   Rcpp::NumericMatrix Z, 
                                   Rcpp::NumericMatrix I, 
                                   Rcpp::NumericMatrix LDA_data ) { 
  
  std::uniform_real_distribution<double> unif_dist(0,999999999);
  uint64_t seed = unif_dist(gen);
  
  Phi_parallel phi_parallel( S1, a00, a01, a10, a11, 
                             alpha, Phi, Z, I, LDA_data, seed );
  
  RcppParallel::parallelFor(0, Phi.nrow(), phi_parallel );
  
  return (Phi);
  
}    





// Update I
Rcpp::NumericMatrix gibbs_I( int K, int S, int S1, int D, 
                             double a00, double a01, double a10, double a11, 
                             double rho0, double rho1, 
                             Rcpp::NumericVector alpha, 
                             Rcpp::NumericMatrix I, Rcpp::NumericMatrix Phi, 
                             Rcpp::NumericMatrix tree_str ) { 
  
  int pa_index;
  int pa_value;
  
  int child_index;
  
  double p0;
  double p1;
  double p0_n;
  double p1_n;
  
  for ( int k=1; k<=K; k++ ){                 
    
    // Indicator variables for internal nodes on the tree: with children nodes
    for ( int l=1; l<=S1; l++ ){                  
      
      //double phi = Phi(k-1,l-1);
      std::vector<int> child_value;
      
      pa_index = tree_str(l-1,1);                 
      if ( pa_index==0 ) { pa_value=0; } 
      else { pa_value = I(k-1, pa_index-1); }             
      
      for ( int n=1; n<=(S+S1); n++ ){
        if ( tree_str(n-1,1)==l ){
          child_index = tree_str(n-1,0); 
          child_value.push_back( I(k-1,child_index-1) );
        }
      }
      
      double prod_p0=1;
      double prod_p1=1;
      
      for ( int n=1; n<=child_value.size(); n++ ){
        prod_p0 = prod_p0 * pow(1-rho0,child_value[n-1]) * pow( rho0,1-child_value[n-1] );
        prod_p1 = prod_p1 * pow(1-rho1,child_value[n-1]) * pow( rho1,1-child_value[n-1] );
      }
      
      p0 = ( pow(rho1,pa_value) * pow(rho0,1-pa_value) ) * prod_p0;
      p1 = ( pow(1-rho1,pa_value) * pow(1-rho0,1-pa_value) ) * prod_p1;
      
      p0_n = p0 / (p0+p1);
      p1_n = p1 / (p0+p1);
      
      // I(k-1,l-1) = R::rbinom(1,p1_n);
      std::bernoulli_distribution bern_dist(p1_n);      
      I(k-1,l-1) = bern_dist(gen);
      
    }     // End of loop for words
    
    
    
    // Indicator variables for terminal nodes on the tree: with no children nodes
    for ( int l=(S1+1); l<=(S+S1); l++ ){                  
      
      double phi = Phi(k-1,l-1-S1);
      
      pa_index = tree_str(l-1,1);                 
      if ( pa_index==0 ) { pa_value=0; } 
      else { pa_value = I(k-1, pa_index-1); }             
      
      p0 = R::dbeta(phi,a00,a01,false) * ( pow(rho1,pa_value) * pow(rho0,1-pa_value) );
      p1 = R::dbeta(phi,a10,a11,false)  * ( pow(1-rho1,pa_value) * pow(1-rho0,1-pa_value) );
      
      p0_n = p0 / (p0+p1);
      p1_n = p1 / (p0+p1);
      
      // I(k-1,l-1) = R::rbinom(1,p1_n);
      std::bernoulli_distribution bern_dist(p1_n);      
      I(k-1,l-1) = bern_dist(gen);
      
    }     // End of loop for words
    
  }    // End of loop for topics
  
  return (I);
  
}





// Update rho: transition probs on the tree
Rcpp::NumericVector gibbs_rho( int K, int S, int S1, 
                               double b00, double b01, double b10, double b11,
                               Rcpp::NumericMatrix I, 
                               Rcpp::NumericMatrix tree_str ) {
  
  Rcpp::NumericVector rho(2); 
  
  // Number of transitions on the tree from node with value 0/1: 
  double N0=0; 
  double N1=0; 
  
  // Number of transitions on the tree (0 to 1; 1 to 0)
  double m01=0; 
  double m11=0;
  
  int index_pa;
  
  double X; 
  double Y;
  
  for ( int k=1; k<=K; k++ ) { 
    
    for ( int l=1; l<=(S+S1); l++ ){ 
      
      index_pa = tree_str(l-1,1);
      
      if ( index_pa == 0 ){ 
        N0 = N0 + 1; 
        if ( I(k-1,l-1)==1 ) { m01 = m01 + 1; }
      } else { 
        
        if ( I(k-1,index_pa-1) == 0 ) { 
          N0 = N0 + 1; 
          if ( I(k-1,l-1)==1 ) { 
            m01 = m01 + 1; }
        } else { 
          N1 = N1 + 1; 
          if ( I(k-1,l-1)==1 ) { m11 = m11 + 1; }
        }
        
      }    
      
    }   // end of loop for words
    
  }   // end of loop for topics 
  
  
  // Sample rho: 
  std::gamma_distribution<double> gamma_dist1( m01+b00, 1.0 );
  std::gamma_distribution<double> gamma_dist2( N0-m01+b01, 1.0 );
  X = gamma_dist1(gen);
  Y = gamma_dist2(gen);
  rho[0] = 1 - X/(X+Y);
  
  std::gamma_distribution<double> gamma_dist3( m11+b10, 1.0 );
  std::gamma_distribution<double> gamma_dist4( N1-m11+b11, 1.0 );
  X = gamma_dist3(gen);
  Y = gamma_dist4(gen);
  rho[1] = 1 - X/(X+Y);
  
  return( rho );
  
}





// Likelihood of data: 
struct L_parallel : public RcppParallel::Worker { 
  
  int S1; 
  RcppParallel::RMatrix<double> Z; 
  RcppParallel::RMatrix<double> Phi;      
  RcppParallel::RMatrix<double> LDA_data; 
  RcppParallel::RVector<double> alpha;
  
  double L; 
  
  L_parallel( int S1, 
              Rcpp::NumericMatrix Z, 
              Rcpp::NumericMatrix Phi, 
              Rcpp::NumericMatrix LDA_data,
              Rcpp::NumericVector alpha 
  ) : S1(S1), Z(Z), Phi(Phi), LDA_data(LDA_data), alpha(alpha), L(0) {}
  
  L_parallel( const L_parallel& l_parallel, RcppParallel::Split):  
    S1(l_parallel.S1), Z(l_parallel.Z), Phi(l_parallel.Phi), LDA_data(l_parallel.LDA_data), alpha(l_parallel.alpha), L(0) {}
  
  
  void operator() ( std::size_t begin, std::size_t end ) { 
    
    for( std::size_t d=begin; d<end; d++ ) {           
      
      for ( std::size_t l=0; l<Z.ncol(); l++ ) { 
        
        double t_ass = Z(d,l); 
        
        if ( LDA_data(d,l) == 1 ){ 
          L = L + log10( Phi(t_ass-1,l) ); 
        } else { 
          L = L + log10( 1 - Phi(t_ass-1,l) );
        }
        
      }
      
    }   
    
  }    
  
  void join ( const L_parallel& rhs ) { L += rhs.L; }
  
};



double gibbs_L_parallel( int S1, 
                         Rcpp::NumericMatrix Z, 
                         Rcpp::NumericMatrix Phi, 
                         Rcpp::NumericMatrix LDA_data,
                         Rcpp::NumericVector alpha ) { 
  
  L_parallel l_parallel( S1, Z, Phi, LDA_data, alpha );
  
  RcppParallel::parallelReduce(0, Z.nrow(), l_parallel );
  
  return (l_parallel.L);
  
}    





// Prior for Z: 
struct Z_prior : public RcppParallel::Worker {
  
  
  RcppParallel::RMatrix<double> Z; 
  RcppParallel::RMatrix<double> Z_sum_m;
  RcppParallel::RVector<double> alpha;
  int S;
  int S1; 
  
  double L;
  
  Z_prior( Rcpp::NumericMatrix Z,
           Rcpp::NumericMatrix Z_sum_m,
           Rcpp::NumericVector alpha, int S, int S1 ) : Z(Z), Z_sum_m(Z_sum_m), alpha(alpha), S(S), S1(S1),
           L(0) {}
  
  Z_prior( const Z_prior& z_prior, RcppParallel::Split):
    Z(z_prior.Z), Z_sum_m(z_prior.Z_sum_m), alpha(z_prior.alpha), S(z_prior.S), S1(z_prior.S1), L(0) {}
  
  void operator() ( std::size_t begin, std::size_t end ) {
    
    for( std::size_t k=begin; k<end; k++ ) {
      
      std::vector<double> lg( (S+S1+1) ); 
      
      for ( int l=0; l<(S+S1+1); l++ ) { 
        lg[l] = lgamma( alpha[k]+l )/log(10); 
      }
      
      for ( std::size_t d=0; d<Z.nrow(); d++ ){ 
        L = L + lg[Z_sum_m(d,k)];
      }
      
    }   
    
  }     
  
  void join ( const Z_prior& rhs ) { L += rhs.L; }
  
};



double prior_Z_parallel_new( int S, int S1, 
                             Rcpp::NumericMatrix Z,
                             Rcpp::NumericMatrix Z_sum_m,
                             Rcpp::NumericVector alpha ) {
  
  double prior_Z; 
  
  //int S2 = Z.ncol() - S1; 
  int D = Z.nrow();
  
  Z_prior z_prior( Z, Z_sum_m, alpha, S, S1 );
  
  RcppParallel::parallelReduce(0, alpha.size(), z_prior );
  
  prior_Z = z_prior.L; 
  
  prior_Z = prior_Z + D * ( lgamma(sum(alpha))/log(10) - lgamma(sum(alpha)+S)/log(10) ); 
  prior_Z = prior_Z - ( D * sum( lgamma(alpha)/log(10) ) );
  
  return (prior_Z);
  
}





double prior_I( Rcpp::NumericMatrix I,
                Rcpp::NumericMatrix tree_str, 
                Rcpp::NumericVector rho ) {
  
  double L_m = 0;     
  
  int pa_index;
  int pa_value;
  
  int K = I.nrow(); 
  int S_all = I.ncol(); 
  
  for ( int k=0; k<K; k++ ){ 
    
    for ( int l=0; l<S_all; l++ ){ 
      
      pa_index = tree_str(l,1); 
      
      if ( pa_index==0 ) { 
        pa_value=0; 
      } else { pa_value = I(k, pa_index-1); }
      
      if ( pa_value==0 ){ 
        L_m =  L_m + I(k,l) * log10( (1-rho[0]) ) + ( 1-I(k,l) ) * log10( rho[0] );   
      } else { L_m =  L_m + I(k,l) * log10( (1-rho[1]) ) + ( 1-I(k,l) ) * log10( rho[1] ); }
      
    }
    
  }
  
  return (L_m);
  
}





double prior_rho( double b00, double b01, double b10, double b11,
                  Rcpp::NumericVector rho, Rcpp::NumericMatrix I ){ 
  
  double L_m = 0; 
  double rho01 = 1 - rho[0]; 
  double rho11 = 1 - rho[1]; 
  
  L_m = L_m + log10(tgamma(b00+b01)) - log10(tgamma(b00)) - log10(tgamma(b01)) + 
    (b00-1)*log10(rho01) + (b01-1)*log10(1-rho01); 
  
  L_m = L_m + log10(tgamma(b10+b11)) - log10(tgamma(b10)) - log10(tgamma(b11)) + 
    (b10-1)*log10(rho11) + (b11-1)*log10(1-rho11); 
  
  return (L_m);
  
}





double prior_Phi( int S1,
                  double a00, double a01, double a10, double a11, 
                  Rcpp::NumericMatrix Phi, Rcpp::NumericMatrix I ) {
  
  double L_m = 0;      
  
  int K = Phi.nrow(); 
  int S = Phi.ncol(); 
  
  for ( int k=0; k<K; k++ ){ 
    
    for ( int l=0; l<S; l++ ){ 
      
      if ( I(k,l+S1)==0 ){ 
        L_m = L_m + log10( R::dbeta( Phi(k,l), a00, a01, false ) ); 
      } else { 
        L_m = L_m + log10( R::dbeta( Phi(k,l), a10, a11, false ) ); 
      }
      
    }
    
  }
  
  return (L_m);
  
}





// Optimize alpha
Rcpp::NumericVector opt_alpha( Rcpp::NumericVector alpha, 
                               Rcpp::List Z_sum,
                               int S, int S1, int D, int K, int opt_N ) {
  
  Rcpp::NumericMatrix Z_count( D,K ); 
  
  Rcpp::NumericVector alpha_new( alpha.length() );
  alpha_new = clone( alpha ); 
  
  Rcpp::NumericVector alpha_old( alpha.length() );
  
  int OC = 0; 
  
  do {
    
    alpha_old = clone(alpha_new); 
    
    double alpha_c = sum(alpha_old); 
    
    double denominator = ( boost::math::digamma( S+alpha_c ) - boost::math::digamma( alpha_c ) ) * D * opt_N; 
    
    for ( int k=1; k<=K; k++ ) {
      
      double numerator = 0.0; 
      
      for ( int t=1; t<=opt_N; t++ ) { 
        
        Z_count = Rcpp::as<Rcpp::NumericMatrix>(Z_sum[t-1]); 
        
        for ( int d=1; d<=D; d++ ) { 
          numerator = numerator + boost::math::digamma( Z_count(d-1,k-1) + alpha_old[k-1] ) - boost::math::digamma( alpha_old[k-1] ); 
        }
        
      } 
      
      alpha_new[k-1] = ( alpha_old[k-1] * numerator )  / ( denominator ); 
      
      if ( alpha_new[k-1]<=0.01 ) { alpha_new[k-1] = alpha_old[k-1]; }
      
    }     
    
    OC = OC + 1; 
    
  } while ( OC <= 3 ); 
  
  return (alpha_new); 
  
}  





// Gibbs sampler: E-step
Rcpp::List gibbs_parallel( int K, int S, int S1, int D,
                           double a00, double a01, double a10, double a11, 
                           double b00, double b01, double b10, double b11, 
                           Rcpp::NumericVector alpha, 
                           Rcpp::NumericVector rho, 
                           Rcpp::NumericMatrix Z, Rcpp::NumericMatrix Z_sum_m,
                           Rcpp::NumericMatrix I, Rcpp::NumericMatrix Phi, 
                           Rcpp::NumericMatrix LDA_data, 
                           Rcpp::NumericMatrix tree_str, int opt_N ) {
  
  Rcpp::List gibbs_result(6);
  Rcpp::List result_Z(2);
  
  Rcpp::List Z_sum(opt_N);
  
  for ( int j=1; j<=opt_N; j++ ) { 
    
    /*
    for ( int i=1; i<=burn_in; i++ ) {      
      
      rho = gibbs_rho( K, S, S1, 
                       b00, b01, b10, b11, 
                       I, tree_str );
      
      I = gibbs_I( K, S, S1, D, 
                   a00, a01, a10, a11, 
                   rho[0], rho[1], alpha, 
                   I, Phi, tree_str ); 
      
      Phi = gibbs_Phi_new( K, S, S1, D, 
                           a00, a01, a10, a11, 
                           rho[0], rho[1], alpha, 
                           Phi, Z, I, LDA_data ); 
      
      result_Z = gibbs_Z_c_new( S1, alpha, Phi, LDA_data, Z, Z_sum_m ); 
      Z = Rcpp::as<Rcpp::NumericMatrix>(result_Z[0]); 
      Z_sum_m = Rcpp::as<Rcpp::NumericMatrix>(result_Z[1]); 
      
    }
    */
    
    rho = gibbs_rho( K, S, S1, 
                     b00, b01, b10, b11, 
                     I, tree_str );
    
    I = gibbs_I( K, S, S1, D, 
                 a00, a01, a10, a11, 
                 rho[0], rho[1], alpha, 
                 I, Phi, tree_str ); 
    
    Phi = gibbs_Phi_new( K, S, S1, D, 
                         a00, a01, a10, a11, 
                         rho[0], rho[1], alpha, 
                         Phi, Z, I, LDA_data ); 
    
    result_Z = gibbs_Z_c_new( S1, alpha, Phi, LDA_data, Z, Z_sum_m ); 
    Z = Rcpp::as<Rcpp::NumericMatrix>(result_Z[0]); 
    Z_sum_m = Rcpp::as<Rcpp::NumericMatrix>(result_Z[1]); 
    
    Z_sum[j-1] = Z_sum_m; 
    
  }
  
  gibbs_result[0] = I;
  gibbs_result[1] = Phi;
  gibbs_result[2] = Z;
  gibbs_result[3] = rho;
  gibbs_result[4] = Z_sum_m;
  gibbs_result[5] = Z_sum; 
  
  return (gibbs_result);
  
}





// [[Rcpp::export]]
// Gibbs-EM training: 
Rcpp::List gibbs( int K, int S, int S1, int D, 
                     double a00, double a01, double a10, double a11, 
                     double b00, double b01, double b10, double b11,
                     Rcpp::NumericVector alpha, 
                     Rcpp::NumericVector rho, 
                     Rcpp::NumericMatrix Z, Rcpp::NumericMatrix I, Rcpp::NumericMatrix Phi, 
                     Rcpp::NumericMatrix LDA_data, 
                     Rcpp::NumericMatrix tree_str,
                     int opt_N, int burn_in, int cycle, int interval ) {
  
  // Final result:   
  Rcpp::List result_final(6);
  // Result for one gibbs sampling iteration:
  Rcpp::List result_gibbs(6);
  
  // Number of posterior samples to collect: 
  int N_samples = ( cycle - burn_in ) / interval;
  int j = 0;      // index for posterior samples of topics 
  
  Rcpp::List Phi_samples(N_samples);
  Rcpp::List I_samples(N_samples);
  Rcpp::List rho_samples(N_samples);
  Rcpp::List Z_sum_samples(N_samples);
  
  
  // Marginal likelihood: 
  Rcpp::NumericVector L_data( cycle );
  Rcpp::NumericVector Z_prior( cycle );
  Rcpp::NumericVector I_prior( cycle );
  Rcpp::NumericVector Phi_prior( cycle );
  Rcpp::NumericVector rho_prior( cycle );
  
  Rcpp::NumericVector L_all( cycle );
  
  
  // Summary of topic assignments for all disease variables for all individuals:
  Rcpp::List Z_sum_1( opt_N );
  Rcpp::NumericMatrix Z_sum_m( D,K );
  
  // Get the summary of topic assignments:
  Z_sum_m = gibbs_Z_count_parallel( Z, alpha.size(), S1 ); 
  
  

  for ( int c=1; c<=cycle; c++ ){ 
    
    std::cout << "Gibbs sampling iteration: " << c << std::endl;
    
    // Gibbs sampler: 
    result_gibbs = gibbs_parallel( K, S, S1, D, 
                                   a00, a01, a10, a11, 
                                   b00, b01, b10, b11, 
                                   alpha, 
                                   rho,
                                   Z, Z_sum_m, I, Phi, 
                                   LDA_data, 
                                   tree_str,
                                   opt_N );
    
    // Update hidden variables:  
    I = Rcpp::as<Rcpp::NumericMatrix>(result_gibbs[0]);
    Phi = Rcpp::as<Rcpp::NumericMatrix>(result_gibbs[1]);
    Z = Rcpp::as<Rcpp::NumericMatrix>(result_gibbs[2]);
    rho = Rcpp::as<Rcpp::NumericVector>(result_gibbs[3]);
    Z_sum_m = Rcpp::as<Rcpp::NumericMatrix>(result_gibbs[4]);
    Z_sum_1 = result_gibbs[5];
    
    // Marginal likelihood: 
    L_data[c-1] = gibbs_L_parallel( S1, Z, Phi, LDA_data, alpha );
    Z_prior[c-1] = prior_Z_parallel_new( S, S1, Z, Z_sum_m, alpha );
    I_prior[c-1] = prior_I( I, tree_str, rho );
    Phi_prior[c-1] = prior_Phi( S1, a00, a01, a10, a11, Phi, I );
    rho_prior[c-1] = prior_rho( b00, b01, b10, b11, rho, I );
    
    L_all[c-1] = L_data[c-1] + Z_prior[c-1] + I_prior[c-1] + Phi_prior[c-1] + rho_prior[c-1];
    
    
    // Collect posterior samples of hidden variables: 
    if ( (c > burn_in) & (c % interval == 0) ) {  
      
      Phi_samples[j] = Phi;
      I_samples[j] = I;
      rho_samples[j] = rho;
      Z_sum_samples[j] = Z_sum_m;
      
      j = j + 1;
      
    }
    
    
  }     
  
  // Collect final results for Gibbs sampler: 
  result_final[0] = Phi_samples;            
  result_final[1] = I_samples;             
  result_final[2] = rho_samples;      
  result_final[3] = Z_sum_samples;
  result_final[4] = alpha;
  
  result_final[5] = L_all;
  
  return(result_final);
  
}

