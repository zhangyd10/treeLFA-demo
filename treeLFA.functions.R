### Training with the gibbs-em algorithm: 
gibbs_EM_train <- function( topic.number,
                            data, tree_str,
                            Phi, I, rho, alpha, Z,
                            burn_in,
                            opt_N_1, cycle_1,
                            opt_N_2, cycle_2 ) {
  
  # Prepare the input data:
  S <- ncol(data)     # number of terminal disease codes on the tree
  S1 <- nrow(tree_str) - ncol(data)    # number of internal disease codes on the tree
  # S1 <- 5
  data <- as.matrix(data)
  
  
  
  # Prepara the tree str: change the names of diseases codes to indexes:
  for ( i in 1:nrow(tree_str) ){

    node_name <- tree_str$node[i]
    tree_str$node[i] <- i

    tree_str[parent==node_name,"parent"] <- i

  }

  tree_str[parent=="root","parent"] <- 0
  
  tree_str$node <- as.double(tree_str$node)
  tree_str$parent <- as.double(tree_str$parent)
  tree_str <- as.matrix(tree_str)



  
  ## Set the hyper-parameters:
  # Beta prior for phi based on incidence of diseases:
  inc <- min(colSums(data)) / nrow(data)

  if ( inc > 0.01 ) {
    a00 <- 0.1
    a01 <- 500
    a10 <- 2
    a11 <- 4
  } else if ( inc <= 0.01 & inc > 0.005 ) {
    a00 <- 0.1
    a01 <- 1000
    a10 <- 1.5
    a11 <- 3
  } else {
    a00 <- 0.1
    a01 <- 4000
    a10 <- 1.2
    a11 <- 3
  }


  K <- topic.number    # number of topics 
  D <- nrow(data)    # number of individuals

  # Beta prior for the transition probability of the Markov process on the tree:
  b00 <- 3
  b01 <- 20
  b10 <- 3
  b11 <- 3

  # Dirichelt prior for topic weights:
  # alpha <- c( 1, rep(0.1,K-1) )
  # alpha <- rep(0.1,K)
  
  
  
  
  # Initialization of hidden variables:
  if(missing(rho)) { 
    rho <- c(0.9,0.5)
  } 
  
  if(missing(Z)) { 
  Z <- matrix( nrow=D, ncol=S )
  for ( d in 1:D ){
    if ( sum(data[d,]) == 0 ) {
      Z[d,] <- rep(1,S)
    } else {
      for ( l in 1:S ) { Z[d,l] <- rcat( rep(1,K) )}
    }
  }
  #Z[1:D,1:S1] <- 0
  } 
  
  if(missing(I)) { 
  #I <- matrix( nrow=K, ncol=(S), 0 )
  I <- matrix( nrow=K, ncol=(S+S1), 0 )
  }
  
  if(missing(Phi)) { 
  Phi <- matrix( nrow=K, ncol=S )
  for ( j in 1:length(Phi) ) { Phi[j] <- rbeta(1,1,5000000) }
  } 
  
  
  
  
  # Run the gibbs-em algorithm:
  result <- gibbs_EM( K, S, S1, D,
            a00, a01, a10, a11,
            b00, b01, b10, b11,
            alpha,
            rho,
            Z, I, Phi,
            data,
            tree_str,
            burn_in,
            opt_N_1, cycle_1,
            opt_N_2, cycle_2 )

  names(result) <- c("Phi_samples","I_samples","rho_samples","alpha_samples","Z_samples","L_all")
  
  return(result)

}










### Training with the gibbs sampling algorithm: alpha fixed at values given by gibbs-em algorithm:
gibbs_train <- function( topic.number,
                         data, tree_str,
                         Phi, I, rho, alpha, Z,
                         burn_in, cycle, interval ) {
  
  # Prepare the input data:
  S <- ncol(data)    # number of terminal disease codes on the tree 
  S1 <- nrow(tree_str) - ncol(data)      # number of internal disease codes on the tree 
  # S1 <- 5
  data <- as.matrix(data)
  
  
  
  # Prepara the tree str: change the names of diseases codes to indexes:
  for ( i in 1:nrow(tree_str) ){
    
    node_name <- tree_str$node[i]
    tree_str$node[i] <- i
    
    tree_str[parent==node_name,"parent"] <- i
    
  }
  
  tree_str[parent=="root","parent"] <- 0
  
  tree_str$node <- as.double(tree_str$node)
  tree_str$parent <- as.double(tree_str$parent)
  tree_str <- as.matrix(tree_str)
  
  
  
  
  # Set the hyper-parameters:
  # Beta prior for phi based on the prevalences of diseases:
  inc <- min(colSums(data)) / nrow(data)
  
  if ( inc > 0.01 ) {
    a00 <- 0.1
    a01 <- 500
    a10 <- 2
    a11 <- 4
  } else if ( inc <= 0.01 & inc > 0.005 ) {
    a00 <- 0.1
    a01 <- 1000
    a10 <- 1.5
    a11 <- 3
  } else {
    a00 <- 0.1
    a01 <- 4000
    a10 <- 1.2
    a11 <- 3
  }
  
  K <- topic.number    # number of topics 
  D <- nrow(data)    # number of individuals
  
  # Beta prior for the transition probability of the Markov process on the tree
  b00 <- 3
  b01 <- 20
  b10 <- 20
  b11 <- 3
  
  # Dirichlet prior for theta: 
  # alpha <- c( 1, rep(0.1,K-1) )
  # alpha <- rep(0.1,K)
  
  
  
  # Run the gibbs-em algorithm:
  result <- gibbs( K, S, S1, D,
                   a00, a01, a10, a11,
                   b00, b01, b10, b11,
                   alpha,
                   rho,
                   Z, I, Phi,
                   data,
                   tree_str,
                   opt_N=1, burn_in, cycle, interval )
  names(result) <- c("Phi_samples","I_samples","rho_samples","Z_sum_samples","alpha","L_all")  
  return(result)
  
}










## Calculate predictive likelihood: 
predL <- function(alpha, phi, data, tree_str, IS){
  
  S1 <- nrow(tree_str) - ncol(data)
  data <- as.matrix(data)
  
  pl <- pred_L_mca( alpha, phi, data, IS=100, S1 )
  
  return(pl)
  
}





# plot topics using pheatmap: 
topics_plot <- function(phi) { 
  
  for ( i in 1:nrow(phi) ) { 
    phi[i,] <- rev(phi[i,])  
  }
  
  rownames(phi) <- paste("Topic",1:nrow(phi),sep="")
  colnames(phi) <- paste("Disease",1:ncol(phi),sep="")
    
  pheatmap( t(phi),
            cluster_cols=FALSE,
            cluster_rows=FALSE,
            color = colorRampPalette(brewer.pal(9,"Blues"))(300) )
  
}
