#test example
# SampleFunc <- function(x) {
#   return((x-4)**2+1)
# }
# 
# o <- psoptim(rep(NA,1),SampleFunc,lower=-5,upper=5,control=list(abstol=1e-8,trace=0,REPORT=1,
#              trace.stats=TRUE))
# print(o$par)
# print(o$value)
# print(o$message)

#next is to calculate our function
source('gradient_descent.R')
source('pareto_simple_opt_func.R')
library(POT)
library(pso)

# m<-rgpd(200,loc=0,scale=1,shape=0.05)
# OptTheta <- psoptim(rep(NA,1),function(theta) G(theta,m),lower=-1000,upper = 1/max(m),control=list(abstol=1e-8,trace=0,REPORT=10,
#                                                               trace.stats=TRUE,maxit=500))
# theta <- OptTheta$par
# shape <- sum(log(1 - theta * m)) / length(m)
# scale <- -1 * shape / theta

# The following function is for minimizing the target function of finding parameters for GPD by PSO 
PSOptTheta <- function(x) {
  OptTheta <- psoptim(rep(NA,1),function(theta) G(theta,x),lower=-1000,upper = 1/max(m),control=list(abstol=1e-8,trace=0,REPORT=10,
                                                                                                     trace.stats=TRUE,maxit=500))
  return(OptTheta)
}

