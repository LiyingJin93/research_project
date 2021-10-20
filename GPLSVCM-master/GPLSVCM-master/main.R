##############################################################################
# Project: Generalized Partial Linear Spatially Varying Coefficient Models
##############################################################################
rm(list = ls())

library(GPLSVCM)


 # Population:
 family=poisson()
 ngrid = 0.02

 # Data generation:
 pop = Datagenerator(family, ngrid)
 N=nrow(pop)

 # Triangulations and setup:
 Tr = Tr0; V = V0; n = 1000; d = 2; r = 1;

 # set up for smoothing parameters in the penalty term:
 lambda_start=0.0001; lambda_end=10; nlambda=10;
 lambda=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))

 # Generate Sample:
 ind_s=sample(N,n,replace=FALSE)
 data=as.matrix(pop[ind_s,])
 Y=as.matrix(data[,1]); X=data[,c(6:9)]; U=data[,c(10:11)];

 # True coefficents
 alpha=data[,c(2:3)]; beta=data[,c(4:5)];

 # Fit the model with model selection based on AIC:
 mfit1 = gplsvcm_fitwMIDF(Y, X, U, V, Tr, d , r , lambda,family,k_n=NULL,method="AIC",off = 0,r_theta = c(2, 8), eps= 0.01)

 # Fit the model with model selection based on BIC:
 mfit2 = gplsvcm_fitwMIDF(Y, X, U, V, Tr, d , r , lambda,family,k_n=NULL,method="BIC",off = 0,r_theta = c(2, 8), eps= 0.01)

 # prediction intervals:
 ind_l=mfit2$ind_l; ind_nl=mfit2$ind_nl;
 set.seed(123)
 PIs=compute_PIs(Y,X,ind_l,ind_nl,U,X,U,V,Tr,d,r,lambda,family,off = 0,r_theta = c(2, 8), eps= 0.01,method="CV+", cp=0.95, nfold = 10)

 # prediction:
  Y_hat = gplsvcm_predict(mfit2, X, U)

 # k-fold cross-validation:
 set.seed(123)
 MSPE = cv_gplsvcm(Y,X,ind_l,ind_nl,U,V,Tr,d,r,lambda,family,off = 0,r_theta =c(2, 8), eps= 0.01,nfold=10)

 # plot the estimated coefficients
 gplsvcm_plot(mfit2,gridnumber=100,display=c(1,1),xlab=c("u1","u1"),ylab=c("u2","u2"),
              main=c(expression(paste("The Estimated Surface for"," ",hat(alpha)[1])),
                     expression(paste("The Estimated Surface for"," ",hat(alpha)[2]))))

