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
Tr = Tr0; V = V0; n = 2000; d = 2; r = 1;

# set up for smoothing parameters in the penalty term:
lambda_start=0.0001; lambda_end=10; nlambda=10
lambda=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))

########### Fit the oracle model:
# Generate Sample:
ind_s=sample(N,n,replace=FALSE)
data=as.matrix(pop[ind_s,])
Y=as.matrix(data[,1]); alpha=data[,c(2:3)]; beta=data[,c(4:5)];
X=as.matrix(cbind(rep(1,length(Y)),data[,c(6:9)])); ind_l=c(1,4,5); ind_nl=c(2,3);
U=as.matrix(data[,c(10:11)]);

# Fit the model
mfit0 = gplsvcm_fit(Y, X,ind_l,ind_nl,U, V, Tr, d , r , lambda,family,off = 0,r_theta = c(2, 8), eps= 0.01)


############ Model Selection
# set up for shrinkage penalty term
lambda_start=0.001
lambda_end=0.2
nlambda=100
lambda1=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))
lambda_start=0.0001
lambda_end=1
nlambda=100
lambda2=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))

# Preparation for normalizing spline basis
pop=as.data.frame(pop)
uvpop=cbind(pop$u,pop$v)
Basis_full = basis(V, Tr, d, r,uvpop)
ind_inside = Basis_full$Ind.inside
pop=pop[ind_inside,]
N=nrow(pop)
ind_s=sample(N,n,replace=FALSE)
data=as.matrix(pop[ind_s,])

# Fit the model with model selection
Y=as.matrix(data[,1])
X=as.matrix(data[,c(6:9)])
uvpop=cbind(pop$u,pop$v)
U=as.matrix(data[,10:11])
index=ind_s
mfit0=gplsvcm_aglasso(Y,X,uvpop=uvpop,U=U,index=index,V,Tr,d=2,r=1,
                      penalty="agrLasso",lambda1=lambda1,lambda2=lambda2,
                      family=family,lambda=lambda)
# the fitted model object
mfit=mfit0$mfit
ind_nl=mfit0$ind_nl
ind_l=mfit0$ind_l


# compute prediction intervals
set.seed(123)
X=cbind(rep(1,length(Y)),data[,c(6:9)])
PIs=compute_PIs(Y,X,ind_l,ind_nl,U,X,U,V,Tr,d,r,lambda,family,method="CV+", cp=0.95, nfold = 5)

# prediction:
Y_hat = gplsvcm_predict(mfit, X, U)

# k-fold cross-validation MSPE:
set.seed(123)
MSPE = cv_gplsvcm(Y,X,ind_l,ind_nl,U,V,Tr,d,r,lambda,family,nfold=5)

# plot the estimated coefficients
gplsvcm_plot(mfit,gridnumber=100,display=c(1,1),xlab=c("u1","u1"),
             ylab=c("u2","u2"),
             main=c(expression(paste("The Estimated Surface for"," ",hat(alpha)[1])),
                    expression(paste("The Estimated Surface for"," ",hat(alpha)[2]))))

