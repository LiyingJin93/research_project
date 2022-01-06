##############################################################################
# Project: Generalized Partial Linear Spatially Varying Coefficient Models
##############################################################################
rm(list = ls())

library(GPLSVCM)
View(sample)

# Fit the model
mfit1= gplsvcm_fit(Y=as.matrix(sample$data[,1]), 
                   X=as.matrix(cbind(rep(1,1000),sample$data[,c(6:9)])),
                   ind_l=sample$ind_l,ind_nl=sample$ind_nl,U=as.matrix(sample$data[,c(10:11)]),
                   V=sample$V,Tr=sample$Tr,d=2,r=1,lambda=sample$lambda,family=sample$family)
View(mfit1)

# Fit the model with model selection
mfit2=gplsvcm_aglasso(Y=as.matrix(sample$data[,1]),X=as.matrix(sample$data[,c(6:9)]),
                      uvpop=cbind(sample$pop$u,sample$pop$v),
                      U=as.matrix(sample$data[,c(10:11)]),index=sample$index,
                      V=sample$V,Tr=sample$Tr,d=2,r=1,
                      penalty="agrLasso",lambda1=sample$lambda1,
                      lambda2=sample$lambda2,family=sample$family,lambda=sample$lambda)
View(mfit2)

# compute prediction intervals
set.seed(123)
PIs=compute_PIs(as.matrix(sample$data[,1]),
                as.matrix(cbind(rep(1,1000),sample$data[,c(6:9)])),
                mfit2$ind_l,mfit2$ind_nl,as.matrix(sample$data[,c(10:11)]),
                as.matrix(cbind(rep(1,1000),sample$data[,c(6:9)])),
                as.matrix(sample$data[,c(10:11)]),sample$V,sample$Tr,2,1,
                sample$lambda,sample$family,method="CV+", cp=0.95, nfold = 5)
PIs

# prediction:
Y_hat = gplsvcm_predict(mfit2$mfit,
                        X=as.matrix(cbind(rep(1,1000),sample$data[,c(6:9)])),
                        U=as.matrix(sample$data[,c(10:11)]))
Y_hat

# k-fold cross-validation MSPE:
set.seed(123)
MSPE = cv_gplsvcm(Y=as.matrix(sample$data[,1]),
                  X=as.matrix(cbind(rep(1,1000),sample$data[,c(6:9)])),
                  ind_l=mfit2$ind_l,ind_nl=mfit2$ind_nl,U=as.matrix(sample$data[,c(10:11)]),
                  sample$V,sample$Tr,2,1,sample$lambda,sample$family,nfold=5)
MSPE

# plot the estimated coefficients
gplsvcm_plot(mfit2$mfit,gridnumber=100,display=c(1,1),xlab=c("u1","u1"),ylab=c("u2","u2"),
             main=c(expression(paste("The Estimated Surface for"," ",hat(alpha)[1])),
                    expression(paste("The Estimated Surface for"," ",hat(alpha)[2]))))

