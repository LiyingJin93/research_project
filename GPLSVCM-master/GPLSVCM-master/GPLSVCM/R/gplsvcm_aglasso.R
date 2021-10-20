#' Fitting the generalized partial linear spatially varying coefficient model
#' with Variable Selection and Model Structure Identification by (Adaptive)
#' group lasso.
#'
#' \code{gplsvcm_aglasso} perform variable selection and model structure
#' identification at the same time to select the covariates with the linear and
#' nonlinear effects respectively out of a large number of covariates using BIC
#' or 10 fold cross validation and then fit the corresponding generalized
#' partial linear spatially varying coefficient model.
#'
#' @importFrom MGLM kr
#' @importFrom BPST basis
#' @importFrom MASS glm.nb
#' @importFrom Matrix bdiag
#' @importFrom grpreg grpreg cv.grpreg
#'
#'@param Y The response variable,a \code{n} by one matrix where \code{n} is the
#'  number of observations.
#'@param X The design matrix (without intercept) of \code{n} by \code{np} where
#'  \code{np} is the number of covariates. Each row is a vector of the
#'  covariates for an observation.
#'@param uvpop The coordinates of population grid points over the domain,
#'  default is \code{NULL}.
#'@param U A \code{n} by two matrix where each row is the coordinates of an
#'  observation.
#'@param index The row indexes of the observed data points \code{U} in the
#'  population grid points \code{uvpop}.
#'@param V A \code{N} by two matrix of vertices of a triangulation, where
#'  \code{N} is the number of vertices and each row is the coordinates for a
#'  vertex.
#'@param Tr A \code{n_Tr} by three triangulation matrix, where \code{n_Tr} is
#'  the number of triangles in the triangulation and each row is the indices of
#'  vertices in \code{V}.
#'@param d The degree of piecewise polynomials -- default is 2.
#'@param r The smoothness parameter and \code{r} \eqn{<} \code{d} -- default is
#'  1.
#'@param penalty The shrinkage method for variable selection and model structure
#'  identification, options are "agrLasso" for adaptive group lasso and
#'  "grLasso" for group lasso -- default is "agrLasso".
#'@param lambda1 The sequence of lambda values for group lasso,also need to be
#'  specified for computing the weight if using adaptive group lasso -- default
#'  is a grid of 50 lambda values that ranges uniformly on the log scale over
#'  0.0001 to 1.
#'@param lambda2 The sequence of lambda values used in the adaptive part for
#'  adaptive group lasso -- default is a grid of 50 lambda values that ranges
#'  uniformly on the log scale over 0.0001 to 1.
#'@param family The family object which specifies the distribution and link to
#'  use (see \code{\link{glm}} and \code{\link{family}}).
#'@param criteria A character string specifying the criteria of selecting lambda
#'  for (adaptive) group lasso. "BIC" is to use traditional Bayesian Information
#'  Criteria, "CV" is to use 10-fold cross validation -- default is "BIC".
#'@param lambda The vector of the candidates of smoothing penalty parameter --
#'  default is grid points of 10 to the power of a sequence from -6 to 6 by 0.5.
#'@param off The offset -- default is 0.
#'
#'@return The function returns a list of fitted object information from S3 class
#'  "gplsvcm", see the items of the list from \code{\link{gplsvcm_fit}}.
#'
#'@details The \code{gplsvcm_aglasso} function is used to fit a generalized
#'  partial linear spatially varying coefficient model when there is a large
#'  number of covariates and the linear and nonlinear parts of the design matrix
#'  \code{X} are not known before analysis. The construction of the polynomial
#'  spline functions is via \code{\link[BPST]{basis}}. It first perform a
#'  variable selection and model structure identification through adaptive group
#'  lasso via \code{\link[grpreg]{grpreg}} or \code{\link[grpreg]{cv.grpreg}}
#'  and output the selected model by specifying the parameters \code{ind_l} and
#'  \code{ind_nl} of the function \code{gplsvcm_fit}. Then the selected model is
#'  fitted by the function \code{gplsvcm_fit}.
#'
#'@references Wood, S., & Wood, M. S. (2015). Package ‘mgcv’. R package version,
#'  1, 29.
#'
#'  Breheny P (2016).grpreg: Regularization Paths for Regression Models with
#'  Grouped Covari-ates.Rpackage version 3.0-2,
#'  URLhttps://CRAN.R-project.org/packages=grpreg.
#'
#'  Wang L, Wang G, Li X, Mu J, Yu S, Wang Y, Kim M, Wang J (2019). BPST:
#'  Smoothing viaBivariate Spline over Triangulation.Rpackage version 1.0,
#'  URLhttps://GitHub.com/funstatpackages/BPST.
#'
#' @examples
#' # Population:
#' family=gaussian()
#' ngrid = 0.02
#'
#' # Data generation:
#' pop = Datagenerator(family, ngrid)
#' N=nrow(pop)
#'
#' # Triangulations and setup:
#' Tr = Tr0; V = V0; n = 2000; d = 2; r = 1;
#'
#' # set up for smoothing parameters in the penalty term:
#' lambda_start=0.0001; lambda_end=10; nlambda=10;
#' lambda=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))
#'
#' # set up for shrinkage parameters in the penalty term:
#' lambda_start=0.1; lambda_end=0.2; nlambda=50;
#' lambda1=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))
#' lambda2=exp(seq(log(lambda_start),log(lambda_end),length.out=nlambda))
#'
#' # Generate Sample:
#' ind_s=sample(N,n,replace=FALSE)
#' data=as.matrix(pop[ind_s,])
#' Y=data[,1]; X=data[,c(6:9)]; U=data[,c(10:11)];
#'
#' # True coefficents
#' beta0=1; alpha=data[,c(2:3)]; beta=data[,c(4:5)];
#'
#' # Fit the model with model selection by adaptive group lasso with BIC:
#' mfit1 = gplsvcm_aglasso(Y,X,uvpop=uvpop,U=U,index=ind_s,V,Tr,d=2,r=1,
#' penalty="agrLasso",lambda1=lambda1,lambda2=lambda2,family=family,lambda=lambda)
#' ind_nl=mfit1$ind_nl; ind_l=mfit1$ind_l;
#'
#' # Fit the model with model selection by adaptive group lasso with 10 fold CV:
#' mfit2 = gplsvcm_aglasso(Y,X,uvpop=uvpop,U=U,index=ind_s,V,Tr,d=2,r=1,
#' penalty="agrLasso",criteria="CV",lambda1=lambda1,lambda2=lambda2,family=family,lambda=lambda)
#'
#'@export


gplsvcm_aglasso=function(Y,X,uvpop=NULL,U,index=NULL,V,Tr,d=2,r=1,penalty="agrLasso",
                         lambda1= exp(seq(log(0.0001),log(1),length.out=50)),
                         lambda2= exp(seq(log(0.0001),log(1),length.out=50)),
                         family,criteria="BIC",lambda= 10^seq(-6,6,by = 0.5),
                         off=0)
{
# structure selection
  Bc=normalize_basis(uvpop,U,index,V,Tr,d,r)
  if (penalty=="grLasso"& criteria=="BIC"){
    mfit0=glasso_bic(Y,X,Bc,family,lambda1)
  }
  if (penalty=="grLasso"& criteria=="CV"){
    mfit0=glasso_cv(Y,X,Bc,family,lambda1)
  }
  if (penalty=="agrLasso"& criteria=="BIC"){
    mfit0=aglasso_bic(Y,X,Bc,family,lambda1,lambda2)
  }
  if (penalty=="agrLasso"& criteria=="CV"){
    mfit0=aglasso_cv(Y,X,Bc,family,lambda1,lambda2)
  }
  ind_l=mfit0$ind_l
  ind_nl=mfit0$ind_nl
  X=as.matrix(cbind(rep(1,dim(X)[1]),X))
  mfit=gplsvcm_fit(Y,X,(ind_l+1),(ind_nl+1),U,V,Tr,d=d,r=r,lambda=lambda,family,off=off)
  return(mfit)
}

