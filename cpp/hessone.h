#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseQR>
#include "xtensor-python/pyarray.hpp"
#include <xtensor/xtensor.hpp>
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xarray.hpp"
#include <omp.h>
#include <complex>
#include <cmath>
#include <complex>
#include <type_traits>
namespace py = pybind11;
using namespace std;
using namespace xt::placeholders;  // required for `_` to work
/*------------------since * is missing -------------------------------------------*/
// template< typename T, typename SCALAR > inline
// typename std::enable_if< !std::is_same<T,SCALAR>::value, std::complex<T> >::type
// operator* ( const std::complex<T>& c, SCALAR n ) { return c * T(n) ; }

// template< typename T, typename SCALAR > inline
// typename std::enable_if< !std::is_same<T,SCALAR>::value, std::complex<T> >::type
// operator* ( SCALAR n, const std::complex<T>& c ) { return T(n) * c ; }
/*------------------since * is missing -------------------------------------------*/
class hessone
{
private:
int nnzr,nnzi,ndof,hesslen,nall,ntrain,drc;
public:
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, 10, 1> Mat;
//typedef Eigen::Matrix<complex<double>, 16, ntrain-2> CmplxMat;
typedef vector< tuple<int,int> > TupleList;
//Definitions for constructor and member functions 
hessone(int real, int imag, int dofs, int train, int rowcolumn)
{

  /* 
   Get all the input parameters for buiding Hessian matirx here
  */  
    nnzr = real;
    nnzi = imag;
    ndof = dofs;
    ntrain = train;
    drc = rowcolumn;
}
/*TO DO--> Test open mp*/
void test(){
    int tid, nthreads;
    #pragma omp parallel private(nthreads, tid)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  printf("Hello World from thread = %d\n", tid);
  /* Only master thread does this */
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  }
}

xt::pyarray<double> calc(const py::list& all,const py::list& realdof,const py::list& imagdof,xt::pyarray<complex<double>>& den,xt::pyarray<double>& x){
    /* TO DO LIST--->Debug
    hesslen, nall, allzs-->cout << "("<<get<0>(t) << " " << get<1>(t)<<")"<<endl;
    verify nzrow and nzcol
    */
    hesslen = nnzr*(nnzr+1)+nnzi*nnzi;
    TupleList allnzs,realnzs,imagnzs;
    for (auto item : all){
        tuple<int,int> t = py::cast<tuple<int,int>>(item);
        allnzs.push_back(t);
    }
    for (auto item : realdof){
        tuple<int,int> t = py::cast<tuple<int,int>>(item);
        realnzs.push_back(t);
    }
    for (auto item : imagdof){
        tuple<int,int> t = py::cast<tuple<int,int>>(item);
        imagnzs.push_back(t);
    }
    int nall = allnzs.size();
    //cout<<"from cpp"<<nall<<"nnzr"<<nnzr<<"nnzi"<<nnzi<<"ndof"<<ndof<<endl;
    vector<int> nzrow(nall,0),nzcol(nall,0);
    for (int i = 0;i<nzrow.size();i++){
        nzrow[i] = get<0>(allnzs[i]);
        nzcol[i] = get<1>(allnzs[i]);
        }

    /*TO DO LIST
    *create equivalent of self.nzreals,nzrealm
    */
    xt::xarray<int> nzrealm, nzimagm;
    nzrealm = -xt::ones<int>({drc,drc});
    nzimagm = -xt::ones<int>({drc,drc});
    int cnt = 0;
    for (int k=0; k<realnzs.size();k++){
        int i = get<0>(realnzs[k]);
        int j = get<1>(realnzs[k]);
        nzrealm(i,j) = cnt;
        cnt +=1;
    }
    for (int k=0; k<imagnzs.size();k++){
        int i = get<0>(imagnzs[k]);
        int j = get<1>(imagnzs[k]);
        nzimagm(i,j) = cnt;
        cnt +=1;
    }
    /*
    Debug: checked following:
    cout<<"Checking nzrealm" <<nzrealm<<endl;
    cout<<"Checking nzimagm" <<nzimagm<<endl;
    */
    int iii,tid;
    int nallsq = nall*nall;
    std::complex<double> myiota {0, 1};
    xt::xarray<complex<double>> term;
    xt::xarray<double> hesselement;
    hesselement = xt::zeros<double>({hesslen,hesslen});
    xt::xarray<complex<double>>stars_s = {{1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1}};
    xt::xarray<complex<double>>stars_a={{-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1}};
    xt::xarray<complex<double>>stars_sa;
    stars_s = xt::transpose(stars_s);
    stars_a = xt::transpose(stars_a);
    stars_sa = stars_s * stars_a;
    cout<<"stars_sa"<<stars_sa<<endl;
    //#pragma omp parallel for private(iii) //shared(den,x)
    for (iii=0; iii<nallsq; iii++){
        int tu = floor(iii/nall);// tu = iii // lh.nall
        int bc = iii % nall;// bc = iii % lh.nall
        int t = nzrow[tu];// t = lh.nzrow[tu]
        int u = nzcol[tu];// u = lh.nzcol[tu]
        int b = nzrow[bc];// b = lh.nzrow[bc]
        int c = nzcol[bc];// c = lh.nzcol[bc]
        //CmplxMat term;
        //Eigen::MatrixXcd term = Eigen::MatrixXcd::Zero(16,ntrain-2);
        term = xt::zeros<complex<double>>({16, ntrain-2});
        /*TO DO LIST
        get term calculations conjugate ?
        */
        
        if (t==b)
        xt::view(term,0,xt::all()) = xt::sum(xt::view(den,xt::all(),u,xt::all())*xt::conj(xt::view(den,xt::all(),c,xt::all())),1);
        else
        xt::view(term,0,xt::all()) = 0;
        if ((t==c)*(b<c))
        xt::view(term,1,xt::all()) = xt::sum(xt::view(den,xt::all(),u,xt::all())*xt::conj(xt::view(den,xt::all(),b,xt::all())),1);
        else
        xt::view(term,1,xt::all()) = 0;
        xt::view(term,2,xt::all()) = -xt::view(den,xt::all(),u,c)*xt::conj(xt::view(den,xt::all(),t,b));
        if (b<c)
        xt::view(term,3,xt::all()) = -xt::view(den,xt::all(),u,b)*xt::conj(xt::view(den,xt::all(),t,c));
        else
        xt::view(term,3,xt::all()) = 0;
        /*-------------------------------------------------------------------------------------------------------------------------*/
        if ((u==b)*(t<u))
        xt::view(term,4,xt::all()) = xt::sum(xt::view(den,xt::all(),t,xt::all())*xt::conj(xt::view(den,xt::all(),c,xt::all())),1);
        else
        xt::view(term,4,xt::all()) = 0;
        if ((u==c)*(t<u)*(b<c))
        xt::view(term,5,xt::all()) = xt::sum(xt::view(den,xt::all(),t,xt::all())*xt::conj(xt::view(den,xt::all(),b,xt::all())),1);
        else
        xt::view(term,5,xt::all()) = 0;
        if (t<u)
        xt::view(term,6,xt::all()) = -xt::view(den,xt::all(),t,c)*xt::conj(xt::view(den,xt::all(),u,b));
        else
        xt::view(term,6,xt::all()) = 0;
        if ((t<u)*(b<c))
        xt::view(term,7,xt::all()) = -xt::view(den,xt::all(),t,b)*xt::conj(xt::view(den,xt::all(),u,c));
        else
        xt::view(term,7,xt::all()) = 0;
        /*------------------------------------------------------------------------------------------------------------------------*/
        xt::view(term,8,xt::all()) = -xt::view(den,xt::all(),b,t)*xt::conj(xt::view(den,xt::all(),c,u));
        if (b<c)
        xt::view(term,9,xt::all()) = -xt::view(den,xt::all(),c,t)*xt::conj(xt::view(den,xt::all(),b,u));
        else
        xt::view(term,9,xt::all()) = 0;
        if (u==c)
        xt::view(term,10,xt::all()) = xt::sum(xt::view(den,xt::all(),xt::all(),t)*xt::conj(xt::view(den,xt::all(),xt::all(),b)),1);
        else
        xt::view(term,10,xt::all()) = 0;
        if ((u==b)*(b<c))
        xt::view(term,11,xt::all()) = xt::sum(xt::view(den,xt::all(),xt::all(),t)*xt::conj(xt::view(den,xt::all(),xt::all(),c)),1);
        else
        xt::view(term,11,xt::all()) = 0;
        /*----------------------------------------------------------------------------------------------------------------------------*/
        if (t<u)
        xt::view(term,12,xt::all()) = -xt::view(den,xt::all(),b,u)*xt::conj(xt::view(den,xt::all(),c,t));
        else
        xt::view(term,12,xt::all()) = 0;
        if ((b<c)*(t<u))
        xt::view(term,13,xt::all()) = -xt::view(den,xt::all(),c,u)*xt::conj(xt::view(den,xt::all(),b,t));
        else
        xt::view(term,13,xt::all()) = 0;
        if ((t==c)*(t<u))
        xt::view(term,14,xt::all()) = xt::sum(xt::view(den,xt::all(),xt::all(),u)*xt::conj(xt::view(den,xt::all(),xt::all(),b)),1);
        else
        xt::view(term,14,xt::all()) = 0;
        if ((t==b)*(t<u)*(b<c))
        xt::view(term,15,xt::all()) = xt::sum(xt::view(den,xt::all(),xt::all(),u)*xt::conj(xt::view(den,xt::all(),xt::all(),c)),1);
        else
        xt::view(term,15,xt::all()) = 0;
        //cout<<"term"<<term<<endl;
        /*----------------------------------------------------------------------------------------------------------------------------*/
        int row00,col00,row01,col01,row02,col02,row11,col11,row12,col12,row22,col22;
        xt::xarray<complex<double>> term01, term02, term11, term12,term22;
        for (int s=0; s<ndof; s++){
            for (int a=0; a<ndof; a++){
                if ((s==0)&&(a==0)&&(nzrealm(t,u)>=0)&&(nzrealm(b,c)>=0)){
                    // no extra factor for 00 block
                    row00 = nzrealm(t,u);
                    col00 = nzrealm(b,c);
                    hesselement(row00,col00) = 2*xt::real(xt::sum(term)[0]);
                    }
                if ((s==0)&&(a<nnzr)&&(nzrealm(t,u)>=0)&&(nzrealm(b,c)>=0)){
                    // work on the 01 block
                    term01 = term*xt::view(x,xt::all(),a);
                    row01 = nzrealm(t,u);
                    col01 = (a+1)*nnzr+nzrealm(b,c);
                    hesselement(row01,col01) = 2*xt::real(xt::sum(term01)[0]);
                    //if(hesselement(row01,col01)==0) cout<<"row01,col01"<<row01<<","<<col01<<endl;
                    }
                if ((s==0)&&(a>=nnzr)&&(nzrealm(t,u)>=0)&&(nzimagm(b,c)>=0)){
                    // work on the 02 block
                    // need a star pattern for index a
                    term02 = term*stars_a;
                    term02 = term02*(myiota)*xt::view(x,xt::all(),a);
                    row02 = nzrealm(t,u);
                    col02 = nnzr*nnzr+(a-nnzr)*nnzi+nzimagm(b,c);
                    hesselement(row02,col02) = 2*xt::real(xt::sum(term02)[0]);
                    }
                if ((s<nnzr)&&(a<nnzr)&&(nzrealm(t,u)>=0)&&(nzrealm(b,c)>=0)){
                    // overall factor for 11 block
                    term11 = term*xt::view(x,xt::all(),s)*xt::view(x,xt::all(),a);
                    row11 = (s+1)*nnzr+nzrealm(t,u);
                    col11 = (a+1)*nnzr+nzrealm(b,c);
                    hesselement(row11,col11) = 2*xt::real(xt::sum(term11)[0]);
                    }
                if ((s<nnzr)&&(a>=nnzr)&&(nzrealm(t,u)>=0)&&(nzimagm(b,c)>=0)){
                    // work on the 12 block
                    // here we need to use the star pattern for index a
                    term12 = term*stars_a;
                    term12 = term12*xt::view(x,xt::all(),s)*(myiota)*xt::view(x,xt::all(),a);
                    row12 = (s+1)*nnzr+nzrealm(t,u);
                    col12 = nnzr*nnzr+(a-nnzr)*nnzi+nzimagm(b,c);
                    hesselement(row12,col12) = 2*xt::real(xt::sum(term12)[0]);
                    }
                if ((s>=nnzr)&&(a>=nnzr)&&(nzimagm(t,u)>=0)&&(nzimagm(b,c)>=0)){
                    // work on the 22 block
                    // here we need to use the star pattern for index s and a
                    term22 = term*stars_sa;
                    term22 = term22*(myiota)*xt::view(x,xt::all(),s)*(myiota)*xt::view(x,xt::all(),a);
                    row22 = nnzr*nnzr+(s-nnzr)*nnzi+nzimagm(t,u);
                    col22 = nnzr*nnzr+(a-nnzr)*nnzi+nzimagm(b,c);
                    hesselement(row22,col22) = 2*xt::real(xt::sum(term22)[0]);
                    }
            
            }    
        }
    }
    //std::vector<size_t> shape = { hesslen,hesslen };
    //xt::xarray<double,xt::layout_type::dynamic> hessmat (shape);
    xt::pyarray<double> hessmat;
    hessmat = xt::zeros<double>({hesslen,hesslen});
    hessmat = hesselement;
    xt::view(hessmat,xt::range(nnzr,_),xt::range(0,nnzr)) = xt::transpose(xt::view(hessmat,xt::range(0,nnzr),xt::range(nnzr,_)));
    xt::view(hessmat,xt::range(nnzr*(nnzr+1),_),xt::range(nnzr,nnzr*(nnzr+1))) = xt::transpose(xt::view(hessmat,xt::range(nnzr,nnzr*(nnzr+1)),xt::range(nnzr*(nnzr+1),_)));
    //cout<<"Hessian Matrix"<<hessmat<<endl;
    return hessmat;  
}
//This was just a test
Eigen::VectorXd myfunc(const Eigen::Ref<const Eigen::MatrixXcd>& a){
    vector <T> tripletList;
    Eigen::MatrixXd B;
    for (int i=0; i<10; i++){
        for (int j=0; j<10; j++){
            B(i,j) = a(i,j).real();
            if (B(i,j)!=0)
                tripletList.push_back(T(i,j,B(i,j)));          
        }
    }   
    SpMat M(10,10);
    Mat c(10,1);
    Eigen::VectorXd x ;
    //b.setIdentity();
    for (int i=0; i<10; i++){
        c(i,1)  =1;
    }
    M.setFromTriplets(tripletList.begin(), tripletList.end());
    //Eigen::SparseQR<SpMat,Eigen::COLAMDOrdering<int>> solver;
    Eigen::LeastSquaresConjugateGradient<SpMat> solver;
    solver.compute(M);
    x = solver.solve(c);
    

    //std::cout<<a.real()<<std::endl;
    // //std::cout<<"print row number"<<a<<std::endl;
    return x;
}
};