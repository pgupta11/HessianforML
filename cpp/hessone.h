#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseQR>
#include "xtensor-python/pyarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include <omp.h>
#include <complex>
#include <cmath>
namespace py = pybind11;
using namespace std;
class hessone
{
private:
int nnzr,nnzi,ndof,hesslen,nall,ntrain;
public:
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, 10, 1> Mat;
//typedef Eigen::Matrix<complex<double>, 16, ntrain-2> CmplxMat;
typedef vector< tuple<int,int> > TupleList;
//Definitions for constructor and member functions 
hessone(int real, int imag, int dofs, int train)
{

  /* 
   Get all the input parameters for buiding Hessian matirx here
  */  
    nnzr = real;
    nnzi = imag;
    ndof = dofs;
    ntrain = train;
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

void calc(const py::list& all,xt::pyarray<complex<double>>& den,xt::pyarray<double>& x){
    /* TO DO LIST
    hesslen, nall
    calculate nzrow and nzcol
    */
    hesslen = nnzr*(nnzr+1)+nnzi*nnzi;
    TupleList allnzs;
    for (auto item : all){
        tuple<int,int> t = py::cast<tuple<int,int>>(item);
        allnzs.push_back(t);
    }
    int nall = allnzs.size();
    cout<<"from cpp"<<nall<<endl;
    vector<int> nzrow(nall,0),nzcol(nall,0);
    for (int i = 0;i<nzrow.size();i++){
        nzrow[i] = get<0>(allnzs[i]);
        nzcol[i] = get<1>(allnzs[i]);
        }
    /* TO DO LIST
    check if we can get denMO and x_inp
    denMO-----> this has 3 axis in python code..eigen can't do that..xtensor is used for
    that
    declare hesselement
    iii---> goes over range(nall**2)
    tu, bc, t,u,b,c
    declare term
    */
    //cout<<"Density shape"<< den.shape(2)<<endl;
    int iii,tid;
    int nallsq = nall*nall;
    xt::xarray<complex<double>> term;
    xt::pyarray<double> term1;
    vector< complex<double> > vec,vec1;
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
        term1 = xt::zeros<double>({16, ntrain-2});
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
        cout<<"term"<<term<<endl;
        /*-------------------------------------------------------------------------------------------------------------------------*/
        if ((u==b)*(t<u))
        xt::view(term,4,xt::all()) = xt::sum(xt::view(den,xt::all(),t,xt::all())*xt::conj(xt::view(den,xt::all(),c,xt::all())),1);
        else
        xt::view(term,4,xt::all()) = 0;
        if ((u==c)*(t<u))
        xt::view(term,5,xt::all()) = xt::sum(xt::view(den,xt::all(),t,xt::all())*xt::conj(xt::view(den,xt::all(),b,xt::all())),1);
        else
        xt::view(term,5,xt::all()) = 0;
        if (t<u)
        xt::view(term,6,xt::all()) = -xt::view(den,xt::all(),u,c)*xt::conj(xt::view(den,xt::all(),u,b));
        else
        xt::view(term,6,xt::all()) = 0;
        if ((t<u)*(b<c))
        xt::view(term,7,xt::all()) = -xt::view(den,xt::all(),u,b)*xt::conj(xt::view(den,xt::all(),u,c));
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
        /*----------------------------------------------------------------------------------------------------------------------------*/

    }
    

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