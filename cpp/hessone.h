#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseQR>
#include "xtensor-python/pyarray.hpp"
#include "xtensor/xio.hpp"
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
    for (int iii=0; iii<nall^2; iii++){
        int tu = floor(iii/nall);// tu = iii // lh.nall
        int bc = iii % nall;// bc = iii % lh.nall
        int t = nzrow[tu];// t = lh.nzrow[tu]
        int u = nzcol[tu];// u = lh.nzcol[tu]
        int b = nzrow[bc];// b = lh.nzrow[bc]
        int c = nzcol[bc];// c = lh.nzcol[bc]
        //CmplxMat term;
        Eigen::MatrixXcd term = Eigen::MatrixXcd::Zero(16,ntrain-2);

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