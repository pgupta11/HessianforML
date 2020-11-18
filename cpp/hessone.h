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
int nnzr,nnzi,ndof,hesslen,nall;
public:
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, 10, 1> Mat;
typedef vector< tuple<int,int> > TupleList;
//Definitions for constructor and member functions 
hessone(int real, int imag, int dofs)
{

  /* 
   Get all the input parameters for buiding Hessian matirx here
  */  
    nnzr = real;
    nnzi = imag;
    ndof = dofs;
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
    denMO-----> this has 3 axis in python code..eigen can't do that..
    declare hesselement
    iii??
    tu, bc, t,u,b,c
    declare term
    */
    cout<<"Density shape"<< den.shape(2)<<endl;
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