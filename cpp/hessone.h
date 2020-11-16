#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseQR>
class hessone
{

public:
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef Eigen::Matrix<double, 10, 1> Mat;
//Definitions for constructor and member functions 
hessone(int size)
{
    
}
int add(int i, int j) {
    printf("Adding number C++ called from the python wrapper %d %d",i,j);
    //std::cout<< i + j<<std::endl;
    return i+j;
}
Eigen::VectorXd myfunc(const Eigen::Ref<const Eigen::MatrixXcd>& a){
    std::vector <T> tripletList;
    Eigen::MatrixXd B;
    //double a[10][10];
        // for (int i=0; i<10; i++){
        //     for (int j=0; j<10; j++){
        //     a(i,j) = i*j;
        //     }          
        // }
    


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