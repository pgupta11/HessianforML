#include <Eigen/Sparse>
#include <vector>
#include <iostream>

typedef Eigen::SparseMatrix<std::complex<double>,RowMajor> SpMat;
typedef Eigen::Triplet<double> T;
vector <T> tripletList;
double a[10][10];
for (int i=0; i<10; i++){
    for (int j=0; j<10; j++){
        a[i][j] = i*j          
    }
}

for (int i=0; i<10; i++){
    for (int j=0; j<10; j++){
        if (a[i][j]!=0)
            tripletList.pushback(i,j,a[i][j]);          
    }
}

Spmat M(10,10)
M.setFromTriplets(tripletList.begin(), tripletList.end());
