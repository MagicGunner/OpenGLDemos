#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {

    Vector3f v(1.0, 2.0, 3.0);
    Vector3f w(1.0, 0.0, 1.0);

    cout << v.cross(w) << endl;

    return 0;
}
