#ifndef KDTREE_H_
#define KDTREE_H_

#include "Common.h"
#include "kdtree.hpp"

namespace KDT
{

template <typename Type, int dim>
class KdTree
{
public:
    ~KdTree()
    {
        SDelete(_KD);
    }

    int Init(Type *pts, int num)
    {
        if (nullptr != _KD) {
            SDelete(_KD);
        }

        _KD = new (std::nothrow) kdtType(pts, pts + num);
        if (nullptr == _KD) {
            return 1;
        }

        _KD->optimise();

        return 0;
    }

    void GetNear(Type &pt)
    {
        if (_KD == nullptr) {
            return;
        }

        std::pair<kdtType::const_iterator, float> nearObj = _KD->find_nearest(pt);
        if (nearObj.first != _KD->end()) {
            pt.x = (*nearObj.first)[0];
            pt.y = (*nearObj.first)[1];
            pt.z = (*nearObj.first)[2];
            pt.idx = (*nearObj.first)[3];
        }
    }

    float GetNearZ(Type &pt)
    {
        if (_KD == nullptr) {
            return 0.0f;
        }

        std::pair<kdtType::const_iterator, float> nearObj = _KD->find_nearest(pt);
        if (nearObj.first != _KD->end()) {
            return (*nearObj.first)[2];
        }
    }

    void GetRange(Type &pt, float &dis, std::vector<Type> &out)
    {
        if (_KD == nullptr) {
            return;
        }

        _KD->find_within_range(pt, dis, std::back_insert_iterator<std::vector<Type>>(out));
    }

private:
    typedef KDTree::KDTree<dim, Type> kdtType;

    kdtType *_KD = nullptr;
};

}    // namespace KDT

#endif    // !KDTREE_H_
