#include "Graphics.h"

namespace COMMON
{
int GetXYIndex(int x, int y, int dimx)
{
    return y * dimx + x;
}

void GetIndexXY(int idx, int dimx, int &x, int &y)
{
    x = idx % dimx;
    y = idx / dimx;
}

bool GetMeshDirNextIdx(const DIMS &dim, const GAMEDIR &dir, int &idx)
{
    int x = 0, y = 0;
    COMMON::GetIndexXY(idx, dim.x, x, y);

    switch (dir) {
        case GAMEDIR::UP:
            if (y == dim.y - 1) {
                return true;
            }
            y++;
            break;

        case GAMEDIR::DOWN:
            if (y == 0) {
                return true;
            }
            y--;
            break;
        case GAMEDIR::LEFT:
            if (x == 0) {
                return true;
            }
            x--;
            break;
        case GAMEDIR::RIGHT:
            if (x == dim.x - 1) {
                return true;
            }
            x++;
            break;
        default:
            break;
    }

    idx = COMMON::GetXYIndex(x, y, dim.x);
    return false;
}
}    // namespace COMMON
