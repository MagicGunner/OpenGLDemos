#include "imgui_internal.h"

void GetWindowsSize(float &w, float &h)
{
    ImGuiWindow *window = ImGui::FindWindowByName("ProgramWindow");
    if (window) {
        w = window->Size.x;
        h = window->Size.y;
    }
}