#ifndef MACROHELPER_H
#define MACROHELPER_H

#define VERSTRING(arg) #arg

#define MAKEVERSION(a1, a2, a3, a4) \
    VERSTRING(a1) "." VERSTRING(a2) "." VERSTRING(a3) "." VERSTRING(a4)
	
#endif
