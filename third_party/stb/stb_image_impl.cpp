// stb_image.h is a single-header library: its actual function bodies only
// get compiled in whichever translation unit defines
// STB_IMAGE_IMPLEMENTATION before including it. This is that one
// translation unit -- every other file just includes stb_image.h normally
// (declarations only, no implementation), same pattern as any header/source
// pair.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
