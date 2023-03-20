#ifndef BITMAP_H
#define BITMAP_H

int SaveBitmapToFile(void* data, unsigned int lWidth, unsigned int lHeight, unsigned int wBitsPerPixel, const char *fileName);
int shraniBMP(void *slika, unsigned int w, unsigned int l, const char *f);

#endif //BITMAP_H
