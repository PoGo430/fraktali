#include "bitmap.h"
#include <stdio.h>
#include <stdint.h>
//#include <windows.h>

//#ifndef WIN32
#pragma pack(push, bmp_packing, 1)

typedef uint8_t BYTE; // 8 nit unsigned
typedef int32_t LONG; // 32 bit signed
typedef uint16_t WORD; // 16 bit unsigned
typedef char* LPCTSTR;
typedef uint32_t DWORD; // 32 bit unsigned
typedef uint16_t WCHAR; // 16 bit unicode char

typedef struct tagRGBQUAD {
  BYTE rgbBlue;
  BYTE rgbGreen;
  BYTE rgbRed;
  BYTE rgbReserved;
} RGBQUAD;

typedef struct tagBITMAPINFOHEADER {
  DWORD biSize;
  LONG  biWidth;
  LONG  biHeight;
  WORD  biPlanes;
  WORD  biBitCount;
  DWORD biCompression;
  DWORD biSizeImage;
  LONG  biXPelsPerMeter;
  LONG  biYPelsPerMeter;
  DWORD biClrUsed;
  DWORD biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;

typedef struct tagBITMAPFILEHEADER {
  WORD  bfType;
  DWORD bfSize;
  WORD  bfReserved1;
  WORD  bfReserved2;
  DWORD bfOffBits;
} BITMAPFILEHEADER, *PBITMAPFILEHEADER;

#pragma pack(pop, bmp_packing)
//#endif

// Return 0 on failure
int SaveBitmapToFile(void* data, unsigned int lWidth, unsigned int lHeight, unsigned int wBitsPerPixel, const char *fileName) {
    RGBQUAD palette[256];
    FILE *f;
    BITMAPINFOHEADER bmpInfoHeader = {0};
    BITMAPFILEHEADER bfh = {0};
    unsigned char *pBitmapBits=(unsigned char *)data;
	unsigned int i;
	char emptySpace[] = {0, 0, 0, 0};

    for(i = 0; i < 256; ++i)
    {
        palette[i].rgbBlue = i;
        palette[i].rgbGreen = (i * 2) % 256;
        palette[i].rgbRed = (i * 3) % 256;
    }

    // Set the size
    bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    // Bit count
    bmpInfoHeader.biBitCount = wBitsPerPixel;
    // Use all colors
    bmpInfoHeader.biClrImportant = 0;
    // Use as many colors according to bits per pixel
    bmpInfoHeader.biClrUsed = 0;
    // Store as un Compressed
    bmpInfoHeader.biCompression = 0; // BI_RGB;
    // Set the height in pixels
    bmpInfoHeader.biHeight = lHeight;
    // Width of the Image in pixels
    bmpInfoHeader.biWidth = lWidth;
    // Default number of planes
    bmpInfoHeader.biPlanes = 1;
    // Calculate the image size in bytes
    bmpInfoHeader.biSizeImage = lWidth* lHeight * (wBitsPerPixel/8);

    // This value should be values of BM letters i.e 0x4D42
    // 0x4D = M 0�42 = B storing in reverse order to match with endian
    bfh.bfType = 'B'+('M' << 8);
    // <<8 used to shift �M� to end
    // Offset to the RGBQUAD
    bfh.bfOffBits = sizeof(BITMAPINFOHEADER) + sizeof(BITMAPFILEHEADER) + sizeof(RGBQUAD) * 256;
    // Total size of image including size of headers
    bfh.bfSize = bfh.bfOffBits + bmpInfoHeader.biSizeImage;

    // Create the file in disk to write
    f=fopen(fileName, "wb");
    if (!f) {
        return 0;
    }

    // Write the File header
    fwrite(&bfh, sizeof(bfh), 1, f);
    // Write the bitmap info header
    fwrite(&bmpInfoHeader, sizeof(bmpInfoHeader), 1, f);
    // Write the palette
    fwrite(&palette[0], sizeof(RGBQUAD), 256, f);
    // Write the RGB Data
    if(lWidth%4 == 0) {
        fwrite(pBitmapBits, bmpInfoHeader.biSizeImage, 1, f);
    } else {
        for(i = 0; i < lHeight; ++i) {
            fwrite(&pBitmapBits[i * lWidth], lWidth, 1, f);
            fwrite(emptySpace, 4 - lWidth % 4, 1, f);
        }
    }
    // Close the file handle
    fclose(f);

    return 1;
}

int shraniBMP(void *slika, unsigned int w, unsigned int l, const char *f) {
	return SaveBitmapToFile(slika, w, l, 8, f);
}
