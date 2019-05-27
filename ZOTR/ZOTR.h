#ifndef __ZOTR_H__
#define __ZOTR_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <fstream>

#include "types.h"
#include "Skinny128256AVX2.h"

#define PIPE 64

#define TAG_BIT_NUMBER              128
#define TAG_BYTE_NUMBER             16

#define TRICK_BIT_NUMBER            120
#define TRICK_BYTE_NUMBER           15

#define ENCCN_BIT_NUMBER            8
#define ENCCN_BYTE_NUMBER           1

#define BLOCK_TRICK_BYTE_NUMBER      (BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER)

#define EXTENDED_KEY_BYTE_NUMBER    ((ROUND_NUMBER)*BLOCK_BYTE_NUMBER)


void ZOTR_enc(
	unsigned char* C,
	unsigned char* T,
	unsigned char* N,
	unsigned char* A,
	unsigned char* M,
	long long AByteN,
	long long MByteN,
	unsigned char *Seedkey
);

void Core_enc(
	unsigned char* C,
	unsigned char* Y,
	unsigned char* N,
	unsigned char* B,
	unsigned char* M,
	long long MByteN,
	unsigned char *Seedkey
);

void Hash_enc(
	unsigned char* Yh,
	unsigned char* Bh,
	long long BhByteN,
	unsigned char *Seedkey
);

#endif