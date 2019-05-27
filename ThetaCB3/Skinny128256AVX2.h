/*
Constants, parameters and intrinsics used for the implementation of
Skinny128/128.
*/

#ifndef SKINNY128128AVX2_H
#define SKINNY128128AVX2_H
#include "immintrin.h"
#include "types.h"

#define ROUND_NUMBER                48

#define BLOCK_BIT_NUMBER            128
#define BLOCK_BYTE_NUMBER           16

#define HALF_BLOCK_BIT_NUMBER       64
#define HALF_BLOCK_BYTE_NUMBER       8

#define KEY_BIT_NUMBER              128
#define KEY_BYTE_NUMBER             16

#define TKEY_BIT_NUMBER             256
#define TKEY_BYTE_NUMBER             32

//Types
#define u16 unsigned short
#define u32 unsigned
#define u64 unsigned long long
#define u256 __m256i

//Intrinsics
#define XOR _mm256_xor_si256
#define AND _mm256_and_si256
#define ANDNOT _mm256_andnot_si256
#define OR _mm256_or_si256
#define NOT(x) _mm256_xor_si256(x, _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1))
#define NOR(x, y) _mm256_xor_si256(_mm256_or_si256(x, y), _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1))
#define SHIFTR64(x, y) _mm256_srli_epi64(x, y)
#define SHIFTL64(x, y) _mm256_slli_epi64(x, y)

#define SR1(x) _mm256_shuffle_epi8(x, _mm256_set_epi8(30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31))
#define SR2(x) _mm256_shuffle_epi8(x, _mm256_set_epi8(29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30))
#define SR3(x) _mm256_shuffle_epi8(x, _mm256_set_epi8(28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30,29))

#define LOAD(src) _mm256_loadu_si256((__m256i *)(src))
#define STORE(dest,src) _mm256_storeu_si256((__m256i *)(dest),src)

#define MASK1 _mm256_set_epi32(0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa, 0xaaaaaaaa)
#define MASK2 _mm256_set_epi32(0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc, 0xcccccccc)
#define MASK4 _mm256_set_epi32(0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0, 0xf0f0f0f0)
#define MASK32 _mm256_set_epi32(0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000, 0xffffffff, 0x00000000)
#define MASK64 _mm256_set_epi32(0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000)

#define SWAPMOVE(a, b, mask, shift) \
{ \
	u256 T = AND(XOR(SHIFTL64(a, shift), b), mask); \
	b = XOR(b, T); \
    a = XOR(a, SHIFTR64(T, shift)); \
}

//Swap move for shifting by 64
#define SWAPMOVEBY64(a, b, mask) \
{ \
	u256 T = AND(XOR(_mm256_shuffle_epi8(a, _mm256_set_epi8(23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,31,30,29,28,27,26,25,24)), b), mask); \
	b = XOR(b, T); \
    a = XOR(a, _mm256_shuffle_epi8(T, _mm256_set_epi8(7,6,5,4,3,2,1,0,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8))); \
}

inline void pack_message(u256 x[32]) {
	int i;

	//Seperate bits for S-box
	for (i = 0; i < 4; i++) {
		SWAPMOVE(x[8 * i + 0], x[8 * i + 1], MASK1, 1);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 3], MASK1, 1);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 5], MASK1, 1);
		SWAPMOVE(x[8 * i + 6], x[8 * i + 7], MASK1, 1);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 2], MASK2, 2);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 3], MASK2, 2);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 6], MASK2, 2);
		SWAPMOVE(x[8 * i + 5], x[8 * i + 7], MASK2, 2);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 4], MASK4, 4);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 5], MASK4, 4);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 6], MASK4, 4);
		SWAPMOVE(x[8 * i + 3], x[8 * i + 7], MASK4, 4);
	}

	//Group the rows for efficient MixColumns implementation
	for (i = 0; i < 8; i++) {
		SWAPMOVE(x[i + 8], x[i + 0], MASK32, 32);
		SWAPMOVE(x[i + 24], x[i + 16], MASK32, 32);

		SWAPMOVEBY64(x[i + 16], x[i + 0], MASK64);
		SWAPMOVEBY64(x[i + 24], x[i + 8], MASK64);
	}
}

inline void pack_key(u256 x[32]) {
	int i;

	//Seperate bits for S-box
	for (i = 0; i < 4; i++) {
		SWAPMOVE(x[8 * i + 0], x[8 * i + 1], MASK1, 1);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 3], MASK1, 1);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 5], MASK1, 1);
		SWAPMOVE(x[8 * i + 6], x[8 * i + 7], MASK1, 1);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 2], MASK2, 2);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 3], MASK2, 2);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 6], MASK2, 2);
		SWAPMOVE(x[8 * i + 5], x[8 * i + 7], MASK2, 2);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 4], MASK4, 4);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 5], MASK4, 4);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 6], MASK4, 4);
		SWAPMOVE(x[8 * i + 3], x[8 * i + 7], MASK4, 4);
	}

	//Group the rows for efficient MixColumns implementation
	for (i = 0; i < 8; i++) {
		SWAPMOVE(x[i + 8], x[i + 0], MASK32, 32);
		SWAPMOVE(x[i + 24], x[i + 16], MASK32, 32);

		SWAPMOVEBY64(x[i + 16], x[i + 0], MASK64);
		SWAPMOVEBY64(x[i + 24], x[i + 8], MASK64);
	}
}

inline void unpack_message(u256 x[32]) {
	int i;

	//Group the rows for efficient MixColumns implementation
	for (i = 0; i < 8; i++) {
		SWAPMOVE(x[i + 8], x[i + 0], MASK32, 32);
		SWAPMOVE(x[i + 24], x[i + 16], MASK32, 32);

		SWAPMOVEBY64(x[i + 16], x[i + 0], MASK64);
		SWAPMOVEBY64(x[i + 24], x[i + 8], MASK64);
	}

	//Seperate bits for S-box
	for (i = 0; i < 4; i++) {
		SWAPMOVE(x[8 * i + 0], x[8 * i + 1], MASK1, 1);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 3], MASK1, 1);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 5], MASK1, 1);
		SWAPMOVE(x[8 * i + 6], x[8 * i + 7], MASK1, 1);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 2], MASK2, 2);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 3], MASK2, 2);
		SWAPMOVE(x[8 * i + 4], x[8 * i + 6], MASK2, 2);
		SWAPMOVE(x[8 * i + 5], x[8 * i + 7], MASK2, 2);

		SWAPMOVE(x[8 * i + 0], x[8 * i + 4], MASK4, 4);
		SWAPMOVE(x[8 * i + 1], x[8 * i + 5], MASK4, 4);
		SWAPMOVE(x[8 * i + 2], x[8 * i + 6], MASK4, 4);
		SWAPMOVE(x[8 * i + 3], x[8 * i + 7], MASK4, 4);
	}
}

const unsigned char RC[62] = {
	0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
	0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
	0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
	0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
	0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13,
	0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28,
	0x10, 0x20 };

inline void key_schedule(u256 k2[32], u256 srk[ROUND_NUMBER][16])
{
	int r, i, j;
	u256 rc;
	u256 tk2[2][32];
	u256 tmp2LFSR;

	rc = _mm256_set_epi64x(0x000000FF000000FFull,
		0x000000FF000000FFull,
		0x000000FF000000FFull,
		0x000000FF000000FFull);

	for (j = 0; j < 32; j++)
	{
		tk2[0][j] = k2[j];
	}

	pack_key(tk2[0]);

	r = 0;
	while (r < ROUND_NUMBER)
	{
		//Get round key
		srk[r][ 0] = tk2[((r >> 1) + 0) & 1][ 0];
		srk[r][ 1] = tk2[((r >> 1) + 0) & 1][ 1];
		srk[r][ 2] = tk2[((r >> 1) + 0) & 1][ 2];
		srk[r][ 3] = tk2[((r >> 1) + 0) & 1][ 3];
		srk[r][ 4] = tk2[((r >> 1) + 0) & 1][ 4];
		srk[r][ 5] = tk2[((r >> 1) + 0) & 1][ 5];
		srk[r][ 6] = tk2[((r >> 1) + 0) & 1][ 6];
		srk[r][ 7] = tk2[((r >> 1) + 0) & 1][ 7];
		srk[r][ 8] = tk2[((r >> 1) + 0) & 1][ 8];
		srk[r][ 9] = tk2[((r >> 1) + 0) & 1][ 9];
		srk[r][10] = tk2[((r >> 1) + 0) & 1][10];
		srk[r][11] = tk2[((r >> 1) + 0) & 1][11];
		srk[r][12] = tk2[((r >> 1) + 0) & 1][12];
		srk[r][13] = tk2[((r >> 1) + 0) & 1][13];
		srk[r][14] = tk2[((r >> 1) + 0) & 1][14];
		srk[r][15] = tk2[((r >> 1) + 0) & 1][15];

		if (RC[r] >> 5 & 1) srk[r][14] = XOR(srk[r][14], rc);
		if (RC[r] >> 4 & 1) srk[r][15] = XOR(srk[r][15], rc);
		if (RC[r] >> 3 & 1) srk[r][4] = XOR(srk[r][4], rc);
		if (RC[r] >> 2 & 1) srk[r][5] = XOR(srk[r][5], rc);
		if (RC[r] >> 1 & 1) srk[r][6] = XOR(srk[r][6], rc);
		if (RC[r] >> 0 & 1) srk[r][7] = XOR(srk[r][7], rc);

		//Apply bit permutation
		for (i = 0; i < 8; i++)
		{
			tk2[((r >> 1) + 1) & 1][16 + 0 + i] = XOR(
				_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][16 + i], _mm256_set_epi8(0xff, 28, 0xff, 29, 0xff, 24, 0xff, 25, 0xff, 20, 0xff, 21, 0xff, 16, 0xff, 17, 0xff, 12, 0xff, 13, 0xff, 8, 0xff, 9, 0xff, 4, 0xff, 5, 0xff, 0, 0xff, 1)),
				_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][24 + i], _mm256_set_epi8(29, 0xff, 31, 0xff, 25, 0xff, 27, 0xff, 21, 0xff, 23, 0xff, 17, 0xff, 19, 0xff, 13, 0xff, 15, 0xff, 9, 0xff, 11, 0xff, 5, 0xff, 7, 0xff, 1, 0xff, 3, 0xff)));
			tk2[((r >> 1) + 1) & 1][16 + 8 + i] = XOR(
				_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][16 + i], _mm256_set_epi8(31, 0xff, 0xff, 30, 27, 0xff, 0xff, 26, 23, 0xff, 0xff, 22, 19, 0xff, 0xff, 18, 15, 0xff, 0xff, 14, 11, 0xff, 0xff, 10, 7, 0xff, 0xff, 6, 3, 0xff, 0xff, 2)),
				_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][24 + i], _mm256_set_epi8(0xff, 28, 30, 0xff, 0xff, 24, 26, 0xff, 0xff, 20, 22, 0xff, 0xff, 16, 18, 0xff, 0xff, 12, 14, 0xff, 0xff, 8, 10, 0xff, 0xff, 4, 6, 0xff, 0xff, 0, 2, 0xff)));
		}

		tmp2LFSR = XOR(tk2[((r >> 1) + 1) & 1][16 + 5], tk2[((r >> 1) + 1) & 1][16 + 7]);
		tk2[((r >> 1) + 1) & 1][16 + 7] = tk2[((r >> 1) + 1) & 1][16 + 6];
		tk2[((r >> 1) + 1) & 1][16 + 6] = tk2[((r >> 1) + 1) & 1][16 + 5];
		tk2[((r >> 1) + 1) & 1][16 + 5] = tk2[((r >> 1) + 1) & 1][16 + 4];
		tk2[((r >> 1) + 1) & 1][16 + 4] = tk2[((r >> 1) + 1) & 1][16 + 3];
		tk2[((r >> 1) + 1) & 1][16 + 3] = tk2[((r >> 1) + 1) & 1][16 + 2];
		tk2[((r >> 1) + 1) & 1][16 + 2] = tk2[((r >> 1) + 1) & 1][16 + 1];
		tk2[((r >> 1) + 1) & 1][16 + 1] = tk2[((r >> 1) + 1) & 1][16 + 0];
		tk2[((r >> 1) + 1) & 1][16 + 0] = tmp2LFSR;

		tmp2LFSR = XOR(tk2[((r >> 1) + 1) & 1][16 + 5 + 8], tk2[((r >> 1) + 1) & 1][16 + 7 + 8]);
		tk2[((r >> 1) + 1) & 1][16 + 7 + 8] = tk2[((r >> 1) + 1) & 1][16 + 6 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 6 + 8] = tk2[((r >> 1) + 1) & 1][16 + 5 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 5 + 8] = tk2[((r >> 1) + 1) & 1][16 + 4 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 4 + 8] = tk2[((r >> 1) + 1) & 1][16 + 3 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 3 + 8] = tk2[((r >> 1) + 1) & 1][16 + 2 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 2 + 8] = tk2[((r >> 1) + 1) & 1][16 + 1 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 1 + 8] = tk2[((r >> 1) + 1) & 1][16 + 0 + 8];
		tk2[((r >> 1) + 1) & 1][16 + 0 + 8] = tmp2LFSR;

		r++;

		//Get round key
		srk[r][0]  = tk2[((r >> 1) + 1) & 1][16 +  0];
		srk[r][1]  = tk2[((r >> 1) + 1) & 1][16 +  1];
		srk[r][2]  = tk2[((r >> 1) + 1) & 1][16 +  2];
		srk[r][3]  = tk2[((r >> 1) + 1) & 1][16 +  3];
		srk[r][4]  = tk2[((r >> 1) + 1) & 1][16 +  4];
		srk[r][5]  = tk2[((r >> 1) + 1) & 1][16 +  5];
		srk[r][6]  = tk2[((r >> 1) + 1) & 1][16 +  6];
		srk[r][7]  = tk2[((r >> 1) + 1) & 1][16 +  7];
		srk[r][8]  = tk2[((r >> 1) + 1) & 1][16 +  8];
		srk[r][9]  = tk2[((r >> 1) + 1) & 1][16 +  9];
		srk[r][10] = tk2[((r >> 1) + 1) & 1][16 + 10];
		srk[r][11] = tk2[((r >> 1) + 1) & 1][16 + 11];
		srk[r][12] = tk2[((r >> 1) + 1) & 1][16 + 12];
		srk[r][13] = tk2[((r >> 1) + 1) & 1][16 + 13];
		srk[r][14] = tk2[((r >> 1) + 1) & 1][16 + 14];
		srk[r][15] = tk2[((r >> 1) + 1) & 1][16 + 15];

		if (RC[r] >> 5 & 1) srk[r][14] = XOR(srk[r][14], rc);
		if (RC[r] >> 4 & 1) srk[r][15] = XOR(srk[r][15], rc);
		if (RC[r] >> 3 & 1) srk[r][4] = XOR(srk[r][4], rc);
		if (RC[r] >> 2 & 1) srk[r][5] = XOR(srk[r][5], rc);
		if (RC[r] >> 1 & 1) srk[r][6] = XOR(srk[r][6], rc);
		if (RC[r] >> 0 & 1) srk[r][7] = XOR(srk[r][7], rc);

		if (r < (ROUND_NUMBER - 1))
		{
			//Apply bit permutation
			for (i = 0; i < 8; i++)
			{
				tk2[((r >> 1) + 1) & 1][0 + i] = XOR(
					_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][16 + i - 16], _mm256_set_epi8(0xff, 28, 0xff, 29, 0xff, 24, 0xff, 25, 0xff, 20, 0xff, 21, 0xff, 16, 0xff, 17, 0xff, 12, 0xff, 13, 0xff, 8, 0xff, 9, 0xff, 4, 0xff, 5, 0xff, 0, 0xff, 1)),
					_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][24 + i - 16], _mm256_set_epi8(29, 0xff, 31, 0xff, 25, 0xff, 27, 0xff, 21, 0xff, 23, 0xff, 17, 0xff, 19, 0xff, 13, 0xff, 15, 0xff, 9, 0xff, 11, 0xff, 5, 0xff, 7, 0xff, 1, 0xff, 3, 0xff)));
				tk2[((r >> 1) + 1) & 1][8 + i] = XOR(
					_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][16 + i - 16], _mm256_set_epi8(31, 0xff, 0xff, 30, 27, 0xff, 0xff, 26, 23, 0xff, 0xff, 22, 19, 0xff, 0xff, 18, 15, 0xff, 0xff, 14, 11, 0xff, 0xff, 10, 7, 0xff, 0xff, 6, 3, 0xff, 0xff, 2)),
					_mm256_shuffle_epi8(tk2[((r >> 1) + 0) & 1][24 + i - 16], _mm256_set_epi8(0xff, 28, 30, 0xff, 0xff, 24, 26, 0xff, 0xff, 20, 22, 0xff, 0xff, 16, 18, 0xff, 0xff, 12, 14, 0xff, 0xff, 8, 10, 0xff, 0xff, 4, 6, 0xff, 0xff, 0, 2, 0xff)));
			}

			tmp2LFSR = XOR(tk2[((r >> 1) + 1) & 1][5], tk2[((r >> 1) + 1) & 1][7]);
			tk2[((r >> 1) + 1) & 1][7] = tk2[((r >> 1) + 1) & 1][6];
			tk2[((r >> 1) + 1) & 1][6] = tk2[((r >> 1) + 1) & 1][5];
			tk2[((r >> 1) + 1) & 1][5] = tk2[((r >> 1) + 1) & 1][4];
			tk2[((r >> 1) + 1) & 1][4] = tk2[((r >> 1) + 1) & 1][3];
			tk2[((r >> 1) + 1) & 1][3] = tk2[((r >> 1) + 1) & 1][2];
			tk2[((r >> 1) + 1) & 1][2] = tk2[((r >> 1) + 1) & 1][1];
			tk2[((r >> 1) + 1) & 1][1] = tk2[((r >> 1) + 1) & 1][0];
			tk2[((r >> 1) + 1) & 1][0] = tmp2LFSR;

			tmp2LFSR = XOR(tk2[((r >> 1) + 1) & 1][5 + 8], tk2[((r >> 1) + 1) & 1][7 + 8]);
			tk2[((r >> 1) + 1) & 1][7 + 8] = tk2[((r >> 1) + 1) & 1][6 + 8];
			tk2[((r >> 1) + 1) & 1][6 + 8] = tk2[((r >> 1) + 1) & 1][5 + 8];
			tk2[((r >> 1) + 1) & 1][5 + 8] = tk2[((r >> 1) + 1) & 1][4 + 8];
			tk2[((r >> 1) + 1) & 1][4 + 8] = tk2[((r >> 1) + 1) & 1][3 + 8];
			tk2[((r >> 1) + 1) & 1][3 + 8] = tk2[((r >> 1) + 1) & 1][2 + 8];
			tk2[((r >> 1) + 1) & 1][2 + 8] = tk2[((r >> 1) + 1) & 1][1 + 8];
			tk2[((r >> 1) + 1) & 1][1 + 8] = tk2[((r >> 1) + 1) & 1][0 + 8];
			tk2[((r >> 1) + 1) & 1][0 + 8] = tmp2LFSR;
		}
		r++;
	}
}

inline void encrypt_64blocks(u256 x[32], u256 k1[32], u256 srk[ROUND_NUMBER][16])
{
#define shfmask1 _mm256_set_epi8(0xff, 28, 0xff, 29, 0xff, 24, 0xff, 25, 0xff, 20, 0xff, 21, 0xff, 16, 0xff, 17, 0xff, 12, 0xff, 13, 0xff, 8, 0xff, 9, 0xff, 4, 0xff, 5, 0xff, 0, 0xff, 1)
#define shfmask2 _mm256_set_epi8(29, 0xff, 31, 0xff, 25, 0xff, 27, 0xff, 21, 0xff, 23, 0xff, 17, 0xff, 19, 0xff, 13, 0xff, 15, 0xff, 9, 0xff, 11, 0xff, 5, 0xff, 7, 0xff, 1, 0xff, 3, 0xff)
#define shfmask3 _mm256_set_epi8(31, 0xff, 0xff, 30, 27, 0xff, 0xff, 26, 23, 0xff, 0xff, 22, 19, 0xff, 0xff, 18, 15, 0xff, 0xff, 14, 11, 0xff, 0xff, 10, 7, 0xff, 0xff, 6, 3, 0xff, 0xff, 2)
#define shfmask4 _mm256_set_epi8(0xff, 28, 30, 0xff, 0xff, 24, 26, 0xff, 0xff, 20, 22, 0xff, 0xff, 16, 18, 0xff, 0xff, 12, 14, 0xff, 0xff, 8, 10, 0xff, 0xff, 4, 6, 0xff, 0xff, 0, 2, 0xff)

	int r, i, j, lo, hi;
	u256 rc, tmp[8];
	u256 tk1[2][32];
	u256 tk1_16[16][16];

	rc = _mm256_set_epi64x(0x000000FF000000FFull,
		0x000000FF000000FFull,
		0x000000FF000000FFull,
		0x000000FF000000FFull);

	pack_message(x);

	for (j = 0; j < 32; j++)
	{
		tk1[0][j] = k1[j];
	}

	pack_key(tk1[0]);

	// key schedule
	r = 0;
	while (r < 16)
	{
		lo = ((r >> 1) + 0) & 1;
		hi = ((r >> 1) + 1) & 1;

		tk1_16[r][ 0] = tk1[lo][ 0];
		tk1_16[r][ 1] = tk1[lo][ 1];
		tk1_16[r][ 2] = tk1[lo][ 2];
		tk1_16[r][ 3] = tk1[lo][ 3];
		tk1_16[r][ 4] = tk1[lo][ 4];
		tk1_16[r][ 5] = tk1[lo][ 5];
		tk1_16[r][ 6] = tk1[lo][ 6];
		tk1_16[r][ 7] = tk1[lo][ 7];
		tk1_16[r][ 8] = tk1[lo][ 8];
		tk1_16[r][ 9] = tk1[lo][ 9];
		tk1_16[r][10] = tk1[lo][10];
		tk1_16[r][11] = tk1[lo][11];
		tk1_16[r][12] = tk1[lo][12];
		tk1_16[r][13] = tk1[lo][13];
		tk1_16[r][14] = tk1[lo][14];
		tk1_16[r][15] = tk1[lo][15];

		//Apply bit permutation
		for (i = 0; i < 8; i++)
		{
			tk1[hi][16 + i] = XOR(
				_mm256_shuffle_epi8(tk1[lo][16 + i], shfmask1),
				_mm256_shuffle_epi8(tk1[lo][24 + i], shfmask2));
			tk1[hi][24 + i] = XOR(
				_mm256_shuffle_epi8(tk1[lo][16 + i], shfmask3),
				_mm256_shuffle_epi8(tk1[lo][24 + i], shfmask4));
		}

		r++;

		tk1_16[r][ 0] = tk1[hi][16 +  0];
		tk1_16[r][ 1] = tk1[hi][16 +  1];
		tk1_16[r][ 2] = tk1[hi][16 +  2];
		tk1_16[r][ 3] = tk1[hi][16 +  3];
		tk1_16[r][ 4] = tk1[hi][16 +  4];
		tk1_16[r][ 5] = tk1[hi][16 +  5];
		tk1_16[r][ 6] = tk1[hi][16 +  6];
		tk1_16[r][ 7] = tk1[hi][16 +  7];
		tk1_16[r][ 8] = tk1[hi][16 +  8];
		tk1_16[r][ 9] = tk1[hi][16 +  9];
		tk1_16[r][10] = tk1[hi][16 + 10];
		tk1_16[r][11] = tk1[hi][16 + 11];
		tk1_16[r][12] = tk1[hi][16 + 12];
		tk1_16[r][13] = tk1[hi][16 + 13];
		tk1_16[r][14] = tk1[hi][16 + 14];
		tk1_16[r][15] = tk1[hi][16 + 15];

		//Apply bit permutation
		for (i = 0; i < 8; i++)
		{
			tk1[hi][0 + i] = XOR(
				_mm256_shuffle_epi8(tk1[lo][0 + i], shfmask1),
				_mm256_shuffle_epi8(tk1[lo][8 + i], shfmask2));
			tk1[hi][8 + i] = XOR(
				_mm256_shuffle_epi8(tk1[lo][0 + i], shfmask3),
				_mm256_shuffle_epi8(tk1[lo][8 + i], shfmask4));
		}

		r++;
	}


	// encrypt
	for (r = 0; r < ROUND_NUMBER; r++)
	{
		//SubBytes
		for (j = 0; j < 4; j++) {
			tmp[7] = XOR(x[2 + 8 * j], NOR(XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j])), XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j]))));
			tmp[6] = XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j]));
			tmp[5] = XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j]));
			tmp[4] = XOR(x[4 + 8 * j], NOR(XOR(x[2 + 8 * j], NOR(XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j])), XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j])))), XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j]))));
			tmp[3] = XOR(x[6 + 8 * j], NOR(XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j])), x[4 + 8 * j]));
			tmp[2] = XOR(x[1 + 8 * j], NOR(x[5 + 8 * j], x[6 + 8 * j]));
			tmp[1] = XOR(x[0 + 8 * j], NOR(XOR(x[1 + 8 * j], NOR(x[5 + 8 * j], x[6 + 8 * j])), XOR(x[2 + 8 * j], NOR(XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j])), XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j]))))));
			tmp[0] = XOR(x[5 + 8 * j], NOR(XOR(x[6 + 8 * j], NOR(XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j])), x[4 + 8 * j])), XOR(x[0 + 8 * j], NOR(XOR(x[1 + 8 * j], NOR(x[5 + 8 * j], x[6 + 8 * j])), XOR(x[2 + 8 * j], NOR(XOR(x[3 + 8 * j], NOR(x[0 + 8 * j], x[1 + 8 * j])), XOR(x[7 + 8 * j], NOR(x[4 + 8 * j], x[5 + 8 * j]))))))));

			x[0 + 8 * j] = tmp[7];
			x[1 + 8 * j] = tmp[6];
			x[2 + 8 * j] = tmp[5];
			x[3 + 8 * j] = tmp[4];
			x[4 + 8 * j] = tmp[3];
			x[5 + 8 * j] = tmp[2];
			x[6 + 8 * j] = tmp[1];
			x[7 + 8 * j] = tmp[0];
		}

		//AddConstant
		//This only adds c2. The other constants are added with the key
		x[22] = XOR(x[22], rc);

		//AddKey
		x[ 0] = XOR(x[ 0], XOR(tk1_16[r & 0xf][ 0], srk[r][ 0]));
		x[ 1] = XOR(x[ 1], XOR(tk1_16[r & 0xf][ 1], srk[r][ 1]));
		x[ 2] = XOR(x[ 2], XOR(tk1_16[r & 0xf][ 2], srk[r][ 2]));
		x[ 3] = XOR(x[ 3], XOR(tk1_16[r & 0xf][ 3], srk[r][ 3]));
		x[ 4] = XOR(x[ 4], XOR(tk1_16[r & 0xf][ 4], srk[r][ 4]));
		x[ 5] = XOR(x[ 5], XOR(tk1_16[r & 0xf][ 5], srk[r][ 5]));
		x[ 6] = XOR(x[ 6], XOR(tk1_16[r & 0xf][ 6], srk[r][ 6]));
		x[ 7] = XOR(x[ 7], XOR(tk1_16[r & 0xf][ 7], srk[r][ 7]));
		x[ 8] = XOR(x[ 8], XOR(tk1_16[r & 0xf][ 8], srk[r][ 8]));
		x[ 9] = XOR(x[ 9], XOR(tk1_16[r & 0xf][ 9], srk[r][ 9]));
		x[10] = XOR(x[10], XOR(tk1_16[r & 0xf][10], srk[r][10]));
		x[11] = XOR(x[11], XOR(tk1_16[r & 0xf][11], srk[r][11]));
		x[12] = XOR(x[12], XOR(tk1_16[r & 0xf][12], srk[r][12]));
		x[13] = XOR(x[13], XOR(tk1_16[r & 0xf][13], srk[r][13]));
		x[14] = XOR(x[14], XOR(tk1_16[r & 0xf][14], srk[r][14]));
		x[15] = XOR(x[15], XOR(tk1_16[r & 0xf][15], srk[r][15]));

		//ShiftRows
		x[8] = SR1(x[8]); x[16] = SR2(x[16]); x[24] = SR3(x[24]);
		x[9] = SR1(x[9]); x[17] = SR2(x[17]); x[25] = SR3(x[25]);
		x[10] = SR1(x[10]); x[18] = SR2(x[18]); x[26] = SR3(x[26]);
		x[11] = SR1(x[11]); x[19] = SR2(x[19]); x[27] = SR3(x[27]);
		x[12] = SR1(x[12]); x[20] = SR2(x[20]); x[28] = SR3(x[28]);
		x[13] = SR1(x[13]); x[21] = SR2(x[21]); x[29] = SR3(x[29]);
		x[14] = SR1(x[14]); x[22] = SR2(x[22]); x[30] = SR3(x[30]);
		x[15] = SR1(x[15]); x[23] = SR2(x[23]); x[31] = SR3(x[31]);

		//MixColumns
		tmp[0] = x[24]; tmp[1] = x[25]; tmp[2] = x[26]; tmp[3] = x[27];
		tmp[4] = x[28]; tmp[5] = x[29]; tmp[6] = x[30]; tmp[7] = x[31];

		x[24] = XOR(x[16], x[0]); x[28] = XOR(x[20], x[4]);
		x[25] = XOR(x[17], x[1]); x[29] = XOR(x[21], x[5]);
		x[26] = XOR(x[18], x[2]); x[30] = XOR(x[22], x[6]);
		x[27] = XOR(x[19], x[3]); x[31] = XOR(x[23], x[7]);

		x[16] = XOR(x[8], x[16]); x[20] = XOR(x[12], x[20]);
		x[17] = XOR(x[9], x[17]); x[21] = XOR(x[13], x[21]);
		x[18] = XOR(x[10], x[18]); x[22] = XOR(x[14], x[22]);
		x[19] = XOR(x[11], x[19]); x[23] = XOR(x[15], x[23]);

		x[8] = x[0]; x[12] = x[4];
		x[9] = x[1]; x[13] = x[5];
		x[10] = x[2]; x[14] = x[6];
		x[11] = x[3]; x[15] = x[7];


		x[0] = XOR(tmp[0], x[24]); x[4] = XOR(tmp[4], x[28]);
		x[1] = XOR(tmp[1], x[25]); x[5] = XOR(tmp[5], x[29]);
		x[2] = XOR(tmp[2], x[26]); x[6] = XOR(tmp[6], x[30]);
		x[3] = XOR(tmp[3], x[27]); x[7] = XOR(tmp[7], x[31]);
	}

	unpack_message(x);
}


#define SC(X)                                   \
{												\
	for (int i = 0; i < BLOCK_BYTE_NUMBER; i++) \
	{											\
		X[i] = S8[X[i]];						\
	}                                           \
}

#define AC(X)	   \
{				   \
	X[8] ^= 0x2;   \
}

#define ART(X, K)	                                 \
{				                                     \
    *((u64 *)X) ^= K;                                \
}

#define SR(X)								  \
{											  \
	SRtmp = _mm_loadu_si128((__m128i *)X);    \
	SRtmp = _mm_shuffle_epi8(SRtmp, SRc);	  \
	_mm_storeu_si128((__m128i *)X, SRtmp);    \
}

#define MC(X)								 \
{											 \
	Row[0] = ((u32 *)X)[0];					 \
	Row[1] = ((u32 *)X)[1];					 \
	Row[2] = ((u32 *)X)[2];					 \
	Row[3] = ((u32 *)X)[3];					 \
	((u32 *)X)[1] = Row[0];					 \
	((u32 *)X)[2] = Row[1] ^ Row[2];		 \
	((u32 *)X)[3] = Row[0] ^ Row[2];		 \
	((u32 *)X)[0] = ((u32 *)X)[3] ^ Row[3];  \
}

#define Round(X, K) \
{					\
	SC(X);			\
	AC(X);			\
	ART(X, K);		\
	SR(X);			\
	MC(X);			\
}

inline void encrypt_1block(__m128i *y, __m128i x, const __m128i k1, const __m128i k2)
{
	/* SKINNY-128 Sbox */
	unsigned char S8[256] = {
		0x65,0x4c,0x6a,0x42,0x4b,0x63,0x43,0x6b,0x55,0x75,0x5a,0x7a,0x53,0x73,0x5b,0x7b,
		0x35,0x8c,0x3a,0x81,0x89,0x33,0x80,0x3b,0x95,0x25,0x98,0x2a,0x90,0x23,0x99,0x2b,
		0xe5,0xcc,0xe8,0xc1,0xc9,0xe0,0xc0,0xe9,0xd5,0xf5,0xd8,0xf8,0xd0,0xf0,0xd9,0xf9,
		0xa5,0x1c,0xa8,0x12,0x1b,0xa0,0x13,0xa9,0x05,0xb5,0x0a,0xb8,0x03,0xb0,0x0b,0xb9,
		0x32,0x88,0x3c,0x85,0x8d,0x34,0x84,0x3d,0x91,0x22,0x9c,0x2c,0x94,0x24,0x9d,0x2d,
		0x62,0x4a,0x6c,0x45,0x4d,0x64,0x44,0x6d,0x52,0x72,0x5c,0x7c,0x54,0x74,0x5d,0x7d,
		0xa1,0x1a,0xac,0x15,0x1d,0xa4,0x14,0xad,0x02,0xb1,0x0c,0xbc,0x04,0xb4,0x0d,0xbd,
		0xe1,0xc8,0xec,0xc5,0xcd,0xe4,0xc4,0xed,0xd1,0xf1,0xdc,0xfc,0xd4,0xf4,0xdd,0xfd,
		0x36,0x8e,0x38,0x82,0x8b,0x30,0x83,0x39,0x96,0x26,0x9a,0x28,0x93,0x20,0x9b,0x29,
		0x66,0x4e,0x68,0x41,0x49,0x60,0x40,0x69,0x56,0x76,0x58,0x78,0x50,0x70,0x59,0x79,
		0xa6,0x1e,0xaa,0x11,0x19,0xa3,0x10,0xab,0x06,0xb6,0x08,0xba,0x00,0xb3,0x09,0xbb,
		0xe6,0xce,0xea,0xc2,0xcb,0xe3,0xc3,0xeb,0xd6,0xf6,0xda,0xfa,0xd3,0xf3,0xdb,0xfb,
		0x31,0x8a,0x3e,0x86,0x8f,0x37,0x87,0x3f,0x92,0x21,0x9e,0x2e,0x97,0x27,0x9f,0x2f,
		0x61,0x48,0x6e,0x46,0x4f,0x67,0x47,0x6f,0x51,0x71,0x5e,0x7e,0x57,0x77,0x5f,0x7f,
		0xa2,0x18,0xae,0x16,0x1f,0xa7,0x17,0xaf,0x01,0xb2,0x0e,0xbe,0x07,0xb7,0x0f,0xbf,
		0xe2,0xca,0xee,0xc6,0xcf,0xe7,0xc7,0xef,0xd2,0xf2,0xde,0xfe,0xd7,0xf7,0xdf,0xff
	};

	__m128i SRc = _mm_set_epi8(12, 15, 14, 13, 9, 8, 11, 10, 6, 5, 4, 7, 3, 2, 1, 0);
	__m128i SRtmp;
	u32 Row[4];
	ALIGNED_TYPE_(u8, 16) State[BLOCK_BYTE_NUMBER];
	__m128i * Stateip = (__m128i *) State;
	_mm_storeu_si128(Stateip, x);

	__m128i TK1tmp = k1;
	__m128i TK2tmp = k2;
	__m128i PT = _mm_set_epi8(7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 14, 10, 13, 8, 15, 9);
	u64 TK1half = _mm_extract_epi64(TK1tmp, 0);
	u64 TK2half = _mm_extract_epi64(TK2tmp, 0);
	u64 rc01;
	u64 roundkey;

	for (int i = 0; i < ROUND_NUMBER; i++)
	{
		rc01 = ((u64)(RC[i] & 0xf)) | ((u64)((RC[i] & 0x30) >> 4));
		roundkey = TK1half ^ TK2half ^ rc01;
		TK1tmp = _mm_shuffle_epi8(TK1tmp, PT);
		TK2tmp = _mm_shuffle_epi8(TK2tmp, PT);
		TK1half = _mm_extract_epi64(TK1tmp, 0);
		TK2half = _mm_extract_epi64(TK2tmp, 0);
		TK2half = (TK2half << 8) | ((TK2half >> (5 * 8)) ^ ((TK2half >> (7 * 8)))) & 0xf;

		Round(State, roundkey);
	}
	_mm_storeu_si128(y, *Stateip);
}

#endif