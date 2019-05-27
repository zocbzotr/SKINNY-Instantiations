#ifndef ASSIST_H__
#define ASSIST_H__

#include "types.h"
#include "ZOCB.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


#if __GNUC__
#define ALIGN(n) __attribute__ ((aligned(n)))
#elif _MSC_VER
#define ALIGN(n) __declspec(align(n))
#define __inline__ __inline
#else 
#define ALIGN(n)
#endif

#define CONST const

typedef unsigned char uint8;
typedef unsigned int	uint32;
typedef ALIGN(16)__m128i block;

#define BLOCK BLOCK_BYTE_NUMBER

///////////////////////////////////////////////////////////// api.h //////////////////////////////////////////////

#define CRYPTO_KEYBYTES 32
#define CRYPTO_ABYTES 16

#define ROUND   ROUND_NUMBER
#define EK_SZ (ROUND+1) 
#define le(b) _mm_shuffle_epi8(b,_mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)) /*Byte order conversion*/
#define le256(b) _mm256_shuffle_epi8(b,_mm256_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)) /*Byte order conversion*/

/*
** Batch doubling for PIPE blocks
*/
inline static void mul2_PIPE(__m128i *dat) {
	const __m128i mask = _mm_set_epi32(135, 1, 1, 1);
	__m128i intmp = le(dat[0]);
	__m128i tmp;

	for (int i = 1; i <= PIPE; i++)
	{
		tmp = _mm_srai_epi32(intmp, 31);
		tmp = _mm_and_si128(tmp, mask);
		tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 1, 0, 3));
		intmp = _mm_slli_epi32(intmp, 1);
		intmp = _mm_xor_si128(intmp, tmp);
		dat[i] = le(intmp);
	}
}


/*
** single doubling
*/
inline static void mul2(block in, block *out) {
	const block shuf = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const block mask = _mm_set_epi32(135, 1, 1, 1);
	block intmp = _mm_shuffle_epi8(in, shuf);
	block tmp = _mm_srai_epi32(intmp, 31);
	tmp = _mm_and_si128(tmp, mask);
	tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 1, 0, 3));
	*out = _mm_slli_epi32(intmp, 1);
	*out = _mm_xor_si128(*out, tmp);
	*out = _mm_shuffle_epi8(*out, shuf);
}

/*
** Batch doubling for PIPE blocks
*/
inline static void mul2_PIPE_256(__m256i *dat) {
	const __m256i mask = _mm256_set_epi32(135, 1, 1, 1, 135, 1, 1, 1);
	__m256i intmp = le256(*dat);
	__m256i tmp;

	for (int i = 1; i <= PIPE; i++)
	{
		tmp = _mm256_srai_epi32(intmp, 31);
		tmp = _mm256_and_si256(tmp, mask);
		tmp = _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(2, 1, 0, 3));
		intmp = _mm256_slli_epi32(intmp, 1);
		intmp = _mm256_xor_si256(intmp, tmp);
		*dat = le256(intmp);
	}
}


/*
** single doubling
*/
inline static void mul2_256(__m256i* in, __m256i *out) {
	const __m256i shuf = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m256i mask = _mm256_set_epi32(135, 1, 1, 1, 135, 1, 1, 1);
	__m256i intmp = _mm256_shuffle_epi8(*in, shuf);
	__m256i tmp = _mm256_srai_epi32(intmp, 31);
	__m256i tmp2;
	tmp = _mm256_and_si256(tmp, mask);
	tmp = _mm256_shuffle_epi32(tmp, _MM_SHUFFLE(2, 1, 0, 3));
	tmp2 = _mm256_slli_epi32(intmp, 1);
	tmp2 = _mm256_xor_si256(tmp2, tmp);
	tmp2 = _mm256_shuffle_epi8(tmp2, shuf);
	*out = tmp2;
}

/*
ozp: one-zero padding for 16-byte block
*/
inline static void ozp(uint32 length, const uint8 *in, block *out) {
	ALIGN(16)uint8 tmp[BLOCK + 1] = { 0 };
	memcpy(tmp, in, length);
	tmp[length] = 0x80;
	*out = _mm_load_si128((block*)tmp);
}

/*
ozpAD: one-zero padding for (BLOCK + TRICK_BYTE_NUMBER)-byte block
*/
inline static void ozpAD(uint32 length, const uint8 *in, uint8 *out) {
	ALIGN(16)uint8 tmp[BLOCK + TRICK_BYTE_NUMBER + 1] = { 0 };
	memcpy(tmp, in, length);
	tmp[length] = 0x80;
	memcpy(out, tmp, BLOCK + TRICK_BYTE_NUMBER);
}

#define ozpInplace(X, BEG, END)				 \
{										 \
	X[BEG] = 0x80;						 \
	for (u64 i = BEG + 1; i < END; i++)  \
	{									 \
		X[i] = 0x00;					 \
	}									 \
}

#endif // ASSIST_H__
