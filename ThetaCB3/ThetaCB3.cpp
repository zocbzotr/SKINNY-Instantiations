#include <iostream>
#include <iomanip>
#include <fstream>
#include "ThetaCB3.h"
#include "Skinny128256AVX2.h"
#include "assist.h"

using namespace std;

void ThetaCB3_enc(
	unsigned char * C,
	unsigned char * T,
	unsigned char * N,
	unsigned char * A,
	unsigned char * M,
	long long AByteN,
	long long MByteN,
	unsigned char *Seedkey
)
{
	const __m128i allzero = _mm_setzero_si128();
	const __m256i allzero256 = _mm256_setzero_si256();
	__m128i Checksum = allzero;
	__m256i Checksum256 = allzero256;

	__m128i K128i = _mm_loadu_si128((__m128i *)Seedkey);
	__m256i SRK264block[ROUND_NUMBER][16];

	__m128i *C128ip = (__m128i *)(C);
	__m128i *T128ip = (__m128i *)(T);
	__m128i *M128ip = (__m128i *)(M);

	__m256i *C256ip = (__m256i *)(C);
	__m256i *T256ip = (__m256i *)(T);
	__m256i *M256ip = (__m256i *)(M);

	__m128i M128i;
	__m128i T128i;

	__m128i Final;
	__m128i Pad;

	__m256i M64block[32];
	__m256i T64block[32];

	u8 * Cip = C;
	u8 * Mip = M;
	u64 Nonce = *(u64 *)N;

	s64 MBlockN = MByteN / BLOCK_BYTE_NUMBER;
	s64 MRe = MByteN % BLOCK_BYTE_NUMBER;

	if (MBlockN >= PIPE)
	{
		for (int i = 0; i < 32; i++)
		{
			M64block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);
		}
		key_schedule(M64block, SRK264block);
	}

	for (s64 mi = 0; (mi + PIPE) <= MBlockN; mi += PIPE)
	{
		for (int i = 0; i < (PIPE >> 1); i++)
		{
			T64block[i] = _mm256_set_epi64x(Nonce, mi + 2 * i + 1, Nonce, mi + 2 * i + 0);
			M256ip = (__m256i *)(Mip + i * 2 * BLOCK_BYTE_NUMBER);
			M64block[i] = _mm256_loadu_si256(M256ip);
			Checksum256 = _mm256_xor_si256(Checksum256, M64block[i]);
		}

		encrypt_64blocks(M64block, T64block, SRK264block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			C256ip = (__m256i *)(Cip + i * 2 * BLOCK_BYTE_NUMBER);
			_mm256_storeu_si256(C256ip, M64block[i]);
		}

		Mip += (BLOCK_BYTE_NUMBER * PIPE);
		Cip += (BLOCK_BYTE_NUMBER * PIPE);
	}

	if (MBlockN >= PIPE)
	{
		Checksum = _mm_xor_si128(_mm256_extracti128_si256(Checksum256, 0), _mm256_extracti128_si256(Checksum256, 1));
	}

	for (s64 mi = 0; mi < (MBlockN % PIPE); mi++)
	{
		u64 cnt = (MBlockN / PIPE) * PIPE + mi;
		T128i = _mm_set_epi64x(Nonce, cnt);
		M128ip = (__m128i *)(Mip);
		M128i = _mm_loadu_si128(M128ip);
		Checksum = _mm_xor_si128(Checksum, M128i);

		encrypt_1block(&M128i, M128i, T128i, K128i);

		C128ip = (__m128i *)(Cip);
		_mm_storeu_si128(C128ip, M128i);

		Mip += BLOCK_BYTE_NUMBER;
		Cip += BLOCK_BYTE_NUMBER;
	}

	if (MRe == 0)
	{
		T128i = _mm_set_epi64x(Nonce, MBlockN + 2);
		encrypt_1block(&Final, Checksum, T128i, K128i);
	} 
	else
	{
		T128i = _mm_set_epi64x(Nonce, MBlockN + 1);
		encrypt_1block(&Pad, allzero, T128i, K128i);

		M128i = _mm_loadu_si128((__m128i *)(Mip));
		u8 * MRep = (u8 *)(&M128i);
		ozpInplace(MRep, MRe, BLOCK_BYTE_NUMBER);
		Checksum = _mm_xor_si128(Checksum, M128i);

		M128i = _mm_xor_si128(M128i, Pad);
		memcpy(Cip, MRep, MRe);

		T128i = _mm_set_epi64x(Nonce, MBlockN + 3);
		encrypt_1block(&Final, Checksum, T128i, K128i);
	}

	__m128i Tag;
	Hash_enc((u8 *)(&Tag), A, AByteN, Seedkey);
	Tag = _mm_xor_si128(Tag, Final);
	memcpy(T, (u8 *)(&Tag), TAG_BYTE_NUMBER);
};

void Hash_enc(
	unsigned char * Y,
	unsigned char * A,
	long long AByteN,
	unsigned char *Seedkey
)
{
	const __m128i allzero = _mm_setzero_si128();
	const __m256i allzero256 = _mm256_setzero_si256();
	__m128i K128i = _mm_loadu_si128((__m128i *)Seedkey);
	__m256i SRK264block[ROUND_NUMBER][16];

	__m128i A128i;
	__m128i T128i;

	__m256i A64block[32];
	__m256i T64block[32];

	__m128i *A128ip = (__m128i *)(A);

	__m256i *A256ip = (__m256i *)(A);

	__m128i Sum = allzero;
	__m256i Sum256 = allzero256;
	u8 * Aip = A;

	s64 ABlockN = AByteN / BLOCK_BYTE_NUMBER;
	s64 ARe = AByteN % BLOCK_BYTE_NUMBER;

	if (ABlockN >= PIPE)
	{
		for (int i = 0; i < 32; i++)
		{
			A64block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);
		}
		key_schedule(A64block, SRK264block);
	}

	for (s64 mi = 0; mi + PIPE <= ABlockN; mi += PIPE)
	{
		for (int i = 0; i < (PIPE >> 1); i++)
		{
			T64block[i] = _mm256_set_epi64x(0x0LL, mi + 2 * i + 1, 0x0LL, mi + 2 * i + 0);
			A256ip = (__m256i *)(Aip + i * 2 * BLOCK_BYTE_NUMBER);
			A64block[i] = _mm256_loadu_si256(A256ip);
		}

		encrypt_64blocks(A64block, T64block, SRK264block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			Sum256 = _mm256_xor_si256(Sum256, A64block[i]);
		}

		Aip += (BLOCK_BYTE_NUMBER * PIPE);
	}

	if (ABlockN >= PIPE)
	{
		Sum = _mm_xor_si128(_mm256_extracti128_si256(Sum256, 0), _mm256_extracti128_si256(Sum256, 1));
	}

	for (s64 mi = 0; mi < ABlockN % PIPE; mi++)
	{
		u64 cnt = (ABlockN / PIPE) * PIPE + mi;
		T128i = _mm_set_epi64x(0x0LL, cnt);
		A128ip = (__m128i *)(Aip);
		A128i = _mm_loadu_si128(A128ip);
		encrypt_1block(&A128i, A128i, T128i, K128i);
		Sum = _mm_xor_si128(Sum, A128i);

		Aip += BLOCK_BYTE_NUMBER;
	}

	if (ARe != 0)
	{
		T128i = _mm_set_epi64x(0x0LL, ABlockN + 1);
		A128ip = (__m128i *)(Aip);
		A128i = _mm_loadu_si128(A128ip);
		u8 * ARep = (u8 *)(&A128i);
		ozpInplace(ARep, ARe, BLOCK_BYTE_NUMBER);
		encrypt_1block(&A128i, A128i, T128i, K128i);
		Sum = _mm_xor_si128(Sum, A128i);
	}
	_mm_store_si128((__m128i *)Y, Sum);
};