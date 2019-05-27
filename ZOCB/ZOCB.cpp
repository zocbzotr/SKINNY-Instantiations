#include <iostream>
#include <iomanip>
#include <fstream>
#include "ZOCB.h"
#include "Skinny128256AVX2.h"
#include "assist.h"

using namespace std;

void ZOCB_enc(
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
	s64 MBlockN = (MByteN +  BLOCK_BYTE_NUMBER - 1) / BLOCK_BYTE_NUMBER;
	if (MByteN == 0) MBlockN = 1LL;
	u64 BByteN = MBlockN * TRICK_BYTE_NUMBER;

	u8 *B = new u8[BByteN];

	__m128i Y;
	__m128i Yh;
	__m128i Temp;

	if (AByteN < BByteN)
	{
		memcpy(B, A, AByteN);
		ozpInplace(B, AByteN, BByteN);
		Yh = _mm_setzero_si128();
	}
	else
	{
		memcpy(B, A, BByteN);
		s64 BhByteN = AByteN - BByteN;
		u8 *Bh;
		if (BhByteN == 0)
		{
			Bh = new u8[BLOCK_TRICK_BYTE_NUMBER];
		}
		else
		{
			Bh = new u8[BhByteN];
		}
		memcpy(Bh, A + BByteN, BhByteN);
		Hash_enc((u8 *)&Yh, Bh, BhByteN, Seedkey);
		delete[] Bh;
	}

	Core_enc(C, (u8 *)&Y, N, B, M, MByteN, Seedkey);
	Temp = _mm_xor_si128(Y, Yh);
	_mm_storeu_si128((__m128i *)T, Temp);

	delete[] B;
};

void Core_enc(
	unsigned char * C,
	unsigned char * Y,
	unsigned char * N,
	unsigned char * B,
	unsigned char * M,
	long long MByteN,
	unsigned char *Seedkey
)
{
	const __m128i allzero = _mm_setzero_si128();
	const __m256i allzero256 = _mm256_setzero_si256();
	__m128i S = allzero;
	__m128i N128i = _mm_loadu_si128((__m128i *)N);
	__m128i K128i = _mm_loadu_si128((__m128i *)Seedkey);

	__m128i M128i;
	__m256i M256i;
	__m256i B256i;
	__m128i B128i1;
	__m128i B128i2;
	__m128i TK1128i;

	__m128i *C128ip = (__m128i *)(C);
	__m128i *Y128ip = (__m128i *)(Y);
	__m128i *M128ip = (__m128i *)(M);
	__m128i *B128ip = (__m128i *)(B);
	__m256i *C256ip = (__m256i *)(C);
	__m256i *Y256ip = (__m256i *)(Y);
	__m256i *M256ip = (__m256i *)(M);
	__m256i *B256ip = (__m256i *)(B);

	u8 * Cip = C;
	u8 * Mip = M;
	u8 * Bip = B;

	__m128i pad;

	__m256i M64block[32];
	__m256i S64block = allzero256;

	__m128i alpha;
	__m128i beta;
	__m256i alpha64block[32];
	__m256i beta64block[32];
	__m256i alphabeta64block[64 + 1];

	__m256i TK164block[32];
#define TK264block TK164block
	__m256i SRK264block[ROUND_NUMBER][16];

	if (MByteN == 0)
	{
		TK1128i = _mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000003ULL);
		encrypt_1block(&alpha, N128i, TK1128i, K128i);
		TK1128i = _mm_set_epi64x(0x0000000000000001ULL, 0x0000000000000003ULL);
		encrypt_1block(&beta, N128i, TK1128i, K128i);

		//ozpInplace(((u8 *)S), 0, BLOCK_BYTE_NUMBER);
		S = _mm_insert_epi8(S, 0x80, 0);
		B128ip = (__m128i *)(B);
		B128i1 = _mm_loadu_si128(B128ip);
		TK1128i = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);
		TK1128i = _mm_insert_epi8(TK1128i, 0x01, 0);
		S = _mm_xor_si128(S, alpha);
		encrypt_1block(Y128ip, S, TK1128i, K128i);
		return;
	}

	TK1128i = _mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000003ULL);
	encrypt_1block(&alpha, N128i, TK1128i, K128i);
	TK1128i = _mm_set_epi64x(0x0000000000000001ULL, 0x0000000000000003ULL);
	encrypt_1block(&beta, N128i, TK1128i, K128i);
	alphabeta64block[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(alpha), beta, 0x1);

	s64 MBlockN = MByteN / BLOCK_BYTE_NUMBER;
	s64 MRe = MByteN % BLOCK_BYTE_NUMBER;
	s64 MBlockN_N_1;
	if (MRe == 0)
	{
		MBlockN_N_1 = MBlockN - 1;
		MRe = BLOCK_BYTE_NUMBER;
	}
	else
	{
		MBlockN_N_1 = MBlockN;
	}

	s64 ResBlocks = MBlockN_N_1;

	if (MBlockN_N_1 >= PIPE)
	{
		for (int i = 0; i < 32; i++)
		{
			TK264block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);
		}
		key_schedule(TK264block, SRK264block);
	}

	while (ResBlocks >= PIPE)
	{
		mul2_PIPE_256(alphabeta64block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			alpha64block[i] = _mm256_permute2x128_si256(alphabeta64block[2 * i + 0], alphabeta64block[2 * i + 1], 0x20);
			beta64block[i] = _mm256_permute2x128_si256(alphabeta64block[2 * i + 0], alphabeta64block[2 * i + 1], 0x31);
			M256ip = (__m256i *)(Mip + i * 2 * BLOCK_BYTE_NUMBER);
			M256i = _mm256_loadu_si256(M256ip);
			M64block[i] = M256i;
			S64block = _mm256_xor_si256(S64block, M256i);
			M64block[i] = _mm256_xor_si256(M64block[i], alpha64block[i]);

			B128ip = (__m128i *)(Bip + i * 2 * TRICK_BYTE_NUMBER);
			B128i1 = _mm_loadu_si128(B128ip);
			B128ip = (__m128i *)(Bip + (i * 2 + 1) * TRICK_BYTE_NUMBER);
			B128i2 = _mm_loadu_si128(B128ip);
			B256i = _mm256_inserti128_si256(_mm256_castsi128_si256(B128i1), B128i2, 0x1);  //_mm256_set_m128i(B128i2, B128i1);
			B256i = _mm256_slli_si256(_mm256_xor_si256(B256i, beta64block[i]), 1);
			TK164block[i] = B256i;
		}

		encrypt_64blocks(M64block, TK164block, SRK264block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			M64block[i] = _mm256_xor_si256(M64block[i], alpha64block[i]);
			C256ip = (__m256i *)(Cip + i * 2 * BLOCK_BYTE_NUMBER);
			_mm256_storeu_si256(C256ip, M64block[i]);
		}

		alphabeta64block[0] = alphabeta64block[PIPE];

		Mip += PIPE * (BLOCK_BYTE_NUMBER);
		Bip += PIPE * (TRICK_BYTE_NUMBER);
		Cip += PIPE * (BLOCK_BYTE_NUMBER);
		ResBlocks -= PIPE;
	}

	if (MBlockN_N_1 >= PIPE)
	{
		S = _mm_xor_si128(_mm256_extracti128_si256(S64block, 0), _mm256_extracti128_si256(S64block, 1));
	}

	while (ResBlocks > 0)
	{
		mul2_256(&(alphabeta64block[0]), &(alphabeta64block[1]));

		alpha = _mm256_extracti128_si256(alphabeta64block[0], 0);
		beta = _mm256_extracti128_si256(alphabeta64block[0], 1);

		M128ip = (__m128i *)(Mip);
		M128i = _mm_loadu_si128(M128ip);
		S = _mm_xor_si128(S, M128i);
		M128i = _mm_xor_si128(M128i, alpha);

		B128ip = (__m128i *)(Bip);
		B128i1 = _mm_loadu_si128(B128ip);
		B128i1 = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);

		encrypt_1block(&M128i, M128i, B128i1, K128i);

		M128i = _mm_xor_si128(M128i, alpha);
		C128ip = (__m128i *)(Cip);
		_mm_storeu_si128(C128ip, M128i);

		alphabeta64block[0] = alphabeta64block[1];

		Mip += 1 * (BLOCK_BYTE_NUMBER);
		Bip += 1 * (TRICK_BYTE_NUMBER);
		Cip += 1 * (BLOCK_BYTE_NUMBER);
		ResBlocks -= 1;
	}

	alpha = _mm256_extracti128_si256(alphabeta64block[0], 0);
	beta = _mm256_extracti128_si256(alphabeta64block[0], 1);

	TK1128i = _mm_slli_si128(beta, 1);
	encrypt_1block(&pad, alpha, TK1128i, K128i);
	pad = _mm_xor_si128(pad, alpha);

	if (MRe != BLOCK_BYTE_NUMBER)
	{
		ozp(MRe, M + MBlockN_N_1 * BLOCK_BYTE_NUMBER, &M128i);
	}
	else
	{
		M128i = _mm_loadu_si128(((__m128i *)M) + MBlockN_N_1);
	}
	S = _mm_xor_si128(S, M128i);
	M128i = _mm_xor_si128(pad, M128i);
	memcpy((u8 *)(C + MBlockN_N_1 * BLOCK_BYTE_NUMBER), (u8*)(&M128i), MRe);

	B128ip = (__m128i *)(B + MBlockN_N_1 * TRICK_BYTE_NUMBER);
	B128i1 = _mm_loadu_si128(B128ip);
	TK1128i = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);
	if (MRe != BLOCK_BYTE_NUMBER)
	{
		TK1128i = _mm_insert_epi8(TK1128i, 0x01, 0);
	}
	else
	{
		TK1128i = _mm_insert_epi8(TK1128i, 0x02, 0);
	}
	S = _mm_xor_si128(S, alpha);
	encrypt_1block(Y128ip, S, TK1128i, K128i);
};


void Hash_enc(
	unsigned char * Yh,
	unsigned char * Bh,
	long long BhByteN,
	unsigned char *Seedkey
)
{
	const __m128i allzero = _mm_setzero_si128();
	__m128i K128i = _mm_loadu_si128((__m128i *)Seedkey);

	__m128i Yh128i = allzero;
	__m128i *Yh128ip = (__m128i *)(Yh);
	__m256i *Bh256ip = (__m256i *)(Bh);
	__m128i *Bh128ip = (__m128i *)(Bh);
	u8 * Bhip = Bh;

	__m256i Bh256i1;
	__m256i Bh256i2;

	__m128i P128i;
	__m128i Q128i;
	__m256i P64block[32];
	__m256i Q64block[32];

	__m128i gamma;
	__m128i delta;
	__m256i gamma64block[32];
	__m256i delta64block[32];
	__m256i gammadelta64block[64 + 1];
	__m128i TK1128i;

	__m256i TK64block[32];
	__m256i SRK264block[ROUND_NUMBER][16];

	TK1128i = _mm_set_epi64x(0x0000000000000002ULL, 0x0000000000000003ULL);
	encrypt_1block(&gamma, allzero, TK1128i, K128i);
	TK1128i = _mm_set_epi64x(0x0000000000000003ULL, 0x0000000000000003ULL);
	encrypt_1block(&delta, allzero, TK1128i, K128i);

	s64 BhBlockN;
	s64 BhRe;
	s64 BhBlockN_N_1;
	u8 lastblock[BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER];

	if (BhByteN != 0)
	{
		gammadelta64block[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(gamma), delta, 0x1);  //_mm256_set_m128i(delta, gamma);

		BhBlockN = BhByteN / (BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER);
		BhRe = BhByteN % (BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER);

		if (BhRe == 0)
		{
			BhBlockN_N_1 = BhBlockN - 1;
			BhRe = BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER;
		}
		else
		{
			BhBlockN_N_1 = BhBlockN;
		}

		s64 ResBlocks = BhBlockN_N_1;

		if (BhBlockN_N_1 >= PIPE)
		{
			for (int i = 0; i < 32; i++)
			{
				TK64block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);
			}
			key_schedule(TK64block, SRK264block);
		}

		while (ResBlocks >= PIPE)
		{
			mul2_PIPE_256(gammadelta64block);

			for (int i = 0; i < (PIPE >> 1); i++)
			{
				gamma64block[i] = _mm256_permute2x128_si256(gammadelta64block[2 * i + 0], gammadelta64block[2 * i + 1], 0x20);
				delta64block[i] = _mm256_permute2x128_si256(gammadelta64block[2 * i + 0], gammadelta64block[2 * i + 1], 0x31);
				Bh256ip = (__m256i *)(Bhip + i * 2 * BLOCK_TRICK_BYTE_NUMBER);
				Bh256i1 = _mm256_loadu_si256(Bh256ip);
				Bh256ip = (__m256i *)(Bhip + (i * 2 + 1) * BLOCK_TRICK_BYTE_NUMBER);
				Bh256i2 = _mm256_loadu_si256(Bh256ip);
				P64block[i] = _mm256_permute2x128_si256(Bh256i1, Bh256i2, 0x8);
				Q64block[i] = _mm256_permute2x128_si256(Bh256i1, Bh256i2, 0xd);

				P64block[i] = _mm256_xor_si256(P64block[i], gamma64block[i]);
				Q64block[i] = _mm256_slli_si256(_mm256_xor_si256(Q64block[i], delta64block[i]), 1);
			}
			encrypt_64blocks(P64block, Q64block, SRK264block);

			for (int i = 0; i < (PIPE >> 1); i++)
			{
				Yh128i = _mm_xor_si128(Yh128i, _mm256_extracti128_si256(P64block[i], 0));
				Yh128i = _mm_xor_si128(Yh128i, _mm256_extracti128_si256(P64block[i], 1));
			}

			gammadelta64block[0] = gammadelta64block[PIPE];
			Bhip += PIPE * (BLOCK_TRICK_BYTE_NUMBER);
			ResBlocks -= PIPE;
		}

		while (ResBlocks > 0)
		{
			mul2_256(&(gammadelta64block[0]), &(gammadelta64block[1]));

			gamma = _mm256_extracti128_si256(gammadelta64block[0], 0);
			delta = _mm256_extracti128_si256(gammadelta64block[0], 1);

			Bh128ip = (__m128i *)(Bhip);
			P128i = _mm_loadu_si128(Bh128ip);
			Bh128ip = (__m128i *)(Bhip + BLOCK_BYTE_NUMBER);
			Q128i = _mm_loadu_si128(Bh128ip);

			P128i = _mm_xor_si128(P128i, gamma);
			Q128i = _mm_slli_si128(_mm_xor_si128(Q128i, delta), 1);

			encrypt_1block(&P128i, P128i, Q128i, K128i);

			Yh128i = _mm_xor_si128(Yh128i, P128i);

			gammadelta64block[0] = gammadelta64block[1];

			Bhip += 1 * (BLOCK_TRICK_BYTE_NUMBER);
			ResBlocks -= 1;
		}
		if (BhRe != BLOCK_TRICK_BYTE_NUMBER)
		{
			ozpAD(BhRe, Bh + BhBlockN_N_1 * (BLOCK_TRICK_BYTE_NUMBER), lastblock);
		}
		else
		{
			memcpy(lastblock, Bh + BhBlockN_N_1 * (BLOCK_TRICK_BYTE_NUMBER), BLOCK_TRICK_BYTE_NUMBER);
		}

		gamma = _mm256_extracti128_si256(gammadelta64block[0], 0);
		delta = _mm256_extracti128_si256(gammadelta64block[0], 1);
	}
	else
	{
		lastblock[0] = 0x80;
		for (int i = 0; i < BLOCK_TRICK_BYTE_NUMBER; i++)
		{
			lastblock[i] = 0;
		}
	}

	P128i = _mm_loadu_si128((__m128i *)lastblock);
	Q128i = _mm_loadu_si128((__m128i *)(lastblock + BLOCK_BYTE_NUMBER));

	Q128i = _mm_slli_si128(_mm_xor_si128(Q128i, delta), 1);

	if (BhRe != BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER)
	{
		Q128i = _mm_insert_epi8(Q128i, 0x01, 0);
	}
	else
	{
		Q128i = _mm_insert_epi8(Q128i, 0x02, 0);
	}
	P128i = _mm_xor_si128(P128i, gamma);
	encrypt_1block(&P128i, P128i, Q128i, K128i);
	Yh128i = _mm_xor_si128(Yh128i, P128i);
	_mm_storeu_si128(Yh128ip, Yh128i);
};