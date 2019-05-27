#include <iostream>
#include <iomanip>
#include <fstream>
#include "ZOTR.h"
#include "Skinny128256AVX2.h"
#include "assist.h"

using namespace std;

void ZOTR_enc(
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
	const __m128i oneone = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
	const __m256i oneone256 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);

	__m128i S = allzero;
	__m128i N128i = _mm_loadu_si128((__m128i *)N);;
	__m128i K128i = _mm_loadu_si128((__m128i *)Seedkey);

	__m128i M128i;
	__m128i M128i_L;
	__m128i M128i_R;
	__m128i C128i;
	__m128i C128i_L;
	__m128i C128i_R;

	__m256i M256i;
	__m256i M256i1;
	__m256i M256i2;

	__m256i B256i;
	__m256i B256i1;
	__m256i B256i2;

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

	__m256i M64block_L[32];
	__m256i M64block_R[32];
	__m256i C64block_L[32];
	__m256i C64block_R[32];

	__m256i S64block = allzero256;

	__m128i alpha;
	__m128i beta;
	__m256i alpha64block[32];
	__m256i beta64block[32];
	__m256i alphabeta64block[64 + 1];

	__m256i TK164block_L[32];
	__m256i TK164block_R[32];

#define TK64block TK164block_L

	__m256i SRK264block[ROUND_NUMBER][16];

	__m256i C64block[32];

	if (MByteN == 0)
	{
		TK1128i = _mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000006ULL);
		encrypt_1block(&alpha, N128i, TK1128i, K128i);
		TK1128i = _mm_set_epi64x(0x0000000000000001ULL, 0x0000000000000006ULL);
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

	TK1128i = _mm_set_epi64x(0x0000000000000000ULL, 0x0000000000000006ULL);
	encrypt_1block(&alpha, N128i, TK1128i, K128i);
	TK1128i = _mm_set_epi64x(0x0000000000000001ULL, 0x0000000000000006ULL);
	encrypt_1block(&beta, N128i, TK1128i, K128i);
	alphabeta64block[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(alpha), beta, 0x1); //_mm256_set_m128i(beta, alpha);

	s64 MBlockN = (MByteN + BLOCK_BYTE_NUMBER - 1) / BLOCK_BYTE_NUMBER;
	s64 MRe = MByteN % BLOCK_BYTE_NUMBER;

	s64 ResBlockN = MBlockN;

	if (MBlockN > (2 * PIPE))
	{
		for (int i = 0; i < 32; i++)
		{
			TK64block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);  //_mm256_set_m128i(K128i, K128i);
		}
		key_schedule(TK64block, SRK264block);
	}

	while (ResBlockN  > (2 * PIPE))
	{
		mul2_PIPE_256(alphabeta64block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			alpha64block[i] = _mm256_permute2x128_si256(alphabeta64block[2 * i + 0], alphabeta64block[2 * i + 1], 0x20);
			beta64block[i] = _mm256_permute2x128_si256(alphabeta64block[2 * i + 0], alphabeta64block[2 * i + 1], 0x31);

			M256ip = (__m256i *)(Mip + (i * 4 + 0) * BLOCK_BYTE_NUMBER);
			M256i1 = _mm256_loadu_si256(M256ip);
			M256ip = (__m256i *)(Mip + (i * 4 + 2) * BLOCK_BYTE_NUMBER);
			M256i2 = _mm256_loadu_si256(M256ip);

			M64block_L[i] = _mm256_permute2x128_si256(M256i1, M256i2, 0x20);
			M64block_R[i] = _mm256_permute2x128_si256(M256i1, M256i2, 0x31);
			S64block = _mm256_xor_si256(S64block, M64block_R[i]);
			C64block_L[i] = _mm256_xor_si256(M64block_L[i], alpha64block[i]);

			B128ip = (__m128i *)(Bip + (i * 4 + 0) * TRICK_BYTE_NUMBER);
			B128i1 = _mm_loadu_si128(B128ip);
			B128ip = (__m128i *)(Bip + (i * 4 + 2) * TRICK_BYTE_NUMBER);
			B128i2 = _mm_loadu_si128(B128ip);
			B256i1 = _mm256_inserti128_si256(_mm256_castsi128_si256(B128i1), B128i2, 0x1);  // _mm256_set_m128i(B128i2, B128i1);

			B256i1 = _mm256_slli_si256(_mm256_xor_si256(B256i1, beta64block[i]), 1);

			B128ip = (__m128i *)(Bip + (i * 4 + 1) * TRICK_BYTE_NUMBER);
			B128i1 = _mm_loadu_si128(B128ip);
			B128ip = (__m128i *)(Bip + (i * 4 + 3) * TRICK_BYTE_NUMBER);
			B128i2 = _mm_loadu_si128(B128ip);
			B256i2 = _mm256_inserti128_si256(_mm256_castsi128_si256(B128i1), B128i2, 0x1); // _mm256_set_m128i(B128i2, B128i1);
			B256i2 = _mm256_slli_si256(_mm256_xor_si256(B256i2, beta64block[i]), 1);

			TK164block_L[i] = B256i1;
			TK164block_R[i] = _mm256_xor_si256(B256i2, oneone256);
		}

		encrypt_64blocks(C64block_L, TK164block_L, SRK264block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			C64block_L[i] = _mm256_xor_si256(C64block_L[i], M64block_R[i]);
			C64block_R[i] = _mm256_xor_si256(C64block_L[i], alpha64block[i]);
		}

		encrypt_64blocks(C64block_R, TK164block_R, SRK264block);

		for (int i = 0; i < (PIPE >> 1); i++)
		{
			C64block_R[i] = _mm256_xor_si256(C64block_R[i], M64block_L[i]);

			M256i1 = _mm256_permute2x128_si256(C64block_L[i], C64block_R[i], 0x20);
			M256i2 = _mm256_permute2x128_si256(C64block_L[i], C64block_R[i], 0x31);
			C256ip = (__m256i *)(Cip + (i * 4 + 0) * BLOCK_BYTE_NUMBER);
			_mm256_storeu_si256(C256ip, M256i1);
			C256ip = (__m256i *)(Cip + (i * 4 + 2) * BLOCK_BYTE_NUMBER);
			_mm256_storeu_si256(C256ip, M256i2);
		}

		alphabeta64block[0] = alphabeta64block[PIPE];

		Mip += (2 * BLOCK_BYTE_NUMBER * PIPE);
		Bip += (2 * TRICK_BYTE_NUMBER * PIPE);
		Cip += (2 * BLOCK_BYTE_NUMBER * PIPE);
		ResBlockN -= (2 * PIPE);
	}

	if (MBlockN > (2 * PIPE))
	{
		S = _mm_xor_si128(_mm256_extracti128_si256(S64block, 0), _mm256_extracti128_si256(S64block, 1));
	}

	while (ResBlockN > 2)
	{
		mul2_256(&(alphabeta64block[0]), &(alphabeta64block[1]));
		alpha = _mm256_extracti128_si256(alphabeta64block[0], 0);
		beta = _mm256_extracti128_si256(alphabeta64block[0], 1);

		M128ip = (__m128i *)(Mip);
		M128i_L = _mm_loadu_si128(M128ip);
		C128i_L = _mm_xor_si128(M128i_L, alpha);

		M128ip = (__m128i *)(Mip + BLOCK_BYTE_NUMBER);
		M128i_R = _mm_loadu_si128(M128ip);
		S = _mm_xor_si128(S, M128i_R);

		B128ip = (__m128i *)(Bip);
		B128i1 = _mm_loadu_si128(B128ip);
		B128i1 = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);
		B128ip = (__m128i *)(Bip + TRICK_BYTE_NUMBER);
		B128i2 = _mm_loadu_si128(B128ip);
		B128i2 = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);
		B128i2 = _mm_xor_si128(B128i2, oneone);

		encrypt_1block(&C128i_L, C128i_L, B128i1, K128i);

		C128i_L = _mm_xor_si128(C128i_L, M128i_R);
		C128i_R = _mm_xor_si128(C128i_L, alpha);

		encrypt_1block(&C128i_R, C128i_R, B128i2, K128i);
		C128i_R = _mm_xor_si128(C128i_R, M128i_L);

		C128ip = (__m128i *)(Cip);
		_mm_storeu_si128(C128ip, C128i_L);
		C128ip = (__m128i *)(Cip + BLOCK_BYTE_NUMBER);
		_mm_storeu_si128(C128ip, C128i_R);

		alphabeta64block[0] = alphabeta64block[1];

		Mip += (2 * BLOCK_BYTE_NUMBER);
		Bip += (2 * TRICK_BYTE_NUMBER);
		Cip += (2 * BLOCK_BYTE_NUMBER);
		ResBlockN -= 2;
	}

	if (ResBlockN == 2)
	{
		alpha = _mm256_extracti128_si256(alphabeta64block[0], 0);
		beta = _mm256_extracti128_si256(alphabeta64block[0], 1);

		M128ip = (__m128i *)(Mip);
		B128ip = (__m128i *)(Bip);

		M128i_L = _mm_loadu_si128(M128ip);
		C128i_L = _mm_xor_si128(M128i_L, alpha);

		B128i1 = _mm_loadu_si128(B128ip);
		B128i1 = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);

		encrypt_1block(&C128i_L, C128i_L, B128i1, K128i);

		S = _mm_xor_si128(S, C128i_L);

		M128ip = (__m128i *)(Mip + BLOCK_BYTE_NUMBER);
		M128i_R = _mm_loadu_si128(M128ip);
		C128i_R = _mm_xor_si128(C128i_L, M128i_R);
		if (MRe != 0)
		{
			u8 * C128iu8a = (u8 *)(&C128i_R);
			ozpInplace(C128iu8a, MRe, BLOCK_BYTE_NUMBER);
			memcpy(Cip + BLOCK_BYTE_NUMBER, C128iu8a, MRe);
		}
		else
		{
			C128ip = (__m128i *)(Cip + BLOCK_BYTE_NUMBER);
			_mm_storeu_si128(C128ip, C128i_R);
		}
		S = _mm_xor_si128(S, C128i_R);

		C128i_R = _mm_xor_si128(C128i_R, alpha);

		B128i2 = _mm_slli_si128(beta, 1);
		B128i2 = _mm_xor_si128(B128i2, oneone);

		encrypt_1block(&C128i_R, C128i_R, B128i1, K128i);

		C128i_R = _mm_xor_si128(C128i_R, M128i_L);
		C128ip = (__m128i *)(Cip);
		_mm_storeu_si128(C128ip, C128i_R);

		Bip += (1 * TRICK_BYTE_NUMBER);
	} 
	else
	{
		alpha = _mm256_extracti128_si256(alphabeta64block[0], 0);
		beta = _mm256_extracti128_si256(alphabeta64block[0], 1);

		B128i1 = _mm_slli_si128(beta, 1);
		encrypt_1block(&C128i_L, alpha, B128i1, K128i);

		M128ip = (__m128i *)(Mip);
		M128i_L = _mm_loadu_si128(M128ip);
		C128i_L = _mm_xor_si128(C128i_L, M128i_L);
		if (MRe != 0)
		{
			u8 * C128iu8a = (u8 *)(&C128i_L);
			memcpy(Cip, C128iu8a, MRe);
			u8 * M128iu8a = (u8 *)(&M128i_L);
			ozpInplace(M128iu8a, MRe, BLOCK_BYTE_NUMBER);
		}
		else
		{
			C128ip = (__m128i *)(Cip);
			_mm_storeu_si128(C128ip, C128i_L);
		}

		S = _mm_xor_si128(S, M128i_L);
	}

	B128ip = (__m128i *)(Bip);
	B128i1 = _mm_loadu_si128(B128ip);
	B128i1 = _mm_slli_si128(_mm_xor_si128(B128i1, beta), 1);

	if (((MBlockN & 1) == 0) && (MRe != 0))
	{
		B128i1 = _mm_insert_epi8(B128i1, 2, 0);
	}
	else if (((MBlockN & 1) == 0) && (MRe == 0))
	{
		B128i1 = _mm_insert_epi8(B128i1, 3, 0);
	}
	else if (((MBlockN & 1) != 0) && (MRe != 0))
	{
		B128i1 = _mm_insert_epi8(B128i1, 4, 0);
	}
	else
	{
		B128i1 = _mm_insert_epi8(B128i1, 5, 0);
	}
	S = _mm_xor_si128(S, alpha);
	encrypt_1block(&S, S, B128i1, K128i);
	_mm_storeu_si128(Y128ip, S);
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
		gammadelta64block[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(gamma), delta, 0x1); // _mm256_set_m128i(delta, gamma);
		
		BhBlockN = BhByteN / (BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER);
		BhRe = BhByteN % (BLOCK_BYTE_NUMBER + TRICK_BYTE_NUMBER);
		BhBlockN_N_1;
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
				P64block[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(K128i), K128i, 0x1);  //_mm256_set_m128i(K128i, K128i);
			}
			key_schedule(P64block, SRK264block);
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