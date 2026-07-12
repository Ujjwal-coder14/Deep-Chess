#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include "chess.hpp"
#include "pieceSquareTables.hpp"
#include <chrono>
#include <fstream>
#include <cmath>
#define byte win_byte_override
#include <windows.h>
#undef byte
#include <filesystem>
#define EIGEN_VECTORIZE_AVX2
#define EIGEN_VECTORIZE_FMA
#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <eigen-5.0.0/Eigen/Dense>

using namespace chess;

constexpr int INF = 100000000;
constexpr int MATE_SCORE = 20000;
constexpr int SEARCH_DEPTH = 6;
constexpr int MAX_PLY = 128;
constexpr int CAPTURE_BASE = 8500;
constexpr int BAD_CAPTURE_SCALE = 2;
constexpr int SAFE_CAPTURE_BONUS = 1000;
constexpr int MAX_HISTORY_BONUS = 16384;

int fixedDepth = -1;
long long node = 0;
uint64_t nnCalls = 0;

Move killerMoves[MAX_PLY][2];
Move counter[64][64];

int history[2][64][64];

std::chrono::steady_clock::time_point searchStart;
int timeLimitMs;

bool stopSearch = false;

Board board;

enum TTFlag
{
    EXACT,
    LOWERBOUND,
    UPPERBOUND
};

struct TTEntry
{
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    TTFlag flag = EXACT;
    Move bestMove = Move::NULL_MOVE;
};

constexpr size_t TT_SIZE = 1 << 21;
constexpr size_t TT_MASK = TT_SIZE - 1;

std::vector<TTEntry> tt(TT_SIZE);

struct EvalEntry
{
    uint64_t key = 0;
    int32_t score = 0;
};

constexpr size_t EVAL_CACHE_SIZE = 1 << 21;
constexpr size_t EVAL_CACHE_MASK = EVAL_CACHE_SIZE - 1;

std::vector<EvalEntry> evalCache(EVAL_CACHE_SIZE);

uint64_t evalCacheProbes = 0;
uint64_t evalCacheHits = 0;

struct SearchStats
{
    uint64_t nodes = 0;

    uint64_t ttProbe = 0;
    uint64_t ttHit = 0;

    uint64_t nullAttempts = 0;
    uint64_t nullCutoffs = 0;

    uint64_t lmrAttempts = 0;
    uint64_t lmrResearches = 0;

    uint64_t pvsSearches = 0;
    uint64_t pvsResearches = 0;

    uint64_t hashMoveFirst = 0;
    uint64_t killerCutoffs = 0;
    uint64_t historyCutoffs = 0;
    uint64_t captureCutoffs = 0;

    uint64_t depthCond = 0;
    uint64_t moveCond = 0;
    uint64_t quietCond = 0;
    uint64_t notCheckCond = 0;

    uint64_t nullAtMax = 0;
    uint64_t nullAtMin = 0;
    uint64_t ttWrites = 0;
};

std::filesystem::path getExeDir()
{
    char path[MAX_PATH];
    GetModuleFileNameA(NULL, path, MAX_PATH);
    return std::filesystem::path(path).parent_path();
}

SearchStats stats;

int moveBudget(int rem, int inc)
{
    int moveTime = rem / 30 + inc * 3 / 4;
    moveTime = std::min(moveTime, rem / 5);
    return std::max(10, moveTime);
}

bool outOfTime()
{
    if (fixedDepth != -1)
        return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count();
    return elapsed >= timeLimitMs;
}

int pieceValue(PieceType pt)
{
    switch (pt.internal())
    {
    case PieceType::PAWN:
        return 100;
    case PieceType::KNIGHT:
        return 305;
    case PieceType::BISHOP:
        return 333;
    case PieceType::ROOK:
        return 563;
    case PieceType::QUEEN:
        return 950;
    default:
        return 0;
    }
}

bool onlyKingsAndPawns()
{
    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));
        if (p.type() != PieceType::NONE and p.type() != PieceType::KING and p.type() != PieceType::PAWN)
            return false;
    }
    return true;
}

void orderMoves(Movelist &moves, int ply)
{

    uint64_t hashKey = board.hash();
    
    size_t tt_index = hashKey & TT_MASK;
    const TTEntry &entry = tt[tt_index];

    Move hashMove = Move::NULL_MOVE;

    if (entry.depth != -1 && entry.key == hashKey)
    {
        hashMove = entry.bestMove;
    }

    for (auto &move : moves)
    {

        int score = 0;

        if (move == hashMove)
        {
            score += 20000;
        }

        if (board.isCapture(move))
        {
            int victimVal;
            if (move.typeOf() == Move::ENPASSANT)
            {
                victimVal = pieceValue(PieceType::PAWN);
            }
            else
            {
                victimVal = pieceValue(board.at(move.to()).type());
            }
            Piece victim =
                board.at(move.to());

            Piece attacker =
                board.at(move.from());

            score =
                CAPTURE_BASE + 10 * victimVal - pieceValue(attacker.type());

            board.makeMove(move);
            bool defended = board.isAttacked(move.to(), board.sideToMove());
            board.unmakeMove(move);

            if (defended)
            {
                int gain = victimVal - pieceValue(attacker.type());
                if (gain < 0)
                    score -= BAD_CAPTURE_SCALE * gain;
            }
            else
                score += SAFE_CAPTURE_BONUS;
        }
        else
        {
            if (move == killerMoves[ply][0])
            {
                score += 9000;
            }
            else if (move == killerMoves[ply][1])
            {
                score += 8000;
            }

            int source = move.from().index();
            int target = move.to().index();
            int color = (board.sideToMove() == Color::WHITE ? 0 : 1);

            score += history[color][source][target];
        }

        move.setScore(score);
    }

    std::sort(moves.begin(),
              moves.end(),
              [](const Move &a,
                 const Move &b)
              {
                  return a.score() > b.score();
              });
}

int kingShield(Color c, int file, int rank)
{
    int ans = 0;
    if (c == Color::WHITE)
    {
        if (rank == 7)
            return 0;
        if (board.at(Square(8 * (rank + 1) + file)).type() == PieceType::PAWN and board.at(Square(8 * (rank + 1) + file)).color() == c)
            ans++;
        if (file > 0)
        {
            if (board.at(Square(8 * (rank + 1) + file - 1)).type() == PieceType::PAWN and board.at(Square(8 * (rank + 1) + file - 1)).color() == c)
                ans++;
        }
        if (file < 7)
        {
            if (board.at(Square(8 * (rank + 1) + file + 1)).type() == PieceType::PAWN and board.at(Square(8 * (rank + 1) + file + 1)).color() == c)
                ans++;
        }
    }
    else
    {
        if (rank == 0)
            return 0;
        if (board.at(Square(8 * (rank - 1) + file)).type() == PieceType::PAWN and board.at(Square(8 * (rank - 1) + file)).color() == c)
            ans++;
        if (file > 0)
        {
            if (board.at(Square(8 * (rank - 1) + file - 1)).type() == PieceType::PAWN and board.at(Square(8 * (rank - 1) + file - 1)).color() == c)
                ans++;
        }
        if (file < 7)
        {
            if (board.at(Square(8 * (rank - 1) + file + 1)).type() == PieceType::PAWN and board.at(Square(8 * (rank - 1) + file + 1)).color() == c)
                ans++;
        }
    }
    return ans;
}

int classic_eval()
{
    // Mobility Bonuses
    const int passedPawnMG[8] = {0, 5, 10, 20, 35, 60, 100, 0};
    const int passedPawnEG[8] = {0, 10, 25, 45, 70, 110, 180, 0};
    const int knightMobilityMG[9] = {-15, -10, -5, 0, 4, 8, 12, 16, 20};
    const int bishopMobilityMG[14] = {-15, -10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    const int rookMobilityMG[15] = {-10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    const int queenMobilityMG[28] = {-10, -8, -6, -4, -2, 0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34};
    const int knightMobilityEG[9] = {-22, -15, -8, 0, 6, 12, 18, 24, 30};
    const int bishopMobilityEG[14] = {-21, -14, -7, 0, 3, 6, 8, 11, 14, 17, 20, 22, 25, 28};
    const int rookMobilityEG[15] = {-13, -7, 0, 3, 5, 8, 10, 13, 16, 18, 21, 23, 26, 29, 31};
    const int queenMobilityEG[28] = {-12, -10, -8, -6, -3, 0, 2, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44};

    constexpr int KNIGHT_ATTACK = 6;
    constexpr int BISHOP_ATTACK = 5;
    constexpr int ROOK_ATTACK = 4;
    constexpr int QUEEN_ATTACK = 5;
    constexpr int attackerMultiplier[9] = {0, 1, 2, 3, 4, 6, 8, 10, 12};

    int score = 0, midGame = 0, endGame = 0;
    int whiteBishop = 0, blackBishop = 0;

    Square whiteKing, blackKing;
    Bitboard whiteKingZone, blackKingZone;

    uint64_t whitePawns = 0ULL;
    uint64_t blackPawns = 0ULL;
    int whitePawnFile[8] = {0};
    int blackPawnFile[8] = {0};

    Bitboard occ = board.occ();
    int phase = 0;

    // Game Phase
    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));
        switch (p.type().internal())
        {
        case PieceType::KNIGHT:
            phase += 1;
            break;
        case PieceType::BISHOP:
            phase += 1;
            break;
        case PieceType::ROOK:
            phase += 2;
            break;
        case PieceType::QUEEN:
            phase += 4;
            break;
        }
        if (p.type() == PieceType::KING)
        {
            if (p.color() == Color::WHITE)
            {
                whiteKing = Square(sq);
                whiteKingZone = attacks::king(Square(sq));
            }
            else
            {
                blackKing = Square(sq);
                blackKingZone = attacks::king(Square(sq));
            }
        }
    }

    int blackAttackScore = 0, whiteAttackScore = 0, whiteAttackers = 0, blackAttackers = 0;

    // Material, PST, Mobility, King Shield, and Pawn Data
    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));
        if (p == Piece::NONE)
            continue;

        int file = sq % 8;
        int rank = sq / 8;
        int val = pieceValue(p.type());

        int mobility;
        Bitboard attacks;

        if (p.color() == Color::WHITE)
        {
            midGame += val;
            endGame += val;
            switch (p.type().internal())
            {
            case PieceType::PAWN:
                midGame += pawnOpenTable[sq];
                endGame += pawnEndTable[sq];
                whitePawns |= (1ULL << sq);
                whitePawnFile[file]++;
                break;
            case PieceType::KNIGHT:
                midGame += knightOpenTable[sq];
                endGame += knightEndTable[sq];
                attacks = attacks::knight(Square(sq));
                mobility = (attacks & ~board.us(Color::WHITE)).count();
                if ((attacks & blackKingZone).count())
                {
                    whiteAttackers++;
                    whiteAttackScore += KNIGHT_ATTACK;
                    whiteAttackScore += 2 * ((attacks & blackKingZone).count() - 1);
                }
                midGame += knightMobilityMG[mobility];
                endGame += knightMobilityEG[mobility];
                break;
            case PieceType::BISHOP:
                midGame += bishopOpenTable[sq];
                endGame += bishopEndTable[sq];
                whiteBishop++;
                attacks = attacks::bishop(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::WHITE)).count();

                if ((attacks & blackKingZone).count())
                {
                    whiteAttackers++;
                    whiteAttackScore += BISHOP_ATTACK;
                    whiteAttackScore += 2 * ((attacks & blackKingZone).count() - 1);
                }
                midGame += bishopMobilityMG[mobility];
                endGame += bishopMobilityEG[mobility];
                break;
            case PieceType::ROOK:
                midGame += rookOpenTable[sq];
                endGame += rookEndTable[sq];
                attacks = attacks::rook(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::WHITE)).count();

                if ((attacks & blackKingZone).count())
                {
                    whiteAttackers++;
                    whiteAttackScore += ROOK_ATTACK;
                    whiteAttackScore += 2 * ((attacks & blackKingZone).count() - 1);
                }
                midGame += rookMobilityMG[mobility];
                endGame += rookMobilityEG[mobility];
                break;
            case PieceType::QUEEN:
                midGame += queenOpenTable[sq];
                endGame += queenEndTable[sq];
                attacks = attacks::queen(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::WHITE)).count();

                if ((attacks & blackKingZone).count())
                {
                    whiteAttackers++;
                    whiteAttackScore += QUEEN_ATTACK;
                    whiteAttackScore += 2 * ((attacks & blackKingZone).count() - 1);
                }
                midGame += queenMobilityMG[mobility];
                endGame += queenMobilityEG[mobility];
                break;
            case PieceType::KING:
                midGame += kingOpenTable[sq];
                endGame += kingEndTable[sq];
                midGame += 12 * kingShield(Color::WHITE, file, rank);
                endGame += 2 * kingShield(Color::WHITE, file, rank);
                break;
            }
        }
        else
        {
            midGame -= val;
            endGame -= val;
            switch (p.type().internal())
            {
            case PieceType::PAWN:
                midGame -= pawnOpenTable[sq ^ 56];
                endGame -= pawnEndTable[sq ^ 56];
                blackPawns |= (1ULL << sq);
                blackPawnFile[file]++;
                break;
            case PieceType::KNIGHT:
                midGame -= knightOpenTable[sq ^ 56];
                endGame -= knightEndTable[sq ^ 56];
                attacks = attacks::knight(Square(sq));
                mobility = (attacks & ~board.us(Color::BLACK)).count();

                if ((attacks & whiteKingZone).count())
                {
                    blackAttackers++;
                    blackAttackScore += KNIGHT_ATTACK;
                    blackAttackScore += 2 * ((attacks & whiteKingZone).count() - 1);
                }
                midGame -= knightMobilityMG[mobility];
                endGame -= knightMobilityEG[mobility];
                break;
            case PieceType::BISHOP:
                midGame -= bishopOpenTable[sq ^ 56];
                endGame -= bishopEndTable[sq ^ 56];
                blackBishop++;
                attacks = attacks::bishop(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::BLACK)).count();

                if ((attacks & whiteKingZone).count())
                {
                    blackAttackers++;
                    blackAttackScore += BISHOP_ATTACK;
                    blackAttackScore += 2 * ((attacks & whiteKingZone).count() - 1);
                }
                midGame -= bishopMobilityMG[mobility];
                endGame -= bishopMobilityEG[mobility];
                break;
            case PieceType::ROOK:
                midGame -= rookOpenTable[sq ^ 56];
                endGame -= rookEndTable[sq ^ 56];
                attacks = attacks::rook(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::BLACK)).count();

                if ((attacks & whiteKingZone).count())
                {
                    blackAttackers++;
                    blackAttackScore += ROOK_ATTACK;
                    blackAttackScore += 2 * ((attacks & whiteKingZone).count() - 1);
                }
                midGame -= rookMobilityMG[mobility];
                endGame -= rookMobilityEG[mobility];
                break;
            case PieceType::QUEEN:
                midGame -= queenOpenTable[sq ^ 56];
                endGame -= queenEndTable[sq ^ 56];
                attacks = attacks::queen(Square(sq), occ);
                mobility = (attacks & ~board.us(Color::BLACK)).count();

                if ((attacks & whiteKingZone).count())
                {
                    blackAttackers++;
                    blackAttackScore += QUEEN_ATTACK;
                    blackAttackScore += 2 * ((attacks & whiteKingZone).count() - 1);
                }
                midGame -= queenMobilityMG[mobility];
                endGame -= queenMobilityEG[mobility];
                break;
            case PieceType::KING:
                midGame -= kingOpenTable[sq ^ 56];
                endGame -= kingEndTable[sq ^ 56];
                midGame -= 12 * kingShield(Color::BLACK, file, rank);
                endGame -= 2 * kingShield(Color::BLACK, file, rank);
                break;
            }
        }
    }

    midGame += attackerMultiplier[std::min(whiteAttackers, 8)] * whiteAttackScore;
    midGame -= attackerMultiplier[std::min(blackAttackers, 8)] * blackAttackScore;

    // Bishop pair bonus
    if (whiteBishop >= 2)
    {
        midGame += 25;
        endGame += 45;
    }
    if (blackBishop >= 2)
    {
        midGame -= 25;
        endGame -= 45;
    }

    // Double Pawns Penalty
    for (int file = 0; file < 8; file++)
    {
        if (whitePawnFile[file] > 1)
        {
            midGame -= 15 * (whitePawnFile[file] - 1);
            endGame -= 8 * (whitePawnFile[file] - 1);
        }
        if (blackPawnFile[file] > 1)
        {
            midGame += 15 * (blackPawnFile[file] - 1);
            endGame += 8 * (blackPawnFile[file] - 1);
        }
    }

    // Pawns and Rooks
    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));
        if (p == Piece::NONE)
            continue;

        int file = sq % 8;
        int rank = sq / 8;

        if (p.type() == PieceType::PAWN)
        {
            if (p.color() == Color::WHITE)
            {
                // Isolated Pawn
                bool isolated = true;
                if (file > 0 && whitePawnFile[file - 1] > 0)
                    isolated = false;
                if (file < 7 && whitePawnFile[file + 1] > 0)
                    isolated = false;
                if (isolated)
                {
                    midGame -= 10;
                    endGame -= 5;
                }

                // Passed Pawn
                bool isPassed = true;
                for (int r = rank + 1; r < 8; r++)
                {
                    if (blackPawns & (1ULL << (r * 8 + file)))
                    {
                        isPassed = false;
                        break;
                    }
                    if (file > 0 && (blackPawns & (1ULL << (r * 8 + file - 1))))
                    {
                        isPassed = false;
                        break;
                    }
                    if (file < 7 && (blackPawns & (1ULL << (r * 8 + file + 1))))
                    {
                        isPassed = false;
                        break;
                    }
                }
                if (isPassed)
                {
                    midGame += passedPawnMG[rank];
                    endGame += passedPawnEG[rank];
                }

                // Chained Pawn
                bool chained = false;
                if (rank > 0)
                {
                    if (file > 0 && (whitePawns & (1ULL << ((rank - 1) * 8 + file - 1))))
                        chained = true;
                    if (file < 7 && (whitePawns & (1ULL << ((rank - 1) * 8 + file + 1))))
                        chained = true;
                }
                if (chained)
                {
                    midGame += 6;
                    endGame += 2;
                }
            }
            else // BLACK
            {
                // Isolated Pawn
                bool isolated = true;
                if (file > 0 && blackPawnFile[file - 1] > 0)
                    isolated = false;
                if (file < 7 && blackPawnFile[file + 1] > 0)
                    isolated = false;
                if (isolated)
                {
                    midGame += 10;
                    endGame += 5;
                }

                // Passed Pawn
                bool isPassed = true;
                for (int r = rank - 1; r >= 0; r--)
                {
                    if (whitePawns & (1ULL << (r * 8 + file)))
                    {
                        isPassed = false;
                        break;
                    }
                    if (file > 0 && (whitePawns & (1ULL << (r * 8 + file - 1))))
                    {
                        isPassed = false;
                        break;
                    }
                    if (file < 7 && (whitePawns & (1ULL << (r * 8 + file + 1))))
                    {
                        isPassed = false;
                        break;
                    }
                }
                if (isPassed)
                {
                    midGame -= passedPawnMG[7 - rank];
                    endGame -= passedPawnEG[7 - rank];
                }

                // Chained Pawn
                bool chained = false;
                if (rank < 7)
                {
                    if (file > 0 && (blackPawns & (1ULL << ((rank + 1) * 8 + file - 1))))
                        chained = true;
                    if (file < 7 && (blackPawns & (1ULL << ((rank + 1) * 8 + file + 1))))
                        chained = true;
                }
                if (chained)
                {
                    midGame -= 6;
                    endGame -= 2;
                }
            }
        }
        else if (p.type() == PieceType::ROOK)
        {
            // Rook on open/semi-open file
            bool friendlyPawn = (p.color() == Color::WHITE) ? (whitePawnFile[file] > 0) : (blackPawnFile[file] > 0);
            bool enemyPawn = (p.color() == Color::WHITE) ? (blackPawnFile[file] > 0) : (whitePawnFile[file] > 0);

            if (!friendlyPawn)
            {
                if (!enemyPawn)
                {
                    midGame += (p.color() == Color::WHITE ? 20 : -20);
                    endGame += (p.color() == Color::WHITE ? 12 : -12);
                }
                else
                {
                    midGame += (p.color() == Color::WHITE ? 10 : -10);
                    endGame += (p.color() == Color::WHITE ? 6 : -6);
                }
            }
        }
    }
    score = ((midGame * phase) + (endGame * (24 - phase))) / 24;

    return score;
}

class NeuralEval
{
public:
    void loadWeights();

    int nn_eval(const Board &board, int ply);

    void refreshRoot(const Board &board);

    void pushMove(int from_ply, int to_ply, const Move &move, const Board &prev_board);

    std::array<float, 787> getInput(const Board &board);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    uint64_t tInput = 0;
    uint64_t tLayer1 = 0;
    uint64_t tLayer2 = 0;
    uint64_t tLayer3 = 0;
    uint64_t tLayer4 = 0;

    uint64_t evalCalls = 0;

private:
    inline int getPieceFeatureIndex(Color color, PieceType type, int sq)
    {
        int offset = (color == Color::BLACK) ? 384 : 0;
        switch (type.internal())
        {
        case PieceType::KING:
            return offset + sq;
        case PieceType::QUEEN:
            return offset + 64 + sq;
        case PieceType::ROOK:
            return offset + 128 + sq;
        case PieceType::BISHOP:
            return offset + 192 + sq;
        case PieceType::KNIGHT:
            return offset + 256 + sq;
        case PieceType::PAWN:
            return offset + 320 + sq;
        default:
            return -1;
        }
    }

    inline int getPieceFeatureIndex(Piece p, int sq)
    {
        if (p == Piece::NONE)
            return -1;
        return getPieceFeatureIndex(p.color(), p.type(), sq);
    }

    Eigen::Matrix<float, 768, 768, Eigen::RowMajor> W1_pieces;
    Eigen::Matrix<float, 768, 19, Eigen::RowMajor> W1_global;
    Eigen::Matrix<float, 432, 768, Eigen::RowMajor> W2;
    Eigen::Matrix<float, 128, 432, Eigen::RowMajor> W3;
    Eigen::Matrix<float, 1, 128, Eigen::RowMajor> W4;

    Eigen::Matrix<float, 768, 1> b1;
    Eigen::Matrix<float, 432, 1> b2;
    Eigen::Matrix<float, 128, 1> b3;

    float b4;
    Eigen::Matrix<float, 768, 1> acc_stack[MAX_PLY];
};

template <typename Derived>
void loadMatrix(const std::string &file,
                Eigen::MatrixBase<Derived> &M)
{
    std::ifstream fin(file);

    if (!fin)
    {
        std::cerr << "Couldn't open " << file << std::endl;
        exit(1);
    }

    for (int i = 0; i < M.rows(); i++)
        for (int j = 0; j < M.cols(); j++)
            fin >> M(i, j);
}

template <typename Derived>
void loadVector(const std::string &file,
                Eigen::MatrixBase<Derived> &V)
{
    std::ifstream fin(file);

    if (!fin)
    {
        std::cerr << "Couldn't open " << file << std::endl;
        exit(1);
    }

    for (int i = 0; i < V.rows(); i++)
        fin >> V(i);
}

void NeuralEval::loadWeights()
{
    auto base = getExeDir();

    Eigen::MatrixXf W1_full(768, 787);
    loadMatrix((base / "fc1.weight.txt").string(), W1_full);

    W1_pieces = W1_full.leftCols<768>();
    W1_global = W1_full.rightCols<19>();

    loadVector((base / "fc1.bias.txt").string(), b1);
    loadMatrix((base / "fc2.weight.txt").string(), W2);
    loadVector((base / "fc2.bias.txt").string(), b2);
    loadMatrix((base / "fc3.weight.txt").string(), W3);
    loadVector((base / "fc3.bias.txt").string(), b3);
    loadMatrix((base / "fc4.weight.txt").string(), W4);

    std::ifstream fin((base / "fc4.bias.txt").string());
    fin >> b4;
}

inline float relu(float x)
{
    return x > 0 ? x : 0;
}

std::array<float, 787> NeuralEval::getInput(const Board &board)
{
    std::array<float, 787> x{};
    x.fill(0.0f);

    int whiteBishop = 0;
    int blackBishop = 0;
    int phase = 0;

    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));

        if (p == Piece::NONE)
            continue;

        bool black = (p.color() == Color::BLACK);

        int offset = black ? 384 : 0;

        switch (p.type().internal())
        {
        case PieceType::KING:
            x[offset + sq] = 1.0f;
            break;
        case PieceType::QUEEN:

            x[offset + 64 + sq] = 1.0f;

            phase += 4;

            if (black)
                x[780] += 1.0f;
            else
                x[775] += 1.0f;

            break;
        case PieceType::ROOK:

            x[offset + 128 + sq] = 1.0f;

            phase += 2;

            if (black)
                x[781] += 1.0f;
            else
                x[776] += 1.0f;

            break;
        case PieceType::BISHOP:

            x[offset + 192 + sq] = 1.0f;

            phase++;

            if (black)
            {
                blackBishop++;
                x[782] += 1.0f;
            }
            else
            {
                whiteBishop++;
                x[777] += 1.0f;
            }

            break;
        case PieceType::KNIGHT:

            x[offset + 256 + sq] = 1.0f;

            phase++;

            if (black)
                x[783] += 1.0f;
            else
                x[778] += 1.0f;

            break;
        case PieceType::PAWN:

            x[offset + 320 + sq] = 1.0f;

            if (black)
                x[784] += 1.0f;
            else
                x[779] += 1.0f;

            break;
        }
    }
    const auto rights = board.castlingRights();

    x[768] = rights.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE);
    x[769] = rights.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE);
    x[770] = rights.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE);
    x[771] = rights.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE);

    x[772] = (board.sideToMove() == Color::WHITE ? 1 : 0);

    x[773] = (whiteBishop == 2);

    x[774] = (blackBishop == 2);
    x[775] /= 1.0f;
    x[776] /= 2.0f;
    x[777] /= 2.0f;
    x[778] /= 2.0f;
    x[779] /= 8.0f;

    x[780] /= 1.0f;
    x[781] /= 2.0f;
    x[782] /= 2.0f;
    x[783] /= 2.0f;
    x[784] /= 8.0f;

    x[785] = phase / 24.0f;
    float whiteMaterial = 9 * x[775] + 5 * x[776] + 3 * x[777] + 3 * x[778] + 1 * x[779];
    float blackMaterial = 9 * x[780] + 5 * x[781] + 3 * x[782] + 3 * x[783] + 1 * x[784];
    x[786] = (whiteMaterial - blackMaterial) / 39.0f;
    return x;
}

int NeuralEval::nn_eval(const Board &board, int ply)
{
    evalCalls++;
    nnCalls++;

    Eigen::Matrix<float, 19, 1> gX;

    const auto rights = board.castlingRights();
    gX[0] = rights.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE);
    gX[1] = rights.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE);
    gX[2] = rights.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE);
    gX[3] = rights.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE);
    gX[4] = (board.sideToMove() == Color::WHITE ? 1.0f : 0.0f);

    float whiteQueens = board.pieces(PieceType::QUEEN, Color::WHITE).count();
    float whiteRooks = board.pieces(PieceType::ROOK, Color::WHITE).count();
    float whiteBishops = board.pieces(PieceType::BISHOP, Color::WHITE).count();
    float whiteKnights = board.pieces(PieceType::KNIGHT, Color::WHITE).count();
    float whitePawns = board.pieces(PieceType::PAWN, Color::WHITE).count();

    float blackQueens = board.pieces(PieceType::QUEEN, Color::BLACK).count();
    float blackRooks = board.pieces(PieceType::ROOK, Color::BLACK).count();
    float blackBishops = board.pieces(PieceType::BISHOP, Color::BLACK).count();
    float blackKnights = board.pieces(PieceType::KNIGHT, Color::BLACK).count();
    float blackPawns = board.pieces(PieceType::PAWN, Color::BLACK).count();

    gX[5] = (whiteBishops == 2.0f) ? 1.0f : 0.0f;
    gX[6] = (blackBishops == 2.0f) ? 1.0f : 0.0f;

    gX[7] = whiteQueens;
    gX[8] = whiteRooks / 2.0f;
    gX[9] = whiteBishops / 2.0f;
    gX[10] = whiteKnights / 2.0f;
    gX[11] = whitePawns / 8.0f;

    gX[12] = blackQueens;
    gX[13] = blackRooks / 2.0f;
    gX[14] = blackBishops / 2.0f;
    gX[15] = blackKnights / 2.0f;
    gX[16] = blackPawns / 8.0f;

    int phase = (int)(whiteQueens * 4 + whiteRooks * 2 + whiteBishops + whiteKnights +
                      blackQueens * 4 + blackRooks * 2 + blackBishops + blackKnights);
    gX[17] = phase / 24.0f;

    float whiteMaterial = 9.0f * whiteQueens + 5.0f * whiteRooks + 3.0f * whiteBishops + 3.0f * whiteKnights + whitePawns;
    float blackMaterial = 9.0f * blackQueens + 5.0f * blackRooks + 3.0f * blackBishops + 3.0f * blackKnights + blackPawns;
    gX[18] = (whiteMaterial - blackMaterial) / 39.0f;

    Eigen::Matrix<float, 768, 1> h1 = (acc_stack[ply] + W1_global * gX).cwiseMax(0.0f);

    Eigen::Matrix<float, 432, 1> h2 = (W2 * h1 + b2).cwiseMax(0.0f);
    Eigen::Matrix<float, 128, 1> h3 = (W3 * h2 + b3).cwiseMax(0.0f);
    float out = (W4 * h3)(0) + b4;

    return std::lround(out);
}

void NeuralEval::refreshRoot(const Board &board)
{
    acc_stack[0] = b1;

    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));
        int idx = getPieceFeatureIndex(p, sq);
        if (idx != -1)
        {
            acc_stack[0] += W1_pieces.col(idx);
        }
    }
}

void NeuralEval::pushMove(int from_ply, int to_ply, const Move &move, const Board &prev_board)
{
    acc_stack[to_ply] = acc_stack[from_ply];

    Square from = move.from();
    Square to = move.to();
    Piece moving_piece = prev_board.at(from);
    Piece captured_piece = prev_board.at(to);

    int from_idx = getPieceFeatureIndex(moving_piece, from.index());
    if (from_idx != -1)
        acc_stack[to_ply] -= W1_pieces.col(from_idx);

    if (move.typeOf() == Move::ENPASSANT)
    {
        Square ep_sq = Square(to.file(), from.rank());
        Piece ep_pawn = prev_board.at(ep_sq);
        int ep_idx = getPieceFeatureIndex(ep_pawn, ep_sq.index());
        if (ep_idx != -1)
            acc_stack[to_ply] -= W1_pieces.col(ep_idx);
    }
    else if (captured_piece != Piece::NONE)
    {
        int cap_idx = getPieceFeatureIndex(captured_piece, to.index());
        if (cap_idx != -1)
            acc_stack[to_ply] -= W1_pieces.col(cap_idx);
    }

    if (move.typeOf() == Move::PROMOTION)
    {
        PieceType promo_type = move.promotionType();
        int promo_idx = getPieceFeatureIndex(moving_piece.color(), promo_type, to.index());
        if (promo_idx != -1)
            acc_stack[to_ply] += W1_pieces.col(promo_idx);
    }
    else
    {
        int to_idx = getPieceFeatureIndex(moving_piece, to.index());
        if (to_idx != -1)
            acc_stack[to_ply] += W1_pieces.col(to_idx);
    }

    if (move.typeOf() == Move::CASTLING)
    {
        int r_from, r_to;
        Color c = moving_piece.color();
        if ((to.index() % 8) == 6)
        {
            r_from = (c == Color::WHITE) ? 7 : 63;
            r_to = (c == Color::WHITE) ? 5 : 61;
        }
        else
        {
            r_from = (c == Color::WHITE) ? 0 : 56;
            r_to = (c == Color::WHITE) ? 3 : 59;
        }
        Piece rook = prev_board.at(Square(r_from));
        acc_stack[to_ply] -= W1_pieces.col(getPieceFeatureIndex(rook, r_from));
        acc_stack[to_ply] += W1_pieces.col(getPieceFeatureIndex(rook, r_to));
    }
}

NeuralEval neural;

int cachedNNEval(const Board &board, int ply)
{
    uint64_t key = board.hash();
    size_t idx = key & EVAL_CACHE_MASK;
    EvalEntry &entry = evalCache[idx];

    evalCacheProbes++;

    if (entry.key == key)
    {
        evalCacheHits++;
        return entry.score;
    }

    int score = neural.nn_eval(board, ply);
    entry = {key, score};
    return score;
}

int get_static_eval(const Board &board, int alpha, int beta, int ply)
{
    if (board.inCheck())
    {
        return cachedNNEval(board, ply);
    }

    int classical_score = classic_eval();

    const int LAZY_MARGIN = 100;
    if (classical_score + LAZY_MARGIN <= alpha)
    {
        return classical_score;
    }

    if (classical_score - LAZY_MARGIN >= beta)
    {
        return classical_score;
    }

    return cachedNNEval(board, ply);
}

int quiescence(int alpha, int beta, bool maximizingPlayer, int ply)
{
    node++;
    stats.nodes++;

    uint64_t key = board.hash();
    size_t tt_index = key & TT_MASK;
    TTEntry &entry = tt[tt_index];

    if (entry.key == key && entry.depth >= 0)
    {
        if (entry.flag == EXACT) return entry.score;
        if (entry.flag == LOWERBOUND && entry.score >= beta) return entry.score;
        if (entry.flag == UPPERBOUND && entry.score <= alpha) return entry.score;
    }

    int static_eval = get_static_eval(board, alpha, beta, ply);

    if (outOfTime())
    {
        stopSearch = true;
        return static_eval;
    }

    bool inCheck = board.inCheck();

    if (!inCheck)
    {
        int standPat = static_eval;

        if (maximizingPlayer)
        {
            if (standPat >= beta)
                return beta;

            alpha = std::max(alpha, standPat);
        }
        else
        {
            if (standPat <= alpha)
                return alpha;

            beta = std::min(beta, standPat);
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);
    orderMoves(moves, ply);

    if (moves.empty())
    {
        if (inCheck)
        {
            return maximizingPlayer
                       ? -MATE_SCORE + ply
                       : MATE_SCORE - ply;
        }

        return maximizingPlayer ? alpha : beta;
    }

    for (const Move &move : moves)
    {
        if (stopSearch)
            break;
        if (!inCheck && !board.isCapture(move))
            continue;

        Board prev_board = board;

        board.makeMove(move);

        neural.pushMove(ply, ply + 1, move, prev_board);

        int score =
            quiescence(
                alpha,
                beta,
                !maximizingPlayer, ply + 1);

        board.unmakeMove(move);

        if (stopSearch)
            return maximizingPlayer ? alpha : beta;

        if (maximizingPlayer)
        {
            alpha = std::max(alpha, score);

            if (alpha >= beta)
                return beta;
        }
        else
        {
            beta = std::min(beta, score);

            if (beta <= alpha)
                return alpha;
        }
    }

    return maximizingPlayer ? alpha : beta;
}

int minimax(int depth, int ply, int alpha, int beta, bool maximizingPlayer, bool wasNull = false)
{
    node++;
    stats.nodes++;

    if (outOfTime())
    {
        stopSearch = true;
        return get_static_eval(board, alpha, beta, ply);
    }
    int alphaOrig = alpha;
    int betaOrig = beta;

    uint64_t key = board.hash();
    stats.ttProbe++;

    size_t tt_index = key & TT_MASK;

    TTEntry &entry = tt[tt_index];

    if (entry.key == key)
    {
        stats.ttHit++;

        if (entry.depth >= depth)
        {
            if (entry.flag == EXACT)
            {
                return entry.score;
            }
            if (entry.flag == LOWERBOUND)
            {
                alpha = std::max(alpha, entry.score);
            }
            if (entry.flag == UPPERBOUND)
            {
                beta = std::min(beta, entry.score);
            }
            if (alpha >= beta)
            {
                return entry.score;
            }
        }
    }

    if (depth == 0)
    {

        return quiescence(alpha, beta, maximizingPlayer, ply);
    }

    int NULL_RED = (depth >= 7 ? 3 : 2);

    if (depth >= 4 and !wasNull and !board.inCheck() and !onlyKingsAndPawns())
    {
        stats.nullAtMax += maximizingPlayer;
        stats.nullAtMin += !maximizingPlayer;
        board.makeNullMove();
        stats.nullAttempts++;
        int score = minimax(depth - 1 - NULL_RED, ply + 1, alpha, beta, !maximizingPlayer, true);
        board.unmakeNullMove();
        if (maximizingPlayer)
        {
            if (score >= beta)
            {
                stats.nullCutoffs++;
                return beta;
            }
        }
        else
        {
            if (score <= alpha)
            {
                stats.nullCutoffs++;
                return alpha;
            }
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, board);

    orderMoves(moves, ply);

    if (moves.empty())
    {

        if (board.inCheck())
        {

            if (maximizingPlayer)
                return -MATE_SCORE + ply; // Rewarding mate in less steps higher
            else
                return MATE_SCORE - ply;
        }

        return 0;
    }

    if (maximizingPlayer)
    {

        int best = -INF;

        Move bestMove = Move::NULL_MOVE;

        int moveNumber = 0;

        for (const Move &move : moves)
        {
            moveNumber++;

            if (stopSearch)
                break;

            bool isCapture = board.isCapture(move);

            Board prev_board = board;

            board.makeMove(move);

            neural.pushMove(ply, ply + 1, move, prev_board);

            int score;
            if (moveNumber == 1 or depth <= 2)
            {
                score = minimax(
                    depth - 1,
                    ply + 1,
                    alpha,
                    beta,
                    false);
            }
            else
            {
                int reduction = 1;

                bool canReduce = depth >= 4 and moveNumber >= 5 and !isCapture and !board.inCheck();

                if (canReduce)
                {
                    stats.lmrAttempts++;
                    score = minimax(
                        depth - 1 - reduction,
                        ply + 1,
                        alpha,
                        alpha + 1,
                        false);

                    if (score > alpha and score < beta)
                    {
                        stats.lmrResearches++;
                        score = minimax(
                            depth - 1,
                            ply + 1,
                            alpha,
                            beta,
                            false);
                    }
                }
                else
                {
                    score = minimax(
                        depth - 1,
                        ply + 1,
                        alpha,
                        alpha + 1,
                        false);

                    if (score > alpha && score < beta)
                    {
                        stats.pvsResearches++;
                        score = minimax(
                            depth - 1,
                            ply + 1,
                            alpha,
                            beta,
                            false);
                    }
                }
            }
            board.unmakeMove(move);

            if (stopSearch)
                return best;

            if (score > best)
            {

                best = score;

                bestMove = move;
            }

            best = std::max(best, score);
            alpha = std::max(alpha, best);

            if (beta <= alpha)
            {
                if (!board.isCapture(move))
                {
                    killerMoves[ply][1] = killerMoves[ply][0];
                    killerMoves[ply][0] = move;

                    int source = move.from().index(), target = move.to().index(), color = (board.sideToMove() == Color::WHITE ? 0 : 1);
                    int bonus = depth * depth;
                    history[color][source][target] = bonus - history[color][source][target] * abs(bonus) / MAX_HISTORY_BONUS;
                }
                break;
            }
        }
        TTFlag flag;

        if (best <= alphaOrig)
            flag = UPPERBOUND;
        else if (best >= betaOrig)
            flag = LOWERBOUND;
        else
            flag = EXACT;

        if (!stopSearch)
        {
            if (entry.key != key || depth >= tt[tt_index].depth) {
                tt[tt_index] = {key, depth, best, flag, bestMove};
                stats.ttWrites++;
            }
        }

        return best;
    }
    else
    {

        int best = INF;

        Move bestMove = Move::NULL_MOVE;

        int moveNumber = 0;

        for (const Move &move : moves)
        {
            moveNumber++;
            if (stopSearch)
                break;

            bool isCapture = board.isCapture(move);

            Board prev_board = board;

            board.makeMove(move);

            neural.pushMove(ply, ply + 1, move, prev_board);

            int score;
            if (moveNumber == 1 or depth <= 2)
            {
                score =
                    minimax(depth - 1,
                            ply + 1,
                            alpha,
                            beta,
                            true);
            }
            else
            {
                int reduction = 1;
                bool canReduce = depth >= 4 and moveNumber >= 5 and !isCapture and !board.inCheck();

                if (canReduce)
                {
                    stats.lmrAttempts++;
                    score =
                        minimax(
                            depth - 1 - reduction,
                            ply + 1,
                            beta - 1,
                            beta,
                            true);

                    if (score > alpha and score < beta)
                    {
                        stats.lmrResearches++;
                        score =
                            minimax(
                                depth - 1,
                                ply + 1,
                                alpha,
                                beta,
                                true);
                    }
                }
                else
                {
                    score =
                        minimax(
                            depth - 1,
                            ply + 1,
                            beta - 1,
                            beta,
                            true);

                    if (score > alpha && score < beta)
                    {
                        stats.pvsResearches++;
                        score =
                            minimax(
                                depth - 1,
                                ply + 1,
                                alpha,
                                beta,
                                true);
                    }
                }
            }

            board.unmakeMove(move);

            if (stopSearch)
                return best;

            if (score < best)
            {

                best = score;

                bestMove = move;
            }

            best = std::min(best, score);
            beta = std::min(beta, best);

            if (beta <= alpha)
            {
                if (!board.isCapture(move))
                {
                    killerMoves[ply][1] = killerMoves[ply][0];
                    killerMoves[ply][0] = move;

                    int source = move.from().index(), target = move.to().index(), color = (board.sideToMove() == Color::WHITE ? 0 : 1);
                    int bonus = depth * depth;
                    history[color][source][target] = bonus - history[color][source][target] * abs(bonus) / MAX_HISTORY_BONUS;
                }
                break;
            }
        }

        TTFlag flag;

        if (best <= alphaOrig)
            flag = UPPERBOUND;
        else if (best >= betaOrig)
            flag = LOWERBOUND;
        else
            flag = EXACT;

        if (!stopSearch)
        {
            if (entry.key != key || depth >= tt[tt_index].depth) {
                tt[tt_index] = {key, depth, best, flag, bestMove};
                stats.ttWrites++;
            }
        }

        return best;
    }
}

Move search()
{
    std::cerr << "Entered search\n";
    node = 0;
    stats = SearchStats{};

    searchStart = std::chrono::steady_clock::now();
    stopSearch = false;

    Move bestMove =
        Move::NULL_MOVE;

    Move lastCompletedBestMove =
        Move::NULL_MOVE;

    bool maximizingPlayer =
        board.sideToMove() ==
        Color::WHITE;

    int maxDepth = (fixedDepth == -1 ? INF : fixedDepth);

    int currentDepth = 1;

    neural.refreshRoot(board);

    for (currentDepth; currentDepth <= maxDepth;
         currentDepth++)
    {
        std::cerr << "Depth " << currentDepth << std::endl;

        Movelist moves;

        movegen::legalmoves(
            moves,
            board);

        orderMoves(moves, 0);

        int bestScore =
            maximizingPlayer
                ? -INF
                : INF;

        int alpha = -INF, beta = INF;
        bool first = true;

        for (const Move &move : moves)
        {
            if (stopSearch)
                break;

            Board prev_board = board;

            board.makeMove(
                move);

            neural.pushMove(0, 1, move, prev_board);

            int score;

            if (first)
            {
                score = minimax(currentDepth - 1, 1, alpha, beta, !maximizingPlayer);
                first = false;
            }
            else
            {
                if (maximizingPlayer)
                {
                    score = minimax(currentDepth - 1, 1, alpha, alpha + 1, false);
                    if (score > alpha and score < beta)
                    {
                        score = minimax(currentDepth - 1, 1, alpha, beta, false);
                    }
                }
                else
                {
                    score = minimax(currentDepth - 1, 1, beta - 1, beta, true);
                    if (score > alpha and score < beta)
                    {
                        score = minimax(currentDepth - 1, 1, alpha, beta, true);
                    }
                }
            }

            board.unmakeMove(
                move);

            if (maximizingPlayer)
            {
                if (score >
                    bestScore)
                {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }
                alpha = std::max(alpha, bestScore);
            }
            else
            {
                if (score <
                    bestScore)
                {
                    bestScore =
                        score;

                    bestMove =
                        move;
                }
                beta = std::min(beta, bestScore);
            }
            if (alpha >= beta)
                break;
        }
        if (stopSearch)
        {
            break;
        }
        lastCompletedBestMove = bestMove;
    }

    std::cout << "Info string Nodes: " << stats.nodes << '\n';
    std::cout << "info string Completed depth = "
              << currentDepth - 1
              << std::endl;

    std::cout << "Info string TT probes: " << stats.ttProbe << '\n';
    std::cout << "Info string TT hits: " << stats.ttHit << '\n';

    if (stats.ttProbe)
    {
        std::cout << "Info string TT hit rate: "
                  << 100.0 * stats.ttHit / stats.ttProbe
                  << "%\n";
    }
    std::cout << "info string Null attempts: " << stats.nullAttempts << '\n';

    std::cout << "info string Null cutoffs: " << stats.nullCutoffs << '\n';

    if (stats.nullAttempts)
    {
        std::cout << "info string Null success: "
                  << 100.0 * stats.nullCutoffs / stats.nullAttempts
                  << "%\n";
    }

    std::cout << "info string lmr attempts: " << stats.lmrAttempts << '\n';

    std::cout << "info string lmr researches: " << stats.lmrResearches << '\n';

    if (stats.lmrAttempts)
    {
        std::cout << "info string Null success: "
                  << 100.0 * stats.lmrResearches / stats.lmrAttempts
                  << "%\n";
    }
    std::cout << "Info string depth condition: " << stats.depthCond << '\n';
    std::cout << "Info string move condition: " << stats.moveCond << '\n';
    std::cout << "Info string quiet condition: " << stats.quietCond << '\n';
    std::cout << "Info string check condition: " << stats.notCheckCond << '\n';

    std::cout << "Info string null from max player" << stats.nullAtMax << '\n';
    std::cout << "Info string null from min player" << stats.nullAtMin << '\n';

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count();
    double seconds = elapsed / 1000.0;

    double nps = stats.nodes / std::max(seconds, 0.001);
    std::cout << "info string Time: " << elapsed << " ms\n";
    std::cout << "info string NPS: " << (long long)nps << '\n';
    std::cout << "info string NN calls = " << nnCalls << '\n';
    std::cout << "info string Eval cache probes = " << evalCacheProbes << '\n';
    std::cout << "info string Eval cache hits = " << evalCacheHits << '\n';
    if (evalCacheProbes)
    {
        std::cout << "info string Eval cache hit rate: "
                  << 100.0 * evalCacheHits / evalCacheProbes
                  << "%\n";
    }
    std::cout << "info string TT writes: " << stats.ttWrites << '\n';

    std::cerr << "Leaving search\n";

    return lastCompletedBestMove;
}

void parsePosition(const std::string &command)
{

    std::stringstream ss(command);

    std::string token;

    ss >> token;
    ss >> token;

    if (token == "startpos")
    {

        board.setFen(constants::STARTPOS);
    }
    else if (token == "fen")
    {

        std::string fen;

        for (int i = 0; i < 6; i++)
        {

            ss >> token;

            fen += token;

            if (i != 5)
                fen += " ";
        }

        board.setFen(fen);
    }

    if (ss >> token && token == "moves")
    {

        while (ss >> token)
        {

            Move move =
                uci::uciToMove(board, token);

            board.makeMove(move);
        }
    }
}

int wtime = 0, btime = 0, winc = 0, binc = 0;

void parseGo(const std::string &command)
{

    std::stringstream ss(command);

    std::string token;

    ss >> token;

    while (ss >> token)
    {

        if (token == "wtime")
        {

            ss >> wtime;
        }
        else if (token == "btime")
        {

            ss >> btime;
        }
        else if (token == "winc")
        {

            ss >> winc;
        }
        else if (token == "binc")
        {

            ss >> binc;
        }
        else if (token == "depth")
        {
            ss >> fixedDepth;
        }
    }
}

int main()
{
    Eigen::setNbThreads(1);

    neural.loadWeights();

    board.setFen(constants::STARTPOS);

    std::string line;

    while (std::getline(std::cin, line))
    {

        if (line == "uci")
        {

            std::cout << "id name trialBot1" << std::endl;
            std::cout << "id author Ujjwal" << std::endl;
            std::cout << "uciok" << std::endl;
        }

        else if (line == "isready")
        {

            std::cout << "readyok" << std::endl;
        }

        else if (line == "ucinewgame")
        {
            std::fill(tt.begin(), tt.end(), TTEntry{});

            memset(killerMoves, 0, sizeof(killerMoves));

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 64; j++)
                {
                    for (int k = 0; k < 64; k++)
                    {
                        history[i][j][k] /= 4;
                    }
                }
            }

            board.setFen(constants::STARTPOS);
        }

        else if (line.rfind("position", 0) == 0)
        {

            parsePosition(line);
        }

        else if (line.rfind("go", 0) == 0)
        {
            parseGo(line);

            if (board.sideToMove() == Color::WHITE)
            {
                timeLimitMs = moveBudget(wtime, winc);
            }
            else
            {
                timeLimitMs = moveBudget(btime, binc);
            }

            Move bestMove = search();

            if (bestMove == Move::NULL_MOVE)
            {

                std::cout << "bestmove 0000" << std::endl;
            }
            else
            {

                std::cout << "bestmove "
                          << uci::moveToUci(bestMove)
                          << std::endl;
            }
        }

        else if (line == "quit")
        {

            break;
        }
    }

    return 0;
}
