#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include "chess.hpp"
#include "pieceSquareTables.hpp"
#include <chrono>

using namespace chess;

constexpr int INF = 100000000;
constexpr int MATE_SCORE = 20000;
constexpr int SEARCH_DEPTH = 6;
constexpr int MAX_PLY = 64;
constexpr int CAPTURE_BASE = 8500;
constexpr int BAD_CAPTURE_SCALE = 2;
constexpr int SAFE_CAPTURE_BONUS = 1000;
constexpr int MAX_HISTORY_BONUS = 16384;

int fixedDepth = -1; // Purely for testing purpose, otherwise useless in game
long long node = 0;

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
    int depth;
    int score;
    TTFlag flag;
    Move bestMove;
};

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
};

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

std::unordered_map<uint64_t, TTEntry> tt; // Transposition table: Memoization

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

    auto it = tt.find(board.hash());

    Move hashMove = Move::NULL_MOVE;

    if (it != tt.end())
    {
        hashMove = it->second.bestMove;
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

int evaluate()
{
    // Mobility Bonuses
    const int passedPawnMG[8] = {0, 5, 10, 20, 35, 60, 100, 0};
    const int passedPawnEG[8] = {0, 10, 25, 45, 70, 110, 180, 0};
    const int knightMobilityMG[9] = {-15, -10, -5, 0, 4, 8, 12, 16, 20};
    const int bishopMobilityMG[14] = {-15, -10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    const int rookMobilityMG[15] = {-10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
    const int queenMobilityMG[28] = {-10, -8, -6, -4, -2, 0, 1, 2, 4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34};
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

int quiescence(int alpha, int beta, bool maximizingPlayer, int ply)
{
    node++;
    stats.nodes++;

    if (outOfTime())
    {
        stopSearch = true;
        return evaluate();
    }

    bool inCheck = board.inCheck();

    if (!inCheck)
    {
        int standPat = evaluate();

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

        return 0;
    }

    for (const Move &move : moves)
    {
        if (stopSearch)
            break;
        if (!inCheck && !board.isCapture(move))
            continue;

        board.makeMove(move);

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
        return 0;
    }
    int alphaOrig = alpha;
    int betaOrig = beta;

    uint64_t key =
        board.hash();
    stats.ttProbe++;
    auto it =
        tt.find(key);

    if (it != tt.end())
        stats.ttHit++;

    if (it != tt.end() &&
        it->second.depth >= depth)
    {
        if (it->second.flag == EXACT)
        {
            return it->second.score;
        }

        if (it->second.flag == LOWERBOUND)
        {
            alpha =
                std::max(
                    alpha,
                    it->second.score);
        }

        if (it->second.flag == UPPERBOUND)
        {
            beta =
                std::min(
                    beta,
                    it->second.score);
        }

        if (alpha >= beta)
        {
            return it->second.score;
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

            board.makeMove(move);

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
            tt[key] =
                {
                    depth,
                    best,
                    flag,
                    bestMove};
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

            board.makeMove(move);

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
            tt[key] =
                {
                    depth,
                    best,
                    flag,
                    bestMove};
        }

        return best;
    }
}

Move search()
{
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

    for (currentDepth; currentDepth <= maxDepth;
         currentDepth++)
    {

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

            board.makeMove(
                move);

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
            tt.clear();

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
