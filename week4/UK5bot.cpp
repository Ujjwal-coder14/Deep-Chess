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

int moveBudget(int rem, int inc)
{
    int moveTime = rem / 30 + inc * 3 / 4;
    moveTime = std::min(moveTime, rem / 5);
    return std::max(10, moveTime);
}

bool outOfTime()
{
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart).count();
    return elapsed >= timeLimitMs;
}

std::unordered_map<uint64_t, TTEntry> tt;

int pieceValue(PieceType pt)
{
    switch (pt.internal())
    {
    case PieceType::PAWN:
        return 100;
    case PieceType::KNIGHT:
        return 300;
    case PieceType::BISHOP:
        return 300;
    case PieceType::ROOK:
        return 500;
    case PieceType::QUEEN:
        return 1000;
    default:
        return 0;
    }
}

void orderMoves(Movelist &moves)
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

            Piece victim =
                board.at(move.to());

            Piece attacker =
                board.at(move.from());

            score =
                10 * pieceValue(victim.type()) - pieceValue(attacker.type());
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

int evaluate()
{
    int score = 0;

    int whiteBishop = 0, blackBishop = 0;

    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));

        if (p == Piece::NONE)
            continue;

        if (p.type() == PieceType::BISHOP)
        {
            if (p.color() == Color::WHITE)
                whiteBishop++;
            else
                blackBishop++;
        }
    }

    for (int sq = 0; sq < 64; sq++)
    {
        Piece p = board.at(Square(sq));

        if (p == Piece::NONE)
            continue;

        int val = pieceValue(p.type());

        if (p.color() == Color::WHITE)
        {
            score += val;
            switch (p.type().internal())
            {
            case PieceType::PAWN:
                score += pawnTable[sq];
                break;
            case PieceType::KNIGHT:
                score += knightTable[sq];
                break;
            case PieceType::BISHOP:
                score += bishopTable[sq];
                break;
            case PieceType::ROOK:
                score += rookTable[sq];
                break;
            case PieceType::QUEEN:
                score += queenTable[sq];
                break;
            }
        }
        else
        {
            score -= val;
            switch (p.type().internal())
            {
            case PieceType::PAWN:
                score -= pawnTable[sq ^ 56];
                break;
            case PieceType::KNIGHT:
                score -= knightTable[sq ^ 56];
                break;
            case PieceType::BISHOP:
                score -= bishopTable[sq ^ 56];
                break;
            case PieceType::ROOK:
                score -= rookTable[sq ^ 56];
                break;
            case PieceType::QUEEN:
                score -= queenTable[sq ^ 56];
                break;
            }
        }
    }
    if(whiteBishop >= 2) score += 30;
    if(blackBishop >= 2) score -= 30;

    return score;
}

int minimax(int depth, int ply, int alpha, int beta, bool maximizingPlayer)
{
    if (outOfTime())
    {
        stopSearch = true;
        return 0;
    }
    int alphaOrig = alpha;
    int betaOrig = beta;

    uint64_t key =
        board.hash();

    auto it =
        tt.find(key);

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

    Movelist moves;
    movegen::legalmoves(moves, board);

    orderMoves(moves);

    if (moves.empty())
    {

        if (board.inCheck())
        {

            if (maximizingPlayer)
                return -MATE_SCORE + ply;
            else
                return MATE_SCORE - ply;
        }

        return 0;
    }

    if (depth == 0)
    {

        int eval =
            evaluate();

        tt[key] =
            {
                depth,
                eval,
                EXACT};

        return eval;
    }

    if (maximizingPlayer)
    {

        int best = -INF;

        Move bestMove = Move::NULL_MOVE;

        for (const Move &move : moves)
        {
            if (stopSearch)
                break;

            board.makeMove(move);

            int score = minimax(
                depth - 1,
                ply + 1,
                alpha,
                beta,
                false);

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
                break;
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

        for (const Move &move : moves)
        {
            if (stopSearch)
                break;

            board.makeMove(move);

            int score = minimax(
                depth - 1,
                ply + 1,
                alpha,
                beta,
                true);

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
                break;
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
    searchStart = std::chrono::steady_clock::now();
    stopSearch = false;

    Move bestMove =
        Move::NULL_MOVE;

    Move lastCompletedBestMove =
        Move::NULL_MOVE;

    bool maximizingPlayer =
        board.sideToMove() ==
        Color::WHITE;

    for (int currentDepth = 1;;
         currentDepth++)
    {

        Movelist moves;

        movegen::legalmoves(
            moves,
            board);

        orderMoves(moves);

        int bestScore =
            maximizingPlayer
                ? -INF
                : INF;

        for (const Move &move : moves)
        {
            if (stopSearch)
                break;

            board.makeMove(
                move);

            int score =
                minimax(
                    currentDepth - 1,
                    1,
                    -INF,
                    INF,
                    !maximizingPlayer);

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
            }
        }
        if (stopSearch)
        {
            break;
        }
        lastCompletedBestMove = bestMove;
    }

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
