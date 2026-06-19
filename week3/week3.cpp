#include<chess.hpp>
#include<unordered_map>
#include<iostream>
#include<string>
using namespace chess;

const int INF = 1e9;

const int MATE_SCORE = 2000000;

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


int evaluate(Board &board){
    int score = 0;
    for(int i=0; i<64; i++){
        Piece piece = board.at(Square(i));
        score += pieceValue(piece.type().internal()) * (piece.color() == Color::WHITE ? 1 : -1);
    }
    return score;
}

void orderMoves(Movelist &moves, Board &board){
    for(auto &move : moves){
        int score = 0;
        if(board.isCapture(move)){
            Piece victim = board.at(move.to());
            Piece attacker = board.at(move.from());
            score = 10 * pieceValue(victim.type()) - pieceValue(attacker.type());
        }
        move.setScore(score);
    }
    std::sort(moves.begin(), moves.end(), [](const Move &a, const Move &b){
        return a.score() > b.score();
    });
}

std::pair<int, Move> minimax(Board &board, int depth, int ply, int alpha, int beta, bool maximizing){
    Movelist moves;
    movegen::legalmoves(moves, board);

    orderMoves(moves, board);

    if(moves.empty()){
        if(board.inCheck()){
            if(maximizing)
                return {-MATE_SCORE + ply, Move::NULL_MOVE};
            else
                return {MATE_SCORE - ply, Move::NULL_MOVE};
        }
        return {0, Move::NULL_MOVE};
    }

    if(depth == 0){
        return {evaluate(board), Move::NULL_MOVE};
    }
    if(maximizing){
        std::pair<int, Move> best = {-INF, Move::NULL_MOVE};
        for(auto &move:moves){
            board.makeMove(move);
            std::pair<int, Move> result = minimax(board, depth-1, ply+1, alpha, beta, false);
            board.unmakeMove(move);
            if(result.first >= best.first){
                best.first = result.first;
                best.second = move;
            }
            alpha = std::max(alpha, best.first);
            if(beta <= alpha){
                break;
            }
        }
        return best;
    }
    else{
        std::pair<int, Move> best = {INF, Move::NULL_MOVE};
        for(auto &move:moves){
            board.makeMove(move);
            std::pair<int, Move> result = minimax(board, depth-1, ply+1, alpha, beta, true);
            board.unmakeMove(move);
            if(result.first <= best.first){
                best.first = result.first;
                best.second = move;
            }
            beta = std::min(beta, best.first);
            if(beta <= alpha){
                break;
            }
        }
        return best;
    }
}

int main(){
    std::cout << "Enter FEN and mate in N: ";
    std::string fen;
    getline(std::cin, fen);
    int mateInN;
    std::cin >> mateInN;
    Board board(fen);
    Movelist moves;
    movegen::legalmoves(moves, board);
    bool maximizing = board.sideToMove() == Color::WHITE;
    std::pair<int, Move> result = minimax(board, 2 * mateInN - 1, 0, -INF, INF, maximizing);
    std::cout << uci::moveToSan(board, result.second) << std::endl;
}
